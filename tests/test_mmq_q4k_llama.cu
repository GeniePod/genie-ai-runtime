// Standalone reference/validation for the llama Q4_K int8 MMQ port.
// Build: nvcc -arch=sm_87 -O3 -I../src/kernels tests/test_mmq_q4k_llama.cu -o t && ./t [M N K]
// Validates load_tiles_q4_K + vec_dot_q8_1_q8_1_mma vs a CPU reference
// (rel-RMS ~3e-4) and reports INT8 tensor-core GOP/s. Not in CMake.
// test_mmq_q4k_llama.cu — Phase 2 standalone validation of the faithful
// llama.cpp Q4_K int8 MMQ port for genie on Jetson Orin (SM 8.7).
//
// Structure:
//   [p1] ggml types/macros/constants + device helpers (hand-written shim)
//   [extracted verbatim from llama mmq.cuh] unpack_scales_q45_K, load_tiles_q4_K,
//       vec_dot_q8_1_q8_1_mma, mmq_write_back_mma
//   [p3] simplified (no stream-K/MoE/channel) mul_mat_q driver + host harness:
//       CPU Q4_K weights, CPU q8_1 activation quant (DS4), exact-mirror CPU ref,
//       GPU run, correctness compare, GOP/s timing.
#include "../src/kernels/llama_mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
using namespace ggml_cuda_mma;

// ---- ggml quant constants (ggml-common.h) ----
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32
#define QR8_0 1
#define QI8_0 8
#define QK8_1 32
#define QR8_1 1
#define QI8_1 8
#define QR4_K 2
#define QI4_K 32

typedef uint16_t ggml_half;

// minimal ggml_type — only Q4_K matters; value is irrelevant for the MMA path
// (the dp4a #else branches that switch on it are preprocessed out under TURING).
enum ggml_type { GGML_TYPE_Q4_K = 12 };

// Flattened (no anonymous union — half/half2 have ctors, disallowed there).
// The kernel reads block_q4_K only via ->dm (half2) and treats activation y as
// raw int*, so these layouts are byte-identical to ggml's for our path.
struct block_q4_K {
    half2   dm;                   // (d, dmin)
    uint8_t scales[K_SCALE_SIZE]; // 6-bit scales + mins
    uint8_t qs[QK_K/2];           // 128 bytes, 256 4-bit quants
};
static_assert(sizeof(block_q4_K) == 4 + K_SCALE_SIZE + QK_K/2, "block_q4_K size");

struct block_q8_1_mmq {
    half2  ds4[4];      // d0,s0,d1,s1,d2,s2,d3,s3  (DS4 layout, used by Q4_K)
    int8_t qs[4*QK8_1]; // 128 int8
};
static_assert(sizeof(block_q8_1_mmq) == 16 + 128, "block_q8_1_mmq size");

// ---- MMQ tile macros (mmq.cuh) ----
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)                  // 36
#define MMQ_MMA_TILE_X_K_Q8_1 (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4) // 76
#define MMQ_ITER_K 256
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

// ---- device helpers (common.cuh / mmq.cuh, NVIDIA SM 8.7 path) ----
static constexpr __device__ int ggml_cuda_get_physical_warp_size() { return 32; }
static constexpr __device__ int mmq_get_nwarps_device() { return 256/ggml_cuda_get_physical_warp_size(); } // 8
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) { return mmq_x >= 48 ? 16 : 8; }

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

// Stubs so the (dead, never-instantiated) dp4a #else branch of load_tiles_q4_K
// still parses. Under -arch=sm_87 TURING_MMA_AVAILABLE is defined, so the MMA
// path is the only one ever instantiated/code-generated.
struct tile_x_sizes { int qs, dm, sc; };
static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type, int) { return {0,0,0}; }

// ===== unpack_scales_q45_K =====
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

// ===== load_tiles_q4_K =====
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < mmq_y) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// ===== vec_dot_q8_1_q8_1_mma =====
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_generic(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    float2 dmA = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.x*dsB.x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.y*dsB.y;
                }
            }
        }
    }
#else
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// ===== mmq_write_back_mma =====
template<ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int tileC_IJ = mmq_get_granularity_device(0);
    typedef tile<tileC_IJ, tileC_IJ, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int rows_per_warp = granularity;
#else
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
#endif // defined(AMD_MFMA_AVAILABLE)
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");
#else
    GGML_UNUSED(nwarps);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

// ===================== [p3] simplified driver + host harness =====================

// Simplified mul_mat_q: no stream-K, no MoE ids, no multi-channel. One block per
// (mmq_y rows x mmq_x cols) output tile; loops all K-blocks. Inner body mirrors
// llama's mul_mat_q_process_tile exactly (shared layout, double tile_y load,
// two vec_dot halves), which is what we are validating.
#ifndef MINBLK
#define MINBLK 2   // 2 blocks/SM (128 regs * 256 thr = 32768 = half of 65536) -> 33% occ
#endif
template <int mmq_x, int mmq_y, bool need_check>
__launch_bounds__(32*8, MINBLK)
__global__ void mul_mat_q_simple(
        const char * __restrict__ x, const int * __restrict__ y, const int * __restrict__ ids_dst,
        float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_dst) {
    constexpr int warp_size = 32;
    constexpr int nwarps    = 8;
    constexpr int qk        = QK_K;          // 256
    constexpr int ne_block  = 4*QK8_1;       // 128
    constexpr int sz        = sizeof(block_q8_1_mmq)/sizeof(int); // 36

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    const int row0 = blockIdx.y * mmq_y;
    const int col0 = blockIdx.x * mmq_x;
    const int tile_x_max_i = nrows_x   - row0 - 1;
    const int tile_y_max_j = ncols_dst - col0 - 1;
    const int offset_x = row0 * stride_row_x;
    const int nkb = stride_row_x;            // 256-blocks per weight row = K/256

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    for (int kb0 = 0; kb0 < nkb; ++kb0) {
        load_tiles_q4_K<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        {   // first 128-K half
            const int * by0 = y + (ncols_dst*(kb0*qk/ne_block) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
        __syncthreads();

        {   // second 128-K half
            const int * by0 = y + (ncols_dst*(kb0*qk/ne_block + 1) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_mma<GGML_TYPE_Q4_K, mmq_x, mmq_y, need_check>(
        sum, ids_dst + col0, dst + row0, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

// ---------------- host helpers ----------------
#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float h2f(half h){ return __half2float(h); }
static half  f2h(float f){ return __float2half(f); }

// ggml Q4_K scale/min extraction (reference)
static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63;                      *m = q[j+4] & 63; }
    else       { *d = (q[j+4]&0xF) | ((q[j-4]>>6)<<4); *m = (q[j+4]>>4)  | ((q[j-0]>>6)<<4); }
}

// run + validate + time one mmq_x tiling
template <int MX>
static bool run_mmq(const char* dW, const int* dY, const int* dIds, float* dC,
                    int M, int N, int K, int nbk, const std::vector<float>& Cref) {
    const int mmq_y=128, nwarps=8, warp=32;
    size_t shmem = (size_t)(MX + GGML_PAD(MX*MMQ_TILE_Y_K, nwarps*warp) + mmq_y*MMQ_MMA_TILE_X_K_Q8_1)*sizeof(int);
    auto kern = mul_mat_q_simple<MX,128,false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    dim3 grid(N/MX, M/mmq_y), block(warp,nwarps);
    kern<<<grid,block,shmem>>>(dW,dY,dIds,dC,M,N,nbk,M);
    CK(cudaDeviceSynchronize());
    std::vector<float> C((size_t)N*M);
    CK(cudaMemcpy(C.data(), dC, (size_t)N*M*sizeof(float), cudaMemcpyDeviceToHost));
    double sumsq=0, refsq=0;
    for (size_t i=0;i<(size_t)N*M;i++){ double rf=Cref[i]; refsq+=rf*rf; }
    double sigrms=sqrt(refsq/((double)N*M)), maxrel=0; int nbad=0;
    for (size_t i=0;i<(size_t)N*M;i++){ double ad=fabs((double)C[i]-Cref[i]); double rel=ad/(fabs(Cref[i])+1e-6);
        if(rel>maxrel)maxrel=rel; if(rel>0.02&&ad>0.05*sigrms)nbad++; sumsq+=ad*ad; }
    double relrms=sqrt(sumsq/(refsq+1e-9)); bool ok=(relrms<0.01)&&(nbad==0);
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int w=0;w<3;w++) kern<<<grid,block,shmem>>>(dW,dY,dIds,dC,M,N,nbk,M);
    CK(cudaDeviceSynchronize());
    const int iters=50; cudaEventRecord(e0);
    for(int i=0;i<iters;i++) kern<<<grid,block,shmem>>>(dW,dY,dIds,dC,M,N,nbk,M);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=iters;
    double gops=2.0*(double)M*N*K/1e9/(ms/1e3);
    printf("  mmq_x=%-3d grid=(%2d,%2d) shmem=%3.0fKB | %s relRMS=%.5f nbad=%d | %.3fms  %.0f GOP/s  (%.1f%%)\n",
           MX, grid.x, grid.y, shmem/1024.0, ok?"PASS":"FAIL", relrms, nbad, ms, gops, gops/32400.0*100.0);
    return ok;
}

int main(int argc, char** argv) {
    const int mmq_x = 64, mmq_y = 128;
    int M = 1024;             // weight rows (output feature dim), multiple of mmq_y
    int N = 256;              // tokens, multiple of mmq_x
    int K = 2048;             // multiple of 256
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    M = ((M + mmq_y - 1)/mmq_y)*mmq_y;
    N = ((N + mmq_x - 1)/mmq_x)*mmq_x;
    K = ((K + 255)/256)*256;
    const int nbk = K/256;        // q4_K blocks per row
    const int nkb128 = K/128;     // q8_1 128-blocks
    printf("M(rows)=%d N(tokens)=%d K=%d  (nbk=%d)\n", M, N, K, nbk);

    srand(1234);
    // ---- random Q4_K weights: [M rows][nbk blocks] ----
    std::vector<block_q4_K> W((size_t)M*nbk);
    for (auto& b : W) {
        b.dm.x = f2h(0.02f + 0.01f*(rand()/(float)RAND_MAX));   // d
        b.dm.y = f2h(0.01f + 0.01f*(rand()/(float)RAND_MAX));   // dmin
        for (int i=0;i<K_SCALE_SIZE;i++) b.scales[i] = rand() & 0xFF;
        for (int i=0;i<QK_K/2;i++)       b.qs[i]     = rand() & 0xFF;
    }
    // ---- random activations: [N tokens][K] float ----
    std::vector<float> A((size_t)N*K);
    for (auto& a : A) a = (rand()/(float)RAND_MAX)*2.f - 1.f;

    // ---- CPU q8_1 quant -> y[nkb128][N] block_q8_1_mmq (DS4) ----
    // also keep dequant info for the exact-mirror reference.
    std::vector<block_q8_1_mmq> Y((size_t)nkb128*N);
    std::vector<float> dact((size_t)N*(K/32)), sumact((size_t)N*(K/32));
    std::vector<int8_t> q8((size_t)N*K);
    for (int c=0;c<N;c++) {
        for (int s=0; s<K/32; s++) {              // 32-subblock index
            float amax=0.f, ssum=0.f;
            for (int t=0;t<32;t++){ float v=A[(size_t)c*K + s*32 + t]; amax=fmaxf(amax,fabsf(v)); ssum+=v; }
            float d = amax/127.f; float dinv = d>0? 1.f/d : 0.f;
            dact[(size_t)c*(K/32)+s]=d; sumact[(size_t)c*(K/32)+s]=ssum;
            int kb128 = s/4, sub = s%4;           // which 128-block, which sub of 4
            block_q8_1_mmq& blk = Y[(size_t)kb128*N + c];
            blk.ds4[sub].x = f2h(d); blk.ds4[sub].y = f2h(ssum);
            for (int t=0;t<32;t++){ int8_t qq=(int8_t)lroundf(A[(size_t)c*K+s*32+t]*dinv); blk.qs[sub*32+t]=qq; q8[(size_t)c*K+s*32+t]=qq; }
        }
    }

    // ---- exact-mirror CPU reference: C[c*M + r] ----
    std::vector<float> Cref((size_t)N*M);
    for (int r=0;r<M;r++) {
        for (int c=0;c<N;c++) {
            double acc=0.0;
            for (int kb=0; kb<nbk; kb++) {
                const block_q4_K& b = W[(size_t)r*nbk + kb];
                float d=h2f(b.dm.x), dmin=h2f(b.dm.y);
                for (int s=0;s<8;s++){               // 8 subblocks of 32 in the 256-block
                    uint8_t sc6,m6; get_scale_min_k4(s, b.scales, &sc6, &m6);
                    int g=s/2; int dot=0;
                    for (int t=0;t<32;t++){
                        int q4 = (s%2==0) ? (b.qs[g*32+t]&0xF) : (b.qs[g*32+t]>>4);
                        int gs = kb*8 + s;           // global 32-subblock
                        dot += q4 * (int)q8[(size_t)c*K + gs*32 + t];
                    }
                    int gs = kb*8 + s;
                    acc += (double)(d*sc6) * dact[(size_t)c*(K/32)+gs] * dot
                         - (double)(dmin*m6) * sumact[(size_t)c*(K/32)+gs];
                }
            }
            Cref[(size_t)c*M + r] = (float)acc;
        }
    }

    // ---- GPU buffers ----
    char* dW; int* dY; int* dIds; float* dC;
    CK(cudaMalloc(&dW, W.size()*sizeof(block_q4_K)));
    CK(cudaMalloc(&dY, Y.size()*sizeof(block_q8_1_mmq)));
    CK(cudaMalloc(&dIds, N*sizeof(int)));
    CK(cudaMalloc(&dC, (size_t)N*M*sizeof(float)));
    CK(cudaMemcpy(dW, W.data(), W.size()*sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dY, Y.data(), Y.size()*sizeof(block_q8_1_mmq), cudaMemcpyHostToDevice));
    std::vector<int> ids(N); for(int i=0;i<N;i++) ids[i]=i;
    CK(cudaMemcpy(dIds, ids.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    // sweep mmq_x (column-tile width); larger -> higher arithmetic intensity
    bool ok = true;
    if (N% 64==0) ok &= run_mmq< 64>(dW,dY,dIds,dC, M,N,K,nbk, Cref);
    if (N% 96==0) ok &= run_mmq< 96>(dW,dY,dIds,dC, M,N,K,nbk, Cref);
    if (N%128==0) ok &= run_mmq<128>(dW,dY,dIds,dC, M,N,K,nbk, Cref);
    return ok?0:2;
}
