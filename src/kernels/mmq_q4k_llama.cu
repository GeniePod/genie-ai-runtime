// mmq_q4k_llama.cu — faithful llama Q4_K int8 MMQ port for genie prefill GEMM.
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
#include "llama_mma.cuh"
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

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

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

template<ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, half * __restrict__ dst,
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

                dst[ids_dst[j]*stride + i] = __float2half(sum[(j0/tile_C::J + n)*tile_C::ne + l]);
            }
        }
    }
}

// ===================== engine glue: quantizer + driver + launcher =====================
// Activation q8_1 DS4 quantizer: x[N x K] half (row-major) -> q8[(K/128) x Npad]
// block_q8_1_mmq, the transposed layout mul_mat_q_simple consumes. One block per
// (token col, 128-K block); 4 warps, warp w handles 32-subblock w.
__global__ void quantize_mmq_q8_1_ds4(const half* __restrict__ x,
                                      block_q8_1_mmq* __restrict__ y,
                                      int N, int K, int Npad) {
    const int col = blockIdx.x;          // token
    const int kb  = blockIdx.y;          // 128-K block
    const int t   = threadIdx.x;         // 0..127
    const int sub = t >> 5;              // warp = subblock (0..3)
    const int lane= t & 31;
    block_q8_1_mmq& blk = y[(int64_t)kb*Npad + col];
    float v = 0.0f;
    if (col < N) v = __half2float(x[(int64_t)col*K + kb*128 + t]);
    float amax = fabsf(v), sum = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum  +=          __shfl_xor_sync(0xffffffff, sum,  o);
    }
    const float d = amax/127.0f;
    const float dinv = d > 0.0f ? 1.0f/d : 0.0f;
    blk.qs[sub*32 + lane] = (int8_t)__float2int_rn(v*dinv);
    if (lane == 0) { blk.ds4[sub].x = __float2half(d); blk.ds4[sub].y = __float2half(sum); }
}

// Simplified mul_mat_q (no stream-K/MoE/channels). stride_y_cols (= Npad) decouples
// the padded activation column stride from the real output width ncols_dst (= N).
template <int mmq_x, int mmq_y, bool need_check>
__launch_bounds__(32*8, 2)
__global__ void mul_mat_q_llama(
        const char * __restrict__ x, const int * __restrict__ y, const int * __restrict__ ids_dst,
        half * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_dst,
        const int stride_y_cols) {
    constexpr int warp_size = 32, nwarps = 8, qk = QK_K, ne_block = 4*QK8_1;
    constexpr int sz = sizeof(block_q8_1_mmq)/sizeof(int);
    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    const int row0 = blockIdx.y * mmq_y;
    const int col0 = blockIdx.x * mmq_x;
    const int tile_x_max_i = nrows_x   - row0 - 1;
    const int tile_y_max_j = ncols_dst - col0 - 1;
    const int offset_x = row0 * stride_row_x;
    const int nkb = stride_row_x;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    for (int kb0 = 0; kb0 < nkb; ++kb0) {
        load_tiles_q4_K<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {   const int * by0 = y + ((int64_t)stride_y_cols*(kb0*qk/ne_block) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            } }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
        __syncthreads();
        {   const int * by0 = y + ((int64_t)stride_y_cols*(kb0*qk/ne_block + 1) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            } }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }
    mmq_write_back_mma<GGML_TYPE_Q4_K, mmq_x, mmq_y, need_check>(
        sum, ids_dst + col0, dst + row0, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

// ---- lazy device scratch (q8 activations + identity ids), grown on demand ----
static block_q8_1_mmq* g_q8 = nullptr;  static size_t g_q8_cap = 0;
static int*            g_ids = nullptr; static int    g_ids_cap = 0;

static void mmq_ck(cudaError_t e, const char* what, size_t bytes) {
    if (e != cudaSuccess) {
        size_t fr=0, tot=0; cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "[mmqllama] %s cudaMalloc(%zu B) FAILED: %s (free=%zu/%zu)\n",
                what, bytes, cudaGetErrorString(e), fr, tot);
    }
}
static void ensure_scratch(size_t q8_blocks, int npad) {
    if (q8_blocks > g_q8_cap) { if (g_q8) cudaFree(g_q8);
        mmq_ck(cudaMalloc(&g_q8, q8_blocks*sizeof(block_q8_1_mmq)), "q8", q8_blocks*sizeof(block_q8_1_mmq)); g_q8_cap=q8_blocks; }
    if (npad > g_ids_cap) {
        if (g_ids) cudaFree(g_ids); mmq_ck(cudaMalloc(&g_ids, npad*sizeof(int)), "ids", npad*sizeof(int)); g_ids_cap=npad;
        int* h=(int*)malloc(npad*sizeof(int)); for(int i=0;i<npad;i++) h[i]=i;
        cudaMemcpy(g_ids,h,npad*sizeof(int),cudaMemcpyHostToDevice); free(h);
    }
}

// Launch mul_mat_q_llama for a compile-time tile (MX cols × MY rows). npad must
// be a multiple of MX (caller pads N up to MX). need_check guards partial M tiles.
template <int MX, int MY>
static void launch_mmq_tile(const void* Wdev, const int* g_q8, const int* g_ids,
                            half* y, int M, int N, int nbk, int npad, cudaStream_t stream) {
    constexpr int nwarps = 8, warp = 32;
    size_t shmem = (size_t)(MX + GGML_PAD(MX*MMQ_TILE_Y_K, nwarps*warp) + MY*MMQ_MMA_TILE_X_K_Q8_1)*sizeof(int);
    dim3 grid(npad/MX, (M + MY - 1)/MY), block(warp, nwarps);
    if (M % MY == 0) {
        cudaFuncSetAttribute(mul_mat_q_llama<MX,MY,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        mul_mat_q_llama<MX,MY,false><<<grid,block,shmem,stream>>>((const char*)Wdev, g_q8, g_ids, y, M, N, nbk, M, npad);
    } else {
        cudaFuncSetAttribute(mul_mat_q_llama<MX,MY,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        mul_mat_q_llama<MX,MY,true><<<grid,block,shmem,stream>>>((const char*)Wdev, g_q8, g_ids, y, M, N, nbk, M, npad);
    }
}

// Tile (MX×MY) selection. Default 64×128 (the shipped config). JLLM_MMQ_TILE
// overrides for in-engine autotuning sweeps: 0=64x128, 1=128x64, 2=128x128,
// 3=64x64. mmq_x (=MX, the token-tile width) amortizes the weight unpack over
// more columns; on the 8-SM Orin bigger tiles trade occupancy for amortization.
static int mmq_tile_choice() {
    static const int t = [] {
        const char* v = getenv("JLLM_MMQ_TILE");
        return v ? atoi(v) : 0;
    }();
    return t;
}

// Engine-facing launcher. y[N x M] half row-major = x[N x K] half · W^T (Q4_K).
// In namespace jllm to match genie's gemv_q4.cu forward declaration.
namespace jllm {
void gemm_q4k_mmq_llama(half* y, const void* W, const half* x,
                        int M, int N, int K, cudaStream_t stream) {
    const int choice = mmq_tile_choice();
    const int MX = (choice == 0 || choice == 3) ? 64 : 128;
    const int npad = ((N + MX - 1)/MX)*MX;
    const int nkb128 = K/128;

    // genie keeps quantized weights in cudaHostRegister'd mmap memory, whose
    // device-accessible address differs from the host VA. Passing the host VA to
    // a kernel faults; translate to the device pointer. (For device/managed
    // memory devicePointer == W, so this is a no-op there.)
    const void* Wdev = W;
    { cudaPointerAttributes pa;
      if (cudaPointerGetAttributes(&pa, W) == cudaSuccess && pa.devicePointer != nullptr)
          Wdev = pa.devicePointer; }
    ensure_scratch((size_t)nkb128*npad, npad);

    // quantize activations -> g_q8 (zero padding cols so they contribute nothing)
    if (npad > N) cudaMemsetAsync(g_q8, 0, (size_t)nkb128*npad*sizeof(block_q8_1_mmq), stream);
    dim3 qgrid(npad, nkb128); // padding cols (col>=N) write zero d/qs via the col<N guard
    quantize_mmq_q8_1_ds4<<<qgrid, 128, 0, stream>>>(x, g_q8, N, K, npad);

    const int nbk = K/256;
    // write_back emits half directly into y (no f32 staging buffer / convert kernel).
    const int* q8 = (const int*)g_q8;
    switch (choice) {
        case 1:  launch_mmq_tile<128, 64 >(Wdev, q8, g_ids, y, M, N, nbk, npad, stream); break;
        case 2:  launch_mmq_tile<128, 128>(Wdev, q8, g_ids, y, M, N, nbk, npad, stream); break;
        case 3:  launch_mmq_tile<64,  64 >(Wdev, q8, g_ids, y, M, N, nbk, npad, stream); break;
        default: launch_mmq_tile<64,  128>(Wdev, q8, g_ids, y, M, N, nbk, npad, stream); break;
    }
}
}  // namespace jllm
