// mmq_q6k_llama.cu — faithful llama.cpp Q6_K int8 MMQ port for genie prefill GEMM.
// Mirrors the Q4_K port (mmq_q4k_llama.cu): vendored mma.cuh + verbatim-extracted
// load_tiles_q6_K / vec_dot_q6_K_q8_1_mma + a non-stream-K mainloop + a q8_1 D4
// activation quantizer + launcher. Q6_K weights are symmetric (signed 6-bit, no
// min term) so activations use the D4 layout (d only). Kernels live in an
// anonymous namespace so they don't collide with the Q4_K TU under -rdc linking.
#include "llama_mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
using namespace ggml_cuda_mma;

namespace {  // internal linkage — avoid ODR clashes with mmq_q4k_llama.cu

// ---- ggml quant constants ----
#define QK_K 256
#define QK8_1 32
#define QR8_1 1
#define QI8_1 8
#define QK8_0 32
#define QI8_0 8
#define QR6_K 2
#define QI6_K 32

enum ggml_type { GGML_TYPE_Q6_K = 14 };

// block_q6_K: ql[128] (low 4 bits), qh[64] (high 2 bits), int8 scales[16], half d.
// Byte-identical to ggml's (== genie's, whose d is stored as uint16_t bits).
struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
};
static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 2, "block_q6_K size");

struct block_q8_1_mmq {
    union { float d4[4]; half2 ds4[4]; half d2s6[8]; };  // Q6_K uses d4 (D4 layout)
    int8_t qs[4*QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 16 + 128, "block_q8_1_mmq size");

// ---- MMQ tile macros ----
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)                          // 36
#define MMQ_MMA_TILE_X_K_Q6_K (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K + MMQ_TILE_NE_K/8 + 7) // 76
#define MMQ_ITER_K 256
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

// ---- device helpers ----
static constexpr __device__ int ggml_cuda_get_physical_warp_size() { return 32; }
static constexpr __device__ int mmq_get_nwarps_device() { return 256/ggml_cuda_get_physical_warp_size(); }
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

// stubs so the dead dp4a #else branch of load_tiles_q6_K parses (never instantiated
// under TURING_MMA_AVAILABLE).
struct tile_x_sizes { int qs, dm, sc; };
static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type, int) { return {0,0,0}; }

// ===== load_tiles_q6_K =====
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K]           = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + threadIdx.x%(MMQ_TILE_NE_K/8)] = get_int_b2(bxi->scales, threadIdx.x%(QI6_K/8));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ===== vec_dot_q6_K_q8_1_mma =====
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
    typedef tile<64,  2, int, input_layout>        tile_load;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_generic(((tile_load *) A)[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + k0, MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B[1];
            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1] / 2;

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B[0]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*MMQ_MMA_TILE_X_K_Q6_K + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*MMQ_MMA_TILE_X_K_Q6_K] * dB;
                }
            }
        }
    }
#elif defined(AMD_WMMA_AVAILABLE) //wmma instructions can handle 16x4 tiles, does not require loading 64x2 tiles
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_generic(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + k0, MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*MMQ_MMA_TILE_X_K_Q6_K + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*MMQ_MMA_TILE_X_K_Q6_K] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + 0),         MMQ_MMA_TILE_X_K_Q6_K);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + tile_A::J), MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q6_K];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
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

// ===================== Q6_K engine glue =====================
// Activation q8_1 D4 quantizer: x[N x K] half -> q8[(K/128) x Npad] block_q8_1_mmq,
// storing per-32 scale d as float d4[sub] (no partial sum — Q6_K is symmetric).
__global__ void quantize_mmq_q8_1_d4(const half* __restrict__ x,
                                     block_q8_1_mmq* __restrict__ y,
                                     int N, int K, int Npad) {
    const int col = blockIdx.x, kb = blockIdx.y, t = threadIdx.x;
    const int sub = t >> 5, lane = t & 31;
    block_q8_1_mmq& blk = y[(int64_t)kb*Npad + col];
    float v = 0.0f;
    if (col < N) v = __half2float(x[(int64_t)col*K + kb*128 + t]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax/127.0f, dinv = d > 0.0f ? 1.0f/d : 0.0f;
    blk.qs[sub*32 + lane] = (int8_t)__float2int_rn(v*dinv);
    if (lane == 0) blk.d4[sub] = d;
}

template <int mmq_x, int mmq_y, bool need_check>
__launch_bounds__(32*8, 2)
__global__ void mul_mat_q6k_llama(
        const char * __restrict__ x, const int * __restrict__ y, const int * __restrict__ ids_dst,
        float * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_dst,
        const int stride_y_cols) {
    constexpr int warp_size = 32, nwarps = 8, qk = QK_K, ne_block = 4*QK8_1;
    constexpr int sz = sizeof(block_q8_1_mmq)/sizeof(int);
    extern __shared__ int data_mul_mat_q6[];
    int * tile_y = data_mul_mat_q6 + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    const int row0 = blockIdx.y * mmq_y, col0 = blockIdx.x * mmq_x;
    const int tile_x_max_i = nrows_x   - row0 - 1;
    const int tile_y_max_j = ncols_dst - col0 - 1;
    const int offset_x = row0 * stride_row_x, nkb = stride_row_x;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    for (int kb0 = 0; kb0 < nkb; ++kb0) {
        load_tiles_q6_K<mmq_y, need_check>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {   const int * by0 = y + ((int64_t)stride_y_cols*(kb0*qk/ne_block) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            } }
        __syncthreads();
        vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
        __syncthreads();
        {   const int * by0 = y + ((int64_t)stride_y_cols*(kb0*qk/ne_block + 1) + col0) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                if (l < mmq_x*MMQ_TILE_Y_K) tile_y[l] = by0[l];
            } }
        __syncthreads();
        vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }
    mmq_write_back_mma<GGML_TYPE_Q6_K, mmq_x, mmq_y, need_check>(
        sum, ids_dst + col0, dst + row0, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

__global__ void f32_to_half_q6k(const float* __restrict__ src, half* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static block_q8_1_mmq* g6_q8 = nullptr; static size_t g6_q8_cap = 0;
static int*            g6_ids = nullptr; static int    g6_ids_cap = 0;
static float*          g6_cf  = nullptr; static size_t g6_cf_cap = 0;
static void ensure6(size_t q8_blocks, int npad, size_t cf_elems) {
    if (q8_blocks > g6_q8_cap) { if (g6_q8) cudaFree(g6_q8); cudaMalloc(&g6_q8, q8_blocks*sizeof(block_q8_1_mmq)); g6_q8_cap=q8_blocks; }
    if (npad > g6_ids_cap) { if (g6_ids) cudaFree(g6_ids); cudaMalloc(&g6_ids, npad*sizeof(int)); g6_ids_cap=npad;
        int* h=(int*)malloc(npad*sizeof(int)); for(int i=0;i<npad;i++) h[i]=i; cudaMemcpy(g6_ids,h,npad*sizeof(int),cudaMemcpyHostToDevice); free(h); }
    if (cf_elems > g6_cf_cap) { if (g6_cf) cudaFree(g6_cf); cudaMalloc(&g6_cf, cf_elems*sizeof(float)); g6_cf_cap=cf_elems; }
}

}  // anonymous namespace

namespace jllm {
// y[N x M] half row-major = x[N x K] half . W^T (Q6_K). W may be host-mapped.
void gemm_q6k_mmq_llama(half* y, const void* W, const half* x, int M, int N, int K, cudaStream_t stream) {
    constexpr int MX = 64, MY = 128;
    const int npad = ((N + MX - 1)/MX)*MX, nkb128 = K/128, nbk = K/256;
    const void* Wdev = W;
    { cudaPointerAttributes pa;
      if (cudaPointerGetAttributes(&pa, W) == cudaSuccess && pa.devicePointer != nullptr) Wdev = pa.devicePointer; }
    ensure6((size_t)nkb128*npad, npad, (size_t)N*M);
    if (npad > N) cudaMemsetAsync(g6_q8, 0, (size_t)nkb128*npad*sizeof(block_q8_1_mmq), stream);
    quantize_mmq_q8_1_d4<<<dim3(npad, nkb128), 128, 0, stream>>>(x, g6_q8, N, K, npad);

    const int nwarps=8, warp=32;
    size_t shmem = (size_t)(MX + GGML_PAD(MX*MMQ_TILE_Y_K, nwarps*warp) + MY*MMQ_MMA_TILE_X_K_Q6_K)*sizeof(int);
    dim3 grid(npad/MX, (M + MY - 1)/MY), block(warp, nwarps);
    if (M % MY == 0) {
        cudaFuncSetAttribute(mul_mat_q6k_llama<MX,MY,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        mul_mat_q6k_llama<MX,MY,false><<<grid,block,shmem,stream>>>((const char*)Wdev,(const int*)g6_q8,g6_ids,g6_cf,M,N,nbk,M,npad);
    } else {
        cudaFuncSetAttribute(mul_mat_q6k_llama<MX,MY,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        mul_mat_q6k_llama<MX,MY,true><<<grid,block,shmem,stream>>>((const char*)Wdev,(const int*)g6_q8,g6_ids,g6_cf,M,N,nbk,M,npad);
    }
    int64_t n = (int64_t)N*M;
    f32_to_half_q6k<<<(n+255)/256, 256, 0, stream>>>(g6_cf, y, n);
}
}  // namespace jllm
