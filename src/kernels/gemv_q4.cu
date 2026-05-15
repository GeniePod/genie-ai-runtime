// gemv_q4.cu — Q4_K dequant-fused GEMV for Orin SM 8.7
// Matches llama.cpp's exact Q4_K block layout and dequant formula.

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace jllm {

static constexpr int QK_K = 256;

static bool debug_kernels_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DEBUG_KERNELS");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool fast_gemv_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_GEMV");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Path C: vectorized weight loads for the Q4_K family of decode gemv
// kernels (Wo / gate-up pair / QKV triple). Lane T reads four packed
// q-bytes (one uint32 from blk.qs) instead of one byte, eliminating
// the il inner loop and folding scales into per-lane constants. See
// dot_q4k_row_uint32 below for the new lane layout.
//
// Default-on after #25 / #26 / #27 each shipped bit-identical output
// against the byte path on Qwen3-4B and cumulatively pushed decode
// from 7.5 → 8.9 tok/s (+18.7%) on Jetson Orin Nano Super. Set
// JLLM_Q4K_UINT32_LOADS=0 to opt back into the byte-by-byte kernels
// (e.g. for an A/B comparison or if a future model trips an
// alignment assumption).
static bool q4k_uint32_loads_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_Q4K_UINT32_LOADS");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Path E (#33): tensor-core MMQ kernel for Q4_K prefill GEMM. Replaces
// gemm_quant_batched_q4k_kernel's scalar FMAs with
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 against a 16×32 FP16
// staging tile dequantized into shared memory. Two iterations:
//
//   - E3 (#36): naive port. ~74 GFLOPS standalone on Wo; integrated
//     regressed prefill 27 % (#37, closed). 139 regs/thread, 32×
//     redundant per-(row, sub-block) scale arithmetic across the warp.
//
//   - E4 (#38): per-(row, sub-block) scale precompute into shared
//     memory + drop `#pragma unroll` on the dequant inner row loop.
//     96 regs, ≥ 210 GFLOPS on every Qwen3-4B prefill shape, ~245
//     GFLOPS aggregate — about 2× the scalar baseline. This PR
//     integrates E4's kernel.
//
// Default-on after #39 shipped +70.8 % prefill (16.6 → 28.3 tok/s) on
// Jetson Orin Nano Super 8 GB with sensibly-identical output text vs
// alpha.6 scalar path. Set JLLM_MMQ_Q4K=0 to opt back into the scalar
// batched Q4_K kernel (same grammar as JLLM_BATCHED_PREFILL and
// JLLM_Q4K_UINT32_LOADS).
static bool mmq_q4k_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_MMQ_Q4K");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

static const void* resolve_weight_device_ptr(const void* W) {
    if (const void* mapped = resolve_mapped_weight_device_ptr(W)) {
        return mapped;
    }

    cudaPointerAttributes attr{};
    cudaError_t err = cudaPointerGetAttributes(&attr, W);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error for ordinary host pointers
        return nullptr;
    }

#if CUDART_VERSION >= 10000
    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        return W;
    }
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
        return W;
    }
#endif

    return nullptr;
}

static int gemv_rows_per_block() {
    static const int rows = [] {
        const char* v = getenv("JLLM_GEMV_ROWS");
        int r = v ? atoi(v) : 4;
        if (r != 4 && r != 8) r = 4;
        return r;
    }();
    return rows;
}

// Use raw uint16 instead of half2 to guarantee no padding
struct __attribute__((packed)) block_q4_K {
    uint16_t d_raw;       // FP16 super-block scale
    uint16_t dmin_raw;    // FP16 super-block min
    uint8_t  scales[12];  // packed 6-bit sub-block scales and mins
    uint8_t  qs[QK_K/2]; // 4-bit quants (128 bytes)
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

struct __attribute__((packed)) block_q5_K {
    uint16_t d_raw;
    uint16_t dmin_raw;
    uint8_t  scales[12];
    uint8_t  qh[QK_K/8];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_q5_K) == 176, "Q5_K block must be 176 bytes");

struct __attribute__((packed)) block_q6_K {
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    uint16_t d_raw;
};
static_assert(sizeof(block_q6_K) == 210, "Q6_K block must be 210 bytes");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        result = mant ? NAN : INFINITY;
    } else {
        result = ldexpf((float)(mant + 1024), (int)exp - 25);
    }
    return sign ? -result : result;
#endif
}

__host__ __device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

__device__ __forceinline__ float dot_q4k_row(
    const block_q4_K* __restrict__ row_blocks,
    const half*       __restrict__ x,
    int n_blocks,
    int lane)
{
    float acc = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = row_blocks[b];

        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        for (int il = 0; il < 4; il++) {
            const int is = 2 * il;

            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            const float d1 = dall * sc1;
            const float dm1 = dmin * m1;
            const float d2 = dall * sc2;
            const float dm2 = dmin * m2;

            const uint8_t* q = blk.qs + 32 * il;
            const int k_lo = k_base + 64 * il + lane;
            const int k_hi = k_lo + 32;

            const float w_lo = d1 * (q[lane] & 0xF) - dm1;
            const float w_hi = d2 * (q[lane] >> 4)  - dm2;

            acc += w_lo * __half2float(x[k_lo]);
            acc += w_hi * __half2float(x[k_hi]);
        }
    }

    return acc;
}

// uint32-load variant of dot_q4k_row.
//
// Lane partitioning (new):
//   il       = lane >> 3           (which 32-byte qs sub-block: 0..3)
//   sub_base = (lane & 7) << 2     (byte offset within il: 0,4,...,28)
//   Each lane reads four contiguous q-bytes as one uint32 from
//   blk.qs + 32*il + sub_base, eliminating the il inner loop. Eight
//   FMAs per block per lane — same total as the byte-by-byte path,
//   but issued as one ld.global.b32 + 8 ld.global.b16 instead of
//   four ld.global.b8 + 8 ld.global.b16. Plus zero inner-loop control.
//
// Per warp instruction:
//   weight: 32 lanes × 4 bytes = 128 bytes = 1 L1 line covering full qs
//   x:      32 lanes × 2 halves × 4 sub-positions = 256 halves = 4 L1 lines
//   compared to byte path (4 inner iters):
//   weight: 4 × (32 × 1 byte) = 4 partial-line transactions
//   x:      4 × (32 × 2 halves) = same 4 lines but issued 4x
//
// The cache-line accesses for weights drop from 4 → 1 per block (the
// hypothesis behind #19's Q4_K-vs-Q6_K arithmetic-intensity gap).
//
// Float ordering: each lane's accumulator now sums 8 FMAs that touch
// K-positions [64*il+sub_base, 64*il+sub_base+4) low + [+32, +36) high
// instead of [lane, +32, +64, +96, +128, +160, +192, +224]. Reduction
// across 32 lanes still covers the full block but the addition order
// differs, so bit-equality with dot_q4k_row is not guaranteed (same
// caveat as #24's split-K).
__device__ __forceinline__ float dot_q4k_row_uint32(
    const block_q4_K* __restrict__ row_blocks,
    const half*       __restrict__ x,
    int n_blocks,
    int lane)
{
    const int il       = lane >> 3;
    const int sub_base = (lane & 7) << 2;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = row_blocks[b];
        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        // Scales for this lane's il — constant across the 4 packed bytes.
        const int is = 2 * il;
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4(is + 0, blk.scales, sc1, m1);
        get_scale_min_k4(is + 1, blk.scales, sc2, m2);

        const float d1  = dall * sc1;
        const float dm1 = dmin * m1;
        const float d2  = dall * sc2;
        const float dm2 = dmin * m2;

        // One uint32 load: 4 packed q-bytes.
        // blk.qs starts at byte 16 of block_q4_K (after d_raw + dmin_raw +
        // scales[12]); 32*il is 32-aligned; sub_base is 4-aligned → the
        // computed address is at least 4-aligned, so the uint32 access is
        // legal even on strict-alignment ARM64.
        const uint32_t qv32 =
            *(const uint32_t*)(blk.qs + 32 * il + sub_base);

        const int k_lo_base = k_base + 64 * il + sub_base;
        // k_hi_base = k_lo_base + 32

        #pragma unroll
        for (int s = 0; s < 4; s++) {
            const uint8_t qb = (uint8_t)(qv32 >> (s * 8));
            const float w_lo = d1 * (qb & 0xF) - dm1;
            const float w_hi = d2 * (qb >> 4)  - dm2;
            acc += w_lo * __half2float(x[k_lo_base + s]);
            acc += w_hi * __half2float(x[k_lo_base + s + 32]);
        }
    }

    return acc;
}

__device__ __forceinline__ float dot_q5k_row(
    const block_q5_K* __restrict__ row_blocks,
    const half*       __restrict__ x,
    int n_blocks,
    int lane)
{
    float acc = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q5_K& blk = row_blocks[b];

        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (int il = 0; il < 4; il++) {
            const int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            const float d1 = dall * sc1;
            const float dm1 = dmin * m1;
            const float d2 = dall * sc2;
            const float dm2 = dmin * m2;

            const uint8_t* ql = blk.qs + 32 * il;
            const uint8_t* qh = blk.qh;
            const int k_lo = k_base + 64 * il + lane;
            const int k_hi = k_lo + 32;

            const float w_lo = d1 * ((ql[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0)) - dm1;
            const float w_hi = d2 * ((ql[lane] >> 4)  + ((qh[lane] & u2) ? 16 : 0)) - dm2;

            acc += w_lo * __half2float(x[k_lo]);
            acc += w_hi * __half2float(x[k_hi]);

            u1 <<= 2;
            u2 <<= 2;
        }
    }

    return acc;
}

__device__ __forceinline__ float dot_q6k_row(
    const block_q6_K* __restrict__ row_blocks,
    const half*       __restrict__ x,
    int n_blocks,
    int lane)
{
    float acc = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q6_K& blk = row_blocks[b];
        const float d = raw_fp16_to_float(blk.d_raw);
        const int k_base = b * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t* ql = blk.ql + (n / 128) * 64;
            const uint8_t* qh = blk.qh + (n / 128) * 32;
            const int8_t*  sc = blk.scales + (n / 128) * 8;
            const int is = lane / 16;

            const int q1 = (int)((ql[lane +  0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[lane +  0] >>  4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[lane + 32] >>  4) | (((qh[lane] >> 6) & 3) << 4)) - 32;

            const int k0 = k_base + n + lane;
            acc += d * (float)sc[is + 0] * (float)q1 * __half2float(x[k0 +  0]);
            acc += d * (float)sc[is + 2] * (float)q2 * __half2float(x[k0 + 32]);
            acc += d * (float)sc[is + 4] * (float)q3 * __half2float(x[k0 + 64]);
            acc += d * (float)sc[is + 6] * (float)q4 * __half2float(x[k0 + 96]);
        }
    }

    return acc;
}

__device__ __forceinline__ float dot_quant_row(
    const void* __restrict__ W,
    int ggml_type,
    int row,
    int n_blocks,
    const half* __restrict__ x,
    int lane)
{
    switch (ggml_type) {
        case 12:
            return dot_q4k_row((const block_q4_K*)W + (int64_t)row * n_blocks,
                               x, n_blocks, lane);
        case 13:
            return dot_q5k_row((const block_q5_K*)W + (int64_t)row * n_blocks,
                               x, n_blocks, lane);
        case 14:
            return dot_q6k_row((const block_q6_K*)W + (int64_t)row * n_blocks,
                               x, n_blocks, lane);
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float acc) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    return acc;
}

__global__ void gemv_q4k_kernel(
    half*              __restrict__ y,
    const block_q4_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q4k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = __float2half(acc);
}

__global__ void gemv_q5k_kernel(
    half*              __restrict__ y,
    const block_q5_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q5k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = __float2half(acc);
}

__global__ void gemv_q6k_kernel(
    half*              __restrict__ y,
    const block_q6_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q6k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = __float2half(acc);
}

__global__ void gemv_q4k_f32_kernel(
    float*             __restrict__ y,
    const block_q4_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q4k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = acc;
}

__global__ void gemv_q5k_f32_kernel(
    float*             __restrict__ y,
    const block_q5_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q5k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = acc;
}

__global__ void gemv_q6k_f32_kernel(
    float*             __restrict__ y,
    const block_q6_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q6k_row(W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);

    if (lane == 0)
        y[row] = acc;
}

__global__ void dequant_embedding_q4k_kernel(
    half*              __restrict__ dst,
    const block_q4_K*  __restrict__ W,
    int token_id,
    int hidden_dim)
{
    const int block_id = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= QK_K) return;

    const int blocks_per_row = hidden_dim / QK_K;
    const block_q4_K& blk = W[(int64_t)token_id * blocks_per_row + block_id];
    const float dall = raw_fp16_to_float(blk.d_raw);
    const float dmin = raw_fp16_to_float(blk.dmin_raw);

    const int il = i >> 6;
    const int lane64 = i & 63;
    const int lane32 = lane64 & 31;
    const int is = 2 * il + (lane64 >= 32);

    uint8_t sc, mn;
    get_scale_min_k4(is, blk.scales, sc, mn);

    const uint8_t q = blk.qs[32 * il + lane32];
    const int qv = (lane64 < 32) ? (q & 0x0F) : (q >> 4);
    const float val = dall * (float)sc * (float)qv - dmin * (float)mn;
    dst[block_id * QK_K + i] = __float2half(val);
}

__global__ void dequant_embedding_q6k_kernel(
    half*              __restrict__ dst,
    const block_q6_K*  __restrict__ W,
    int token_id,
    int hidden_dim)
{
    const int block_id = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= QK_K) return;

    const int blocks_per_row = hidden_dim / QK_K;
    const block_q6_K& blk = W[(int64_t)token_id * blocks_per_row + block_id];
    const float d = raw_fp16_to_float(blk.d_raw);

    const int half_block = i >> 7;
    const int local = i & 127;
    const int lane = local & 31;
    const int quarter = local >> 5;

    const uint8_t* ql = blk.ql + half_block * 64;
    const uint8_t* qh = blk.qh + half_block * 32;
    const int8_t* sc = blk.scales + half_block * 8;
    const int is = lane >> 4;

    int qv;
    int scale;
    if (quarter == 0) {
        qv = (int)(ql[lane] & 0x0F) | (((int)(qh[lane] >> 0) & 3) << 4);
        scale = sc[is + 0];
    } else if (quarter == 1) {
        qv = (int)(ql[lane + 32] & 0x0F) | (((int)(qh[lane] >> 2) & 3) << 4);
        scale = sc[is + 2];
    } else if (quarter == 2) {
        qv = (int)(ql[lane] >> 4) | (((int)(qh[lane] >> 4) & 3) << 4);
        scale = sc[is + 4];
    } else {
        qv = (int)(ql[lane + 32] >> 4) | (((int)(qh[lane] >> 6) & 3) << 4);
        scale = sc[is + 6];
    }

    const float val = d * (float)scale * (float)(qv - 32);
    dst[block_id * QK_K + i] = __float2half(val);
}

__global__ void gemv_quant_pair_kernel(
    half*        __restrict__ y0,
    const void*  __restrict__ W0,
    int type0,
    int M0,
    half*        __restrict__ y1,
    const void*  __restrict__ W1,
    int type1,
    int M1,
    const half*  __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int total_M = M0 + M1;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    half* y = nullptr;
    const void* W = nullptr;
    int type = 0;
    int local_row = row;

    if (row < M0) {
        y = y0;
        W = W0;
        type = type0;
    } else {
        y = y1;
        W = W1;
        type = type1;
        local_row = row - M0;
    }

    float acc = dot_quant_row(W, type, local_row, n_blocks, x, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0)
        y[local_row] = __float2half(acc);
}

__global__ void gemv_quant_triple_kernel(
    half*        __restrict__ y0,
    const void*  __restrict__ W0,
    int type0,
    int M0,
    half*        __restrict__ y1,
    const void*  __restrict__ W1,
    int type1,
    int M1,
    half*        __restrict__ y2,
    const void*  __restrict__ W2,
    int type2,
    int M2,
    const half*  __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int total_M = M0 + M1 + M2;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    half* y = nullptr;
    const void* W = nullptr;
    int type = 0;
    int local_row = row;

    if (row < M0) {
        y = y0;
        W = W0;
        type = type0;
    } else if (row < M0 + M1) {
        y = y1;
        W = W1;
        type = type1;
        local_row = row - M0;
    } else {
        y = y2;
        W = W2;
        type = type2;
        local_row = row - M0 - M1;
    }

    float acc = dot_quant_row(W, type, local_row, n_blocks, x, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0)
        y[local_row] = __float2half(acc);
}

template<int Type>
__device__ __forceinline__ float dot_quant_row_t(
    const void* __restrict__ W,
    int row,
    int n_blocks,
    const half* __restrict__ x,
    int lane)
{
    if constexpr (Type == 12) {
        return dot_q4k_row((const block_q4_K*)W + (int64_t)row * n_blocks,
                           x, n_blocks, lane);
    } else if constexpr (Type == 13) {
        return dot_q5k_row((const block_q5_K*)W + (int64_t)row * n_blocks,
                           x, n_blocks, lane);
    } else if constexpr (Type == 14) {
        return dot_q6k_row((const block_q6_K*)W + (int64_t)row * n_blocks,
                           x, n_blocks, lane);
    } else {
        return 0.0f;
    }
}

template<int Type>
__global__ void gemv_quant_add_typed_kernel(
    half*        __restrict__ y,
    const void*  __restrict__ W,
    const half*  __restrict__ x,
    const half*  __restrict__ residual,
    int M,
    int K,
    int rows_per_block)
{
    const int row = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_quant_row_t<Type>(W, row, n_blocks, x, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const half projected = __float2half(acc);
        y[row] = __float2half(__half2float(projected) +
                              __half2float(residual[row]));
    }
}

// Q4_K-only variant of gemv_quant_add_typed_kernel using
// dot_q4k_row_uint32 (uint32 weight loads, no il inner loop).
// Same block shape and grid as the byte path, drop-in dispatch from
// gemv_quant_add_gpu's case 12 when JLLM_Q4K_UINT32_LOADS is set.
__global__ void gemv_quant_add_uint32_q4k_kernel(
    half*             __restrict__ y,
    const block_q4_K* __restrict__ W,
    const half*       __restrict__ x,
    const half*       __restrict__ residual,
    int M, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    float acc = dot_q4k_row_uint32(
        W + (int64_t)row * n_blocks, x, n_blocks, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const half projected = __float2half(acc);
        y[row] = __float2half(__half2float(projected) +
                              __half2float(residual[row]));
    }
}

template<int Type0, int Type1>
__global__ void gemv_quant_pair_typed_kernel(
    half*        __restrict__ y0,
    const void*  __restrict__ W0,
    int M0,
    half*        __restrict__ y1,
    const void*  __restrict__ W1,
    int M1,
    const half*  __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int total_M = M0 + M1;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    float acc;
    if (row < M0) {
        acc = dot_quant_row_t<Type0>(W0, row, n_blocks, x, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y0[row] = __float2half(acc);
    } else {
        const int local_row = row - M0;
        acc = dot_quant_row_t<Type1>(W1, local_row, n_blocks, x, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y1[local_row] = __float2half(acc);
    }
}

// Q4_K + Q4_K dual-output variant using dot_q4k_row_uint32 (the same
// uint32-weight-load helper added for the Wo path in #25). Used for
// decode gate/up, which is ~44% of decode wall-clock on Qwen3-4B per
// #19. Both outputs use the same `x`, so the inner dot helper's work
// is per-row and identical to the single-output case.
__global__ void gemv_quant_pair_uint32_q4k_kernel(
    half*             __restrict__ y0,
    const block_q4_K* __restrict__ W0,
    int M0,
    half*             __restrict__ y1,
    const block_q4_K* __restrict__ W1,
    int M1,
    const half*       __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row     = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int total_M = M0 + M1;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    if (row < M0) {
        float acc = dot_q4k_row_uint32(
            W0 + (int64_t)row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y0[row] = __float2half(acc);
    } else {
        const int local_row = row - M0;
        float acc = dot_q4k_row_uint32(
            W1 + (int64_t)local_row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y1[local_row] = __float2half(acc);
    }
}

// Q4_K + Q4_K + Q4_K triple-output uint32 kernel. Used by QKV when all
// three matrices are Q4_K (some model configs). Same pattern as the
// pair kernel: warp owns one row of W0/W1/W2 based on row range, all
// three use dot_q4k_row_uint32.
__global__ void gemv_quant_triple_uint32_q4k_kernel(
    half*             __restrict__ y0,
    const block_q4_K* __restrict__ W0,
    int M0,
    half*             __restrict__ y1,
    const block_q4_K* __restrict__ W1,
    int M1,
    half*             __restrict__ y2,
    const block_q4_K* __restrict__ W2,
    int M2,
    const half*       __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row     = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int total_M = M0 + M1 + M2;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    if (row < M0) {
        float acc = dot_q4k_row_uint32(
            W0 + (int64_t)row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y0[row] = __float2half(acc);
    } else if (row < M0 + M1) {
        const int local_row = row - M0;
        float acc = dot_q4k_row_uint32(
            W1 + (int64_t)local_row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y1[local_row] = __float2half(acc);
    } else {
        const int local_row = row - M0 - M1;
        float acc = dot_q4k_row_uint32(
            W2 + (int64_t)local_row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y2[local_row] = __float2half(acc);
    }
}

// Q4_K + Q4_K + Q6_K triple-output kernel. This is the Qwen3-4B case
// (Wq, Wk are Q4_K; Wv is Q6_K). The Q4_K rows take the uint32 path;
// the Q6_K rows still go through the byte-by-byte dot_q6k_row helper
// because we haven't written a uint32 variant for Q6_K yet (Q6_K
// already runs at 30-40% of peak — less room to win, different block
// layout). The block of 32 rows is heterogeneous: some warps in a
// block run dot_q4k_row_uint32 and others run dot_q6k_row. That's
// fine — warps are independent and the lane mapping for each helper
// is self-contained.
__global__ void gemv_quant_triple_uint32_q4k_q4k_q6k_kernel(
    half*             __restrict__ y0,
    const block_q4_K* __restrict__ W0,
    int M0,
    half*             __restrict__ y1,
    const block_q4_K* __restrict__ W1,
    int M1,
    half*             __restrict__ y2,
    const block_q6_K* __restrict__ W2,
    int M2,
    const half*       __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row     = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int total_M = M0 + M1 + M2;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    if (row < M0) {
        float acc = dot_q4k_row_uint32(
            W0 + (int64_t)row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y0[row] = __float2half(acc);
    } else if (row < M0 + M1) {
        const int local_row = row - M0;
        float acc = dot_q4k_row_uint32(
            W1 + (int64_t)local_row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y1[local_row] = __float2half(acc);
    } else {
        const int local_row = row - M0 - M1;
        float acc = dot_q6k_row(
            W2 + (int64_t)local_row * n_blocks, x, n_blocks, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y2[local_row] = __float2half(acc);
    }
}

template<int Type0, int Type1, int Type2>
__global__ void gemv_quant_triple_typed_kernel(
    half*        __restrict__ y0,
    const void*  __restrict__ W0,
    int M0,
    half*        __restrict__ y1,
    const void*  __restrict__ W1,
    int M1,
    half*        __restrict__ y2,
    const void*  __restrict__ W2,
    int M2,
    const half*  __restrict__ x,
    int K,
    int rows_per_block)
{
    const int row = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int total_M = M0 + M1 + M2;
    if (row >= total_M) return;

    const int n_blocks = K / QK_K;
    float acc;
    if (row < M0) {
        acc = dot_quant_row_t<Type0>(W0, row, n_blocks, x, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y0[row] = __float2half(acc);
    } else if (row < M0 + M1) {
        const int local_row = row - M0;
        acc = dot_quant_row_t<Type1>(W1, local_row, n_blocks, x, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y1[local_row] = __float2half(acc);
    } else {
        const int local_row = row - M0 - M1;
        acc = dot_quant_row_t<Type2>(W2, local_row, n_blocks, x, lane);
        acc = warp_reduce_sum(acc);
        if (lane == 0) y2[local_row] = __float2half(acc);
    }
}

// Debug kernel: print first few output values
__global__ void debug_print_half(const half* data, int n, const char* label) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU %s] first 8: ", label);
        for (int i = 0; i < 8 && i < n; i++)
            printf("%.4f ", __half2float(data[i]));
        printf("\n");
    }
}

void gemv_q4(half* y, const void* W_q4, const half* scales, const half* x,
             int M, int K, int group_size, cudaStream_t stream) {
    (void)scales;
    (void)group_size;
    gemv_quant(y, W_q4, 12, x, M, K, stream);
}

static bool gemv_quant_gpu(half* y, const void* W, int ggml_type,
                           const half* x, int M, int K, cudaStream_t stream) {
    const void* W_device = resolve_weight_device_ptr(W);
    if (!W_device) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "[GEMV] CUDA-visible mapped weight pointer is unavailable; "
                    "falling back to CPU reference GEMV\n");
            warned = true;
        }
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M + rows_per_block - 1) / rows_per_block;

    switch (ggml_type) {
        case 12:
            gemv_q4k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q4_K*)W_device, x, M, K, rows_per_block);
            break;
        case 13:
            gemv_q5k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q5_K*)W_device, x, M, K, rows_per_block);
            break;
        case 14:
            gemv_q6k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q6_K*)W_device, x, M, K, rows_per_block);
            break;
        default:
            fprintf(stderr, "[GEMV] FATAL: unsupported GPU GGML type %d (M=%d K=%d)\n",
                    ggml_type, M, K);
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] GPU launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool gemv_quant_gpu_f32(float* y, const void* W, int ggml_type,
                               const half* x, int M, int K, cudaStream_t stream) {
    const void* W_device = resolve_weight_device_ptr(W);
    if (!W_device) {
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M + rows_per_block - 1) / rows_per_block;

    switch (ggml_type) {
        case 12:
            gemv_q4k_f32_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q4_K*)W_device, x, M, K, rows_per_block);
            break;
        case 13:
            gemv_q5k_f32_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q5_K*)W_device, x, M, K, rows_per_block);
            break;
        case 14:
            gemv_q6k_f32_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q6_K*)W_device, x, M, K, rows_per_block);
            break;
        default:
            fprintf(stderr, "[GEMV] FATAL: unsupported GPU GGML type %d (M=%d K=%d)\n",
                    ggml_type, M, K);
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] GPU FP32 launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool supported_gpu_type(int ggml_type) {
    return ggml_type == 12 || ggml_type == 13 || ggml_type == 14;
}

static bool gemv_quant_pair_gpu(
    half* y0, const void* W0, int type0, int M0,
    half* y1, const void* W1, int type1, int M1,
    const half* x, int K, cudaStream_t stream)
{
    const void* W0_device = resolve_weight_device_ptr(W0);
    const void* W1_device = resolve_weight_device_ptr(W1);
    if (!W0_device || !W1_device ||
        !supported_gpu_type(type0) || !supported_gpu_type(type1)) {
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M0 + M1 + rows_per_block - 1) / rows_per_block;

    if (type0 == 12 && type1 == 12 && q4k_uint32_loads_enabled()) {
        // Q4_K + Q4_K uint32 weight-load path (decode gate/up). Same env
        // var as the residual-fused Wo path so a single flag covers
        // every uint32-accelerated Q4_K caller.
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[GEMV] Q4_K uint32 pair weight-load path enabled\n");
            logged = true;
        }
        gemv_quant_pair_uint32_q4k_kernel<<<grid, block, 0, stream>>>(
            y0, (const block_q4_K*)W0_device, M0,
            y1, (const block_q4_K*)W1_device, M1,
            x, K, rows_per_block);
        cudaError_t err_sk = cudaGetLastError();
        if (err_sk == cudaSuccess) return true;
        fprintf(stderr, "[GEMV] Q4_K uint32 pair launch failed: %s; falling back\n",
                cudaGetErrorString(err_sk));
        // Fall through to the typed byte-load kernel below.
    }

    if (type0 == 12 && type1 == 12) {
        gemv_quant_pair_typed_kernel<12, 12><<<grid, block, 0, stream>>>(
            y0, W0_device, M0, y1, W1_device, M1, x, K, rows_per_block);
    } else {
        gemv_quant_pair_kernel<<<grid, block, 0, stream>>>(
            y0, W0_device, type0, M0,
            y1, W1_device, type1, M1,
            x, K, rows_per_block);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] pair GPU launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool gemv_quant_triple_gpu(
    half* y0, const void* W0, int type0, int M0,
    half* y1, const void* W1, int type1, int M1,
    half* y2, const void* W2, int type2, int M2,
    const half* x, int K, cudaStream_t stream)
{
    const void* W0_device = resolve_weight_device_ptr(W0);
    const void* W1_device = resolve_weight_device_ptr(W1);
    const void* W2_device = resolve_weight_device_ptr(W2);
    if (!W0_device || !W1_device || !W2_device ||
        !supported_gpu_type(type0) || !supported_gpu_type(type1) ||
        !supported_gpu_type(type2)) {
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M0 + M1 + M2 + rows_per_block - 1) / rows_per_block;

    // Q4_K uint32 triple paths — same env-var family as #25 / #26.
    if (q4k_uint32_loads_enabled() &&
        type0 == 12 && type1 == 12 && (type2 == 12 || type2 == 14)) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr,
                    "[GEMV] Q4_K uint32 triple weight-load path enabled (V=%s)\n",
                    type2 == 14 ? "Q6_K (byte path)" : "Q4_K");
            logged = true;
        }
        if (type2 == 12) {
            gemv_quant_triple_uint32_q4k_kernel<<<grid, block, 0, stream>>>(
                y0, (const block_q4_K*)W0_device, M0,
                y1, (const block_q4_K*)W1_device, M1,
                y2, (const block_q4_K*)W2_device, M2,
                x, K, rows_per_block);
        } else { // type2 == 14
            gemv_quant_triple_uint32_q4k_q4k_q6k_kernel<<<grid, block, 0, stream>>>(
                y0, (const block_q4_K*)W0_device, M0,
                y1, (const block_q4_K*)W1_device, M1,
                y2, (const block_q6_K*)W2_device, M2,
                x, K, rows_per_block);
        }
        cudaError_t err_uint32 = cudaGetLastError();
        if (err_uint32 == cudaSuccess) return true;
        fprintf(stderr,
                "[GEMV] Q4_K uint32 triple launch failed: %s; falling back\n",
                cudaGetErrorString(err_uint32));
        // Fall through to the typed byte-load kernel below.
    }

    if (type0 == 12 && type1 == 12 && type2 == 14) {
        gemv_quant_triple_typed_kernel<12, 12, 14><<<grid, block, 0, stream>>>(
            y0, W0_device, M0, y1, W1_device, M1, y2, W2_device, M2,
            x, K, rows_per_block);
    } else if (type0 == 12 && type1 == 12 && type2 == 12) {
        gemv_quant_triple_typed_kernel<12, 12, 12><<<grid, block, 0, stream>>>(
            y0, W0_device, M0, y1, W1_device, M1, y2, W2_device, M2,
            x, K, rows_per_block);
    } else {
        gemv_quant_triple_kernel<<<grid, block, 0, stream>>>(
            y0, W0_device, type0, M0,
            y1, W1_device, type1, M1,
            y2, W2_device, type2, M2,
            x, K, rows_per_block);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] triple GPU launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool gemv_quant_add_gpu(half* y, const void* W, int ggml_type,
                               const half* x, const half* residual,
                               int M, int K, cudaStream_t stream) {
    const void* W_device = resolve_weight_device_ptr(W);
    if (!W_device || !supported_gpu_type(ggml_type)) {
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M + rows_per_block - 1) / rows_per_block;

    // Q4_K only — uint32 weight loads, opt-in via JLLM_Q4K_UINT32_LOADS.
    // On Q5_K / Q6_K the per-byte loop already runs at higher arithmetic
    // intensity per byte; vectorizing those is a separate exercise.
    if (ggml_type == 12 && q4k_uint32_loads_enabled()) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[GEMV] Q4_K uint32 weight-load path enabled\n");
            logged = true;
        }
        gemv_quant_add_uint32_q4k_kernel<<<grid, block, 0, stream>>>(
            y, (const block_q4_K*)W_device, x, residual, M, K, rows_per_block);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return true;
        fprintf(stderr, "[GEMV] Q4_K uint32 launch failed: %s; falling back\n",
                cudaGetErrorString(err));
        // Fall through to the byte-load typed kernel below.
    }

    switch (ggml_type) {
        case 12:
            gemv_quant_add_typed_kernel<12><<<grid, block, 0, stream>>>(
                y, W_device, x, residual, M, K, rows_per_block);
            break;
        case 13:
            gemv_quant_add_typed_kernel<13><<<grid, block, 0, stream>>>(
                y, W_device, x, residual, M, K, rows_per_block);
            break;
        case 14:
            gemv_quant_add_typed_kernel<14><<<grid, block, 0, stream>>>(
                y, W_device, x, residual, M, K, rows_per_block);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] add GPU launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ── Batched K-quant GEMM (prefill path) ─────────────────────────────────
//
// y[N×M] = x[N×K] · Wᵀ[K×M] where W is Q4_K/Q5_K/Q6_K.
//
// Memory-traffic principle: each warp owns one output row r. For every
// weight value loaded (once per block-iteration) we accumulate N partial
// sums (one per token). The weights are read from DRAM once and reused
// N times — N=18 on a 15-token prompt + Qwen3 chat-template wrapping,
// which is exactly the regime where per-token gemv is bandwidth-bound.
//
// Each thread stores a fixed-size acc[MAX_BATCH] register array; the
// host dispatcher chunks N > MAX_BATCH into multiple kernel launches.

// Sized just above the typical chat-template-wrapped prompt length we serve
// from GenieClaw (N ≈ 18 for a single user turn). The token-loop is
// #pragma-unrolled, so every iteration past N is predicated-off but still
// consumes issue slots — shrinking the unroll factor cuts that waste.
// Longer prompts get chunked by the host dispatcher.
constexpr int GEMM_MAX_BATCH = 20;

// The token-loop inside the kernel is #pragma-unrolled so the compiler can
// keep acc[] in registers — a runtime-bounded loop with dynamic indexing
// forces a spill to local memory and erases the bandwidth win.

__global__ void gemm_quant_batched_q4k_kernel(
    half*             __restrict__ y,         // [N × M] row-major
    const block_q4_K* __restrict__ W,         // [M × n_blocks]
    const half*       __restrict__ x,         // [N × K] row-major
    int M, int N, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q4_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc[GEMM_MAX_BATCH];
    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) acc[t] = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = row_blocks[b];
        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        for (int il = 0; il < 4; il++) {
            const int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            const float d1 = dall * sc1;
            const float dm1 = dmin * m1;
            const float d2 = dall * sc2;
            const float dm2 = dmin * m2;

            const uint8_t qv = blk.qs[32 * il + lane];
            const float w_lo = d1 * (qv & 0xF) - dm1;
            const float w_hi = d2 * (qv >> 4)  - dm2;

            const int k_lo = k_base + 64 * il + lane;
            const int k_hi = k_lo + 32;

            #pragma unroll
            for (int t = 0; t < GEMM_MAX_BATCH; t++) {
                if (t < N) {
                    const half* xt = x + (int64_t)t * K;
                    acc[t] += w_lo * __half2float(xt[k_lo]);
                    acc[t] += w_hi * __half2float(xt[k_hi]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) {
        if (t < N) {
            float a = warp_reduce_sum(acc[t]);
            if (lane == 0) y[(int64_t)t * M + row] = __float2half(a);
        }
    }
}

__global__ void gemm_quant_batched_q5k_kernel(
    half*             __restrict__ y,
    const block_q5_K* __restrict__ W,
    const half*       __restrict__ x,
    int M, int N, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q5_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc[GEMM_MAX_BATCH];
    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) acc[t] = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q5_K& blk = row_blocks[b];
        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        uint8_t u1 = 1, u2 = 2;
        for (int il = 0; il < 4; il++) {
            const int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);
            const float d1 = dall * sc1;
            const float dm1 = dmin * m1;
            const float d2 = dall * sc2;
            const float dm2 = dmin * m2;

            const uint8_t ql = blk.qs[32 * il + lane];
            const uint8_t qh = blk.qh[lane];
            const float w_lo = d1 * ((ql & 0xF) + ((qh & u1) ? 16 : 0)) - dm1;
            const float w_hi = d2 * ((ql >> 4)  + ((qh & u2) ? 16 : 0)) - dm2;

            const int k_lo = k_base + 64 * il + lane;
            const int k_hi = k_lo + 32;

            #pragma unroll
            for (int t = 0; t < GEMM_MAX_BATCH; t++) {
                if (t < N) {
                    const half* xt = x + (int64_t)t * K;
                    acc[t] += w_lo * __half2float(xt[k_lo]);
                    acc[t] += w_hi * __half2float(xt[k_hi]);
                }
            }
            u1 <<= 2; u2 <<= 2;
        }
    }

    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) {
        if (t < N) {
            float a = warp_reduce_sum(acc[t]);
            if (lane == 0) y[(int64_t)t * M + row] = __float2half(a);
        }
    }
}

__global__ void gemm_quant_batched_q6k_kernel(
    half*             __restrict__ y,
    const block_q6_K* __restrict__ W,
    const half*       __restrict__ x,
    int M, int N, int K, int rows_per_block)
{
    const int row  = blockIdx.x * rows_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q6_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc[GEMM_MAX_BATCH];
    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) acc[t] = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q6_K& blk = row_blocks[b];
        const float d = raw_fp16_to_float(blk.d_raw);
        const int k_base = b * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t* ql = blk.ql + (n / 128) * 64;
            const uint8_t* qh = blk.qh + (n / 128) * 32;
            const int8_t*  sc = blk.scales + (n / 128) * 8;
            const int is = lane / 16;

            const int q1 = (int)((ql[lane +  0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[lane +  0] >>  4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[lane + 32] >>  4) | (((qh[lane] >> 6) & 3) << 4)) - 32;

            const float w1 = d * (float)sc[is + 0] * (float)q1;
            const float w2 = d * (float)sc[is + 2] * (float)q2;
            const float w3 = d * (float)sc[is + 4] * (float)q3;
            const float w4 = d * (float)sc[is + 6] * (float)q4;

            const int k0 = k_base + n + lane;
            #pragma unroll
            for (int t = 0; t < GEMM_MAX_BATCH; t++) {
                if (t < N) {
                    const half* xt = x + (int64_t)t * K;
                    acc[t] += w1 * __half2float(xt[k0 +  0]);
                    acc[t] += w2 * __half2float(xt[k0 + 32]);
                    acc[t] += w3 * __half2float(xt[k0 + 64]);
                    acc[t] += w4 * __half2float(xt[k0 + 96]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < GEMM_MAX_BATCH; t++) {
        if (t < N) {
            float a = warp_reduce_sum(acc[t]);
            if (lane == 0) y[(int64_t)t * M + row] = __float2half(a);
        }
    }
}

// Path E E5: multi-warp cooperative MMQ Q4_K kernel. Replaces the E4
// single-warp variant. 4 warps per CUDA block share a single
// dequanted 16×32 FP16 A-tile across 4 contiguous N-stripes (8 tokens
// each). Dequant amortized 4× vs E4; register count 96 → 39; all
// Qwen3-4B prefill shapes uniform around ~407 GFLOPS (vs E4's 210–276).
//
// Validated standalone in #41 (test_mmq_q4k_multiwarp.cu).
// Grid: (ceil(N / MMQ_Q4K_BLOCK_N), ceil(M / MMQ_Q4K_TILE_M), 1).
// Block: 4 warps × 32 threads = 128 threads.

constexpr int MMQ_Q4K_TILE_M  = 16;
constexpr int MMQ_Q4K_TILE_N  = 8;
constexpr int MMQ_Q4K_N_WARPS = 4;
constexpr int MMQ_Q4K_BLOCK_N = MMQ_Q4K_TILE_N * MMQ_Q4K_N_WARPS;   // 32

__global__ void gemm_mmq_q4k_kernel(half*             __restrict__ y,
                                    const block_q4_K* __restrict__ W,
                                    const half*       __restrict__ x,
                                    int M, int N, int K)
{
    const int row_base     = blockIdx.y * MMQ_Q4K_TILE_M;
    const int blk_tok_base = blockIdx.x * MMQ_Q4K_BLOCK_N;
    if (row_base >= M) return;

    const int t_id    = threadIdx.x;          // 0..127
    const int warp_id = t_id >> 5;            // 0..3
    const int lane    = t_id & 31;
    const int groupID = lane >> 2;
    const int tinG    = lane &  3;

    const int tok_base = blk_tok_base + warp_id * MMQ_Q4K_TILE_N;

    const int n_blocks = K / QK_K;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    __shared__ half  A_tile[MMQ_Q4K_TILE_M][32];
    __shared__ float per_d [MMQ_Q4K_TILE_M][8];
    __shared__ float per_dm[MMQ_Q4K_TILE_M][8];

    for (int b = 0; b < n_blocks; b++) {
        // Step 1: 128 (d, dm) pairs across 128 threads, 1 entry each.
        {
            const int row = t_id >> 3;        // 0..15
            const int sb  = t_id &  7;        // 0..7
            const int g_row = row_base + row;
            float d = 0.0f, dm = 0.0f;
            if (g_row < M) {
                const block_q4_K& blk = W[(int64_t)g_row * n_blocks + b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                uint8_t sc, mn;
                get_scale_min_k4(sb, blk.scales, sc, mn);
                d  = dall * sc;
                dm = dmin * mn;
            }
            per_d [row][sb] = d;
            per_dm[row][sb] = dm;
        }
        __syncthreads();

        // Step 2: 8 sub-blocks × {cooperative dequant, 4 MMAs}.
        for (int sb = 0; sb < 8; sb++) {
            const int il     = sb >> 1;
            const int parity = sb &  1;

            // Cooperative dequant: 128 threads × 4 elements = 512 = 16×32.
            // Thread t covers row = t/8, cols (t&7)*4 + [0..3] — keeps
            // each thread on one row across its 4 cols.
            {
                const int row    = t_id >> 3;
                const int col_4  = (t_id & 7) << 2;
                const int g_row  = row_base + row;
                float d = 0.0f, dm = 0.0f;
                const block_q4_K* blk_ptr = nullptr;
                if (g_row < M) {
                    blk_ptr = &W[(int64_t)g_row * n_blocks + b];
                    d  = per_d [row][sb];
                    dm = per_dm[row][sb];
                }
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    const int col = col_4 + s;
                    if (g_row < M) {
                        const uint8_t qb = blk_ptr->qs[32 * il + col];
                        const float val  = parity
                            ? (d * (qb >> 4)  - dm)
                            : (d * (qb & 0xF) - dm);
                        A_tile[row][col] = __float2half(val);
                    } else {
                        A_tile[row][col] = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();

            const int k_block_base = b * QK_K + sb * 32;

            #pragma unroll
            for (int km = 0; km < 2; km++) {
                const int ka = km * 16;

                half2 a0_h = *reinterpret_cast<const half2*>(&A_tile[groupID    ][ka + tinG * 2]);
                half2 a1_h = *reinterpret_cast<const half2*>(&A_tile[groupID + 8][ka + tinG * 2]);
                half2 a2_h = *reinterpret_cast<const half2*>(&A_tile[groupID    ][ka + tinG * 2 + 8]);
                half2 a3_h = *reinterpret_cast<const half2*>(&A_tile[groupID + 8][ka + tinG * 2 + 8]);

                uint32_t a0 = *reinterpret_cast<uint32_t*>(&a0_h);
                uint32_t a1 = *reinterpret_cast<uint32_t*>(&a1_h);
                uint32_t a2 = *reinterpret_cast<uint32_t*>(&a2_h);
                uint32_t a3 = *reinterpret_cast<uint32_t*>(&a3_h);

                const int g_tok = tok_base + groupID;
                const int k_pos = k_block_base + ka;

                uint32_t b0, b1;
                if (g_tok < N) {
                    half2 b0_h = *reinterpret_cast<const half2*>(&x[(int64_t)g_tok * K + k_pos + tinG * 2]);
                    half2 b1_h = *reinterpret_cast<const half2*>(&x[(int64_t)g_tok * K + k_pos + tinG * 2 + 8]);
                    b0 = *reinterpret_cast<uint32_t*>(&b0_h);
                    b1 = *reinterpret_cast<uint32_t*>(&b1_h);
                } else {
                    b0 = 0;
                    b1 = 0;
                }

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1)
                );
            }
            __syncthreads();
        }
    }

    const int tok0  = tok_base + tinG * 2;
    const int tok1  = tok_base + tinG * 2 + 1;
    const int row_a = row_base + groupID;
    const int row_b = row_base + groupID + 8;

    if (row_a < M) {
        if (tok0 < N) y[(int64_t)tok0 * M + row_a] = __float2half(d0);
        if (tok1 < N) y[(int64_t)tok1 * M + row_a] = __float2half(d1);
    }
    if (row_b < M) {
        if (tok0 < N) y[(int64_t)tok0 * M + row_b] = __float2half(d2);
        if (tok1 < N) y[(int64_t)tok1 * M + row_b] = __float2half(d3);
    }
}

static bool gemm_quant_batched_gpu(half* y, const void* W, int ggml_type,
                                   const half* x, int M, int N, int K,
                                   cudaStream_t stream) {
    const void* W_device = resolve_weight_device_ptr(W);
    if (!W_device || !supported_gpu_type(ggml_type)) {
        return false;
    }

    const int rows_per_block = gemv_rows_per_block();
    const int block = rows_per_block * 32;
    const int grid = (M + rows_per_block - 1) / rows_per_block;

    // Path E E5: Q4_K only, default on. Multi-warp cooperative
    // variant — 4 warps share one dequanted A-tile across 4 N-stripes.
    // Set JLLM_MMQ_Q4K=0 to opt back into the scalar batched Q4_K
    // kernel below. Falls through to the scalar kernel on any
    // cudaError (mirrors the Path C fallback after the #29 Q6_K
    // crash). Q5_K / Q6_K continue through the existing scalar
    // batched kernels.
    if (ggml_type == 12 && mmq_q4k_enabled()) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            fprintf(stderr,
                    "[GEMM] Path E MMQ Q4_K tensor-core kernel active "
                    "(multi-warp, default on; set JLLM_MMQ_Q4K=0 to "
                    "disable)\n");
        }
        dim3 mmq_grid((N + MMQ_Q4K_BLOCK_N - 1) / MMQ_Q4K_BLOCK_N,
                      (M + MMQ_Q4K_TILE_M  - 1) / MMQ_Q4K_TILE_M,  1);
        gemm_mmq_q4k_kernel<<<mmq_grid, MMQ_Q4K_N_WARPS * 32, 0, stream>>>(
            y, (const block_q4_K*)W_device, x, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return true;
        }
        fprintf(stderr,
                "[GEMM] MMQ Q4_K launch failed (%s), falling back to "
                "scalar batched kernel\n",
                cudaGetErrorString(err));
    }

    switch (ggml_type) {
        case 12:
            gemm_quant_batched_q4k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q4_K*)W_device, x, M, N, K, rows_per_block);
            break;
        case 13:
            gemm_quant_batched_q5k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q5_K*)W_device, x, M, N, K, rows_per_block);
            break;
        case 14:
            gemm_quant_batched_q6k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q6_K*)W_device, x, M, N, K, rows_per_block);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMM] batched GPU launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

void gemm_quant_batched(half* y, const void* W, int ggml_type, const half* x,
                        int M, int N, int K, cudaStream_t stream) {
    // N=1 is just gemv — keep the decode/per-token path bit-identical by
    // routing there. Anything > GEMM_MAX_BATCH is chunked.
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (N == 1) {
        gemv_quant(y, W, ggml_type, x, M, K, stream);
        return;
    }
    if (fast_gemv_enabled()) {
        int processed = 0;
        bool ok = true;
        while (processed < N && ok) {
            int chunk = N - processed;
            if (chunk > GEMM_MAX_BATCH) chunk = GEMM_MAX_BATCH;
            ok = gemm_quant_batched_gpu(
                y + (int64_t)processed * M,
                W, ggml_type,
                x + (int64_t)processed * K,
                M, chunk, K, stream);
            processed += chunk;
        }
        if (ok) return;
        // Fall through to per-token gemv on any GPU failure.
        fprintf(stderr, "[GEMM] batched GPU path failed; falling back to N gemv calls\n");
    }
    for (int t = 0; t < N; t++) {
        gemv_quant(y + (int64_t)t * M, W, ggml_type, x + (int64_t)t * K,
                   M, K, stream);
    }
}

static bool gemv_quant_cpu(std::vector<float>& h_y, const void* W, int ggml_type,
                           const std::vector<half>& h_x, int M, int K) {
    if (ggml_type != 12 && ggml_type != 13 && ggml_type != 14) {
        fprintf(stderr, "[GEMV] FATAL: unsupported GGML type %d (M=%d K=%d)\n",
                ggml_type, M, K);
        return false;
    }

    const int n_blocks = K / QK_K;
    h_y.assign(M, 0.0f);

    for (int row = 0; row < M; row++) {
        float acc = 0.0f;

        if (ggml_type == 12) {
            const block_q4_K* row_blocks = (const block_q4_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q4_K& blk = row_blocks[b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                const uint8_t* q = blk.qs;
                int is = 0;
                int k = b * QK_K;

                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blk.scales, sc1, m1);
                    get_scale_min_k4(is + 1, blk.scales, sc2, m2);
                    const float d1 = dall * sc1;
                    const float dm1 = dmin * m1;
                    const float d2 = dall * sc2;
                    const float dm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) {
                        acc += (d1 * (q[l] & 0xF) - dm1) * __half2float(h_x[k + l]);
                    }
                    for (int l = 0; l < 32; l++) {
                        acc += (d2 * (q[l] >> 4) - dm2) * __half2float(h_x[k + 32 + l]);
                    }
                    q += 32;
                    is += 2;
                    k += 64;
                }
            }
        } else if (ggml_type == 13) {
            const block_q5_K* row_blocks = (const block_q5_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q5_K& blk = row_blocks[b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                const uint8_t* ql = blk.qs;
                const uint8_t* qh = blk.qh;
                uint8_t u1 = 1, u2 = 2;
                int is = 0;
                int k = b * QK_K;

                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blk.scales, sc1, m1);
                    get_scale_min_k4(is + 1, blk.scales, sc2, m2);
                    const float d1 = dall * sc1;
                    const float dm1 = dmin * m1;
                    const float d2 = dall * sc2;
                    const float dm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) {
                        const float w = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - dm1;
                        acc += w * __half2float(h_x[k + l]);
                    }
                    for (int l = 0; l < 32; l++) {
                        const float w = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - dm2;
                        acc += w * __half2float(h_x[k + 32 + l]);
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                    k += 64;
                }
            }
        } else {
            const block_q6_K* row_blocks = (const block_q6_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q6_K& blk = row_blocks[b];
                const float d = raw_fp16_to_float(blk.d_raw);
                const uint8_t* ql = blk.ql;
                const uint8_t* qh = blk.qh;
                const int8_t* sc = blk.scales;
                int k_base = b * QK_K;

                for (int n = 0; n < QK_K; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        const int k = k_base + n + l;
                        acc += d * (float)sc[is + 0] * (float)q1 * __half2float(h_x[k +  0]);
                        acc += d * (float)sc[is + 2] * (float)q2 * __half2float(h_x[k + 32]);
                        acc += d * (float)sc[is + 4] * (float)q3 * __half2float(h_x[k + 64]);
                        acc += d * (float)sc[is + 6] * (float)q4 * __half2float(h_x[k + 96]);
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
        }

        h_y[row] = acc;
    }

    return true;
}

void gemv_quant(half* y, const void* W, int ggml_type, const half* x,
                int M, int K, cudaStream_t stream) {
    if (fast_gemv_enabled()) {
        if (gemv_quant_gpu(y, W, ggml_type, x, M, K, stream)) {
            static int dbg_count = 0;
            if (debug_kernels_enabled() && dbg_count < 3) {
                cudaError_t err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[GEMV] GPU execution failed: %s\n",
                            cudaGetErrorString(err));
                    cudaMemsetAsync(y, 0, M * sizeof(half), stream);
                    return;
                }
                half h_y[8], h_x[8];
                cudaMemcpy(h_y, y, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[GEMV-GPU #%d] type=%d M=%d K=%d out: ", dbg_count, ggml_type, M, K);
                for (int i = 0; i < 8 && i < M; i++) fprintf(stderr, "%.4f ", __half2float(h_y[i]));
                fprintf(stderr, " | in: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
                fprintf(stderr, "\n");
                dbg_count++;
            }
            return;
        }
    }

    cudaStreamSynchronize(stream);

    std::vector<half> h_x(K);
    std::vector<float> h_y_f32;
    std::vector<half> h_y(M);
    cudaMemcpy(h_x.data(), x, K * sizeof(half), cudaMemcpyDeviceToHost);

    if (!gemv_quant_cpu(h_y_f32, W, ggml_type, h_x, M, K)) {
        cudaMemsetAsync(y, 0, M * sizeof(half), stream);
        return;
    }

    for (int i = 0; i < M; i++) h_y[i] = __float2half(h_y_f32[i]);
    cudaMemcpy(y, h_y.data(), M * sizeof(half), cudaMemcpyHostToDevice);

    static int dbg_count = 0;
    if (debug_kernels_enabled() && dbg_count < 3) {
        cudaStreamSynchronize(stream);
        half h_y[8], h_x[8];
        cudaMemcpy(h_y, y, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GEMV #%d] type=%d M=%d K=%d out: ", dbg_count, ggml_type, M, K);
        for (int i = 0; i < 8 && i < M; i++) fprintf(stderr, "%.4f ", __half2float(h_y[i]));
        fprintf(stderr, " | in: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
        fprintf(stderr, "\n");
        dbg_count++;
    }
}

void gemv_quant_add(half* y, const void* W, int ggml_type, const half* x,
                    const half* residual, int M, int K, cudaStream_t stream) {
    if (fast_gemv_enabled()) {
        if (gemv_quant_add_gpu(y, W, ggml_type, x, residual, M, K, stream)) {
            return;
        }
    }

    cudaStreamSynchronize(stream);

    std::vector<half> h_x(K);
    std::vector<half> h_residual(M);
    std::vector<float> h_y_f32;
    std::vector<half> h_y(M);
    cudaMemcpy(h_x.data(), x, K * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_residual.data(), residual, M * sizeof(half), cudaMemcpyDeviceToHost);

    if (!gemv_quant_cpu(h_y_f32, W, ggml_type, h_x, M, K)) {
        cudaMemsetAsync(y, 0, M * sizeof(half), stream);
        return;
    }

    for (int i = 0; i < M; i++) {
        const half projected = __float2half(h_y_f32[i]);
        h_y[i] = __float2half(__half2float(projected) +
                              __half2float(h_residual[i]));
    }
    cudaMemcpy(y, h_y.data(), M * sizeof(half), cudaMemcpyHostToDevice);
}

void gemv_quant_pair(
    half* y0, const void* W0, int ggml_type0, int M0,
    half* y1, const void* W1, int ggml_type1, int M1,
    const half* x, int K, cudaStream_t stream)
{
    if (fast_gemv_enabled() &&
        gemv_quant_pair_gpu(y0, W0, ggml_type0, M0,
                            y1, W1, ggml_type1, M1,
                            x, K, stream)) {
        return;
    }

    gemv_quant(y0, W0, ggml_type0, x, M0, K, stream);
    gemv_quant(y1, W1, ggml_type1, x, M1, K, stream);
}

void gemv_quant_triple(
    half* y0, const void* W0, int ggml_type0, int M0,
    half* y1, const void* W1, int ggml_type1, int M1,
    half* y2, const void* W2, int ggml_type2, int M2,
    const half* x, int K, cudaStream_t stream)
{
    if (fast_gemv_enabled() &&
        gemv_quant_triple_gpu(y0, W0, ggml_type0, M0,
                              y1, W1, ggml_type1, M1,
                              y2, W2, ggml_type2, M2,
                              x, K, stream)) {
        return;
    }

    gemv_quant(y0, W0, ggml_type0, x, M0, K, stream);
    gemv_quant(y1, W1, ggml_type1, x, M1, K, stream);
    gemv_quant(y2, W2, ggml_type2, x, M2, K, stream);
}

void gemv_quant_f32(float* y, const void* W, int ggml_type, const half* x,
                    int M, int K, cudaStream_t stream) {
    if (fast_gemv_enabled()) {
        if (gemv_quant_gpu_f32(y, W, ggml_type, x, M, K, stream)) {
            return;
        }
    }

    cudaStreamSynchronize(stream);

    std::vector<half> h_x(K);
    std::vector<float> h_y;
    cudaMemcpy(h_x.data(), x, K * sizeof(half), cudaMemcpyDeviceToHost);

    if (!gemv_quant_cpu(h_y, W, ggml_type, h_x, M, K)) {
        cudaMemsetAsync(y, 0, M * sizeof(float), stream);
        return;
    }

    cudaMemcpy(y, h_y.data(), M * sizeof(float), cudaMemcpyHostToDevice);
}

bool dequant_embedding_row(half* dst, const void* W, int ggml_type,
                           int token_id, int hidden_dim, cudaStream_t stream) {
    if (hidden_dim <= 0 || hidden_dim % QK_K != 0) {
        return false;
    }

    const void* W_device = resolve_weight_device_ptr(W);
    if (!W_device) {
        return false;
    }

    const int blocks = hidden_dim / QK_K;
    switch (ggml_type) {
        case 12:
            dequant_embedding_q4k_kernel<<<blocks, QK_K, 0, stream>>>(
                dst, (const block_q4_K*)W_device, token_id, hidden_dim);
            break;
        case 14:
            dequant_embedding_q6k_kernel<<<blocks, QK_K, 0, stream>>>(
                dst, (const block_q6_K*)W_device, token_id, hidden_dim);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[embd] GPU dequant launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace jllm
