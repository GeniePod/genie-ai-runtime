// test_mmq_q4k_precompute.cu — Path E phase E4
//
// Two targeted fixes layered on top of the E3 kernel (test_mmq_q4k_gemm.cu):
//
//   1. Per-(row, sub-block) scale precompute. Across the 32 lanes of a
//      warp, the E3 kernel each redundantly computed
//        dall = raw_fp16_to_float(blk.d_raw)
//        dmin = raw_fp16_to_float(blk.dmin_raw)
//        get_scale_min_k4(sb, blk.scales, sc, mn)
//      per (row, sub-block). Only the byte-per-lane is lane-unique, so
//      the warp does 32x redundant scalar work per (row, sb). E4
//      computes the 16 × 8 = 128 (d, dm) pairs once into shared memory
//      at the start of each Q4_K block, distributed across the 32
//      lanes (4 pairs per lane).
//
//   2. Drop `#pragma unroll` on the inner row loop. E3 unrolled
//      16-deep, which forced the compiler to keep 16 copies of every
//      intermediate and ballooned register count to 139 → ~29 %
//      occupancy. Without the unroll the compiler should reuse a
//      single set of intermediates → register count ~50–60 → much
//      better occupancy.
//
// Correctness target: identical to E3 (max abs ≤ 0.25 vs host scalar
// reference at the same problem size).
// Throughput target: ≥ 200 GFLOPS on Qwen3-4B Wo shape (vs E3's 74).
// Anything below ~150 GFLOPS keeps us under the scalar batched kernel's
// effective rate on the integrated mix of shapes, so we'd not re-attempt
// integration without an E5 multi-warp-block rewrite.
//
// Issue: #33. PR: path-e/04-mmq-q4k-precompute.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

// ── Q4_K block ──────────────────────────────────────────────────────────

constexpr int QK_K = 256;

struct __attribute__((packed)) block_q4_K {
    uint16_t d_raw;
    uint16_t dmin_raw;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0)         result = std::ldexp((float)mant, -24);
    else if (exp == 31)   result = mant ? NAN : INFINITY;
    else                  result = std::ldexp((float)(mant + 1024), (int)exp - 25);
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
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ── E4 kernel ───────────────────────────────────────────────────────────

constexpr int TILE_M = 16;
constexpr int TILE_N = 8;

__global__ void gemm_mmq_q4k_kernel_v2(half*             __restrict__ y,
                                       const block_q4_K* __restrict__ W,
                                       const half*       __restrict__ x,
                                       int M, int N, int K)
{
    const int row_base = blockIdx.y * TILE_M;
    const int tok_base = blockIdx.x * TILE_N;
    if (row_base >= M) return;

    const int lane    = threadIdx.x & 31;
    const int groupID = lane >> 2;
    const int tinG    = lane &  3;

    const int n_blocks = K / QK_K;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // FP16 staging tile (16 rows × 32 cols, one sub-block at a time).
    __shared__ half A_tile[TILE_M][32];

    // Per-(row, sub-block) scale precompute: 16 × 8 = 128 (d, dm) pairs
    // per Q4_K block iteration. 4 entries per lane.
    __shared__ float per_d [TILE_M][8];
    __shared__ float per_dm[TILE_M][8];

    for (int b = 0; b < n_blocks; b++) {
        // ── Step 1: precompute 128 (d, dm) pairs across 32 lanes ──
        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
            const int idx = lane * 4 + slot;       // 0..127
            const int row = idx >> 3;              // 0..15
            const int sb  = idx &  7;              // 0..7
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
        __syncwarp();

        // ── Step 2: 8 sub-blocks, each with a 16×32 dequant + 2 MMAs ──
        for (int sb = 0; sb < 8; sb++) {
            const int il     = sb >> 1;
            const int parity = sb &  1;
            const int col    = lane;

            // No #pragma unroll here — let the compiler share a single
            // set of intermediate registers across the 16 iterations.
            for (int row = 0; row < TILE_M; row++) {
                const int g_row = row_base + row;
                if (g_row < M) {
                    const block_q4_K& blk = W[(int64_t)g_row * n_blocks + b];
                    const uint8_t qb = blk.qs[32 * il + col];
                    const float d  = per_d [row][sb];
                    const float dm = per_dm[row][sb];
                    const float val = parity
                        ? (d * (qb >> 4)  - dm)
                        : (d * (qb & 0xF) - dm);
                    A_tile[row][col] = __float2half(val);
                } else {
                    A_tile[row][col] = __float2half(0.0f);
                }
            }
            __syncwarp();

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
            __syncwarp();
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

// ── Host reference ──────────────────────────────────────────────────────

static void host_reference(const std::vector<block_q4_K>& W,
                           const std::vector<half>&       X,
                           std::vector<half>&             Y,
                           int M, int N, int K)
{
    const int n_blocks = K / QK_K;
    for (int t = 0; t < N; t++) {
        for (int r = 0; r < M; r++) {
            float acc = 0.0f;
            for (int b = 0; b < n_blocks; b++) {
                const block_q4_K& blk = W[(int64_t)r * n_blocks + b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                for (int sb = 0; sb < 8; sb++) {
                    const int il     = sb >> 1;
                    const int parity = sb &  1;
                    uint8_t sc, mn;
                    get_scale_min_k4(sb, blk.scales, sc, mn);
                    const float d  = dall * sc;
                    const float dm = dmin * mn;
                    const int k_base = b * QK_K + sb * 32;
                    for (int col = 0; col < 32; col++) {
                        const uint8_t qb = blk.qs[32 * il + col];
                        const float val  = parity
                            ? (d * (qb >> 4)  - dm)
                            : (d * (qb & 0xF) - dm);
                        acc += val * __half2float(X[(int64_t)t * K + k_base + col]);
                    }
                }
            }
            Y[(int64_t)t * M + r] = __float2half(acc);
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

static void fill_random_q4k(std::vector<block_q4_K>& W, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int>    d_byte(0, 255);
    std::uniform_real_distribution<float> d_dall(0.005f, 0.020f);
    std::uniform_real_distribution<float> d_dmin(0.001f, 0.004f);
    for (auto& blk : W) {
        blk.d_raw    = __half_as_ushort(__float2half(d_dall(rng)));
        blk.dmin_raw = __half_as_ushort(__float2half(d_dmin(rng)));
        for (int i = 0; i < 12;       i++) blk.scales[i] = (uint8_t)d_byte(rng);
        for (int i = 0; i < QK_K / 2; i++) blk.qs[i]     = (uint8_t)d_byte(rng);
    }
}

static void fill_random_half(std::vector<half>& X, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : X) v = __float2half(d(rng));
}

struct CompareStats { float max_abs; float max_rel; int bad; };

static CompareStats compare_half(const std::vector<half>& a,
                                 const std::vector<half>& b,
                                 float tol)
{
    CompareStats s{0.0f, 0.0f, 0};
    for (size_t i = 0; i < a.size(); i++) {
        const float fa = __half2float(a[i]);
        const float fb = __half2float(b[i]);
        const float e  = std::fabs(fa - fb);
        const float r  = e / (std::fabs(fb) + 1e-6f);
        if (e > s.max_abs) s.max_abs = e;
        if (r > s.max_rel) s.max_rel = r;
        if (e > tol) s.bad++;
    }
    return s;
}

// ── Test 1: correctness ─────────────────────────────────────────────────

static bool run_correctness_test() {
    constexpr int M = 128;
    constexpr int N = 16;
    constexpr int K = 512;
    constexpr int n_blocks = K / QK_K;
    constexpr float kTol = 0.25f;

    printf("\n── Correctness: M=%d N=%d K=%d ────────────────\n", M, N, K);

    std::vector<block_q4_K> h_W(M * n_blocks);
    std::vector<half>       h_X(N * K);
    std::vector<half>       h_Y(N * M, __float2half(0.0f));
    std::vector<half>       h_Y_ref(N * M);
    fill_random_q4k(h_W,  0xC0FFEE);
    fill_random_half(h_X, 0xFEEDBEEF, -1.0f, 1.0f);

    host_reference(h_W, h_X, h_Y_ref, M, N, K);

    block_q4_K* d_W; half* d_X; half* d_Y;
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(block_q4_K)));
    CHECK_CUDA(cudaMalloc(&d_X, h_X.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_Y, h_Y.size() * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(half),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_Y, 0, h_Y.size() * sizeof(half)));

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);
    gemm_mmq_q4k_kernel_v2<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);

    auto s = compare_half(h_Y, h_Y_ref, kTol);
    printf("  Sample: Y[0][0] ref=%g mma=%g | Y[%d][%d] ref=%g mma=%g\n",
           __half2float(h_Y_ref[0]),       __half2float(h_Y[0]),
           N - 1, M - 1,
           __half2float(h_Y_ref[N * M - 1]),
           __half2float(h_Y[N * M - 1]));
    printf("  Max abs error: %g | Max rel error: %g | %d / %d outside %g\n",
           s.max_abs, s.max_rel, s.bad, N * M, kTol);
    if (s.bad > 0) { printf("  FAIL.\n"); return false; }
    printf("  PASS.\n");
    return true;
}

// ── Test 2: throughput on Qwen3-4B prefill shapes ───────────────────────

static void bench_shape(const char* label, int M, int N, int K) {
    const int n_blocks = K / QK_K;

    std::vector<block_q4_K> h_W(M * n_blocks);
    std::vector<half>       h_X(N * K);
    fill_random_q4k(h_W,  0xBEEF + M + K);
    fill_random_half(h_X, 0xCAFE + N,     -1.0f, 1.0f);

    block_q4_K* d_W; half* d_X; half* d_Y;
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(block_q4_K)));
    CHECK_CUDA(cudaMalloc(&d_X, h_X.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_Y, (size_t)N * M * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(half),       cudaMemcpyHostToDevice));

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_mmq_q4k_kernel_v2<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    constexpr int ITERS = 100;
    cudaEventRecord(e0);
    for (int i = 0; i < ITERS; i++) {
        gemm_mmq_q4k_kernel_v2<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, e0, e1);
    const float ms_per_call = ms_total / ITERS;
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = flops / (ms_per_call * 1.0e6);

    printf("  %-12s M=%5d N=%2d K=%5d  ms=%6.3f  GFLOPS=%6.1f  util=%4.1f%%\n",
           label, M, N, K, ms_per_call, gflops, 100.0 * gflops / 15000.0);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
}

static void run_throughput_test() {
    printf("\n── Throughput on Qwen3-4B Q4_K_M prefill shapes (N=33) ──\n");

    // Mirrors the actual prefill GEMM shapes in alpha.6.
    bench_shape("Wo",       2560, 33, 2560);
    bench_shape("gate/up",  9728, 33, 2560);
    bench_shape("down",     2560, 33, 9728);
    bench_shape("Wq",       2560, 33, 2560);
    bench_shape("Wk",       1024, 33, 2560);
    bench_shape("Wv",       1024, 33, 2560);
}

int main(int, char**) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SM %d.%d, %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    bool ok = run_correctness_test();
    run_throughput_test();
    return ok ? 0 : 1;
}
