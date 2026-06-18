// test_mmq_q4k_gemm.cu — Path E phase E3
//
// Scale the E2 single-tile skeleton into a real GEMM that handles
// arbitrary M / N / K with proper boundary masking. Two test runs:
//
//   1. Correctness: a small problem (M=128, K=512, N=16) where a CPU
//      scalar reference is cheap. Compare element-wise.
//   2. Throughput: the Qwen3-4B Wo prefill shape (M=2560, K=2560,
//      N=33). Measure kernel-only wall-clock and compute GFLOPS.
//
// No engine integration in this PR. The point is to get the full-size
// MMQ kernel working standalone with a clean GFLOPS number we can use
// to decide whether E3b (wire-into-dispatcher) is worth doing now or
// whether the kernel needs more tuning first.
//
// Issue: #33. PR: path-e/03-mmq-q4k-gemm.

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

// ── Q4_K block (matches src/kernels/gemv_q4.cu) ──────────────────────────

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

// ── E3 full-size kernel ──────────────────────────────────────────────────
//
// Grid: (ceil(N / TILE_N), ceil(M / TILE_M), 1).
// Block: 32 threads (one warp) computing a single TILE_M × TILE_N output.
// W is row-major [M × n_blocks] of block_q4_K.
// x is row-major [N × K]    of FP16   (per-token contiguous).
// y is row-major [N × M]    of FP16   (per-token contiguous).
//
// Each warp loops over n_blocks Q4_K blocks, and within each block over
// 8 sub-blocks of 32 quants. Two m16n8k16 MMAs per sub-block ⇒ 16 MMAs
// per Q4_K block per warp. The 16 × 32 FP16 staging tile lives in
// shared memory.

constexpr int TILE_M = 16;
constexpr int TILE_N = 8;

__global__ void gemm_mmq_q4k_kernel(half*             __restrict__ y,
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

    __shared__ half A_tile[TILE_M][32];

    for (int b = 0; b < n_blocks; b++) {
        #pragma unroll
        for (int sb = 0; sb < 8; sb++) {
            const int il     = sb >> 1;
            const int parity = sb &  1;
            const int col    = lane;

            #pragma unroll
            for (int row = 0; row < TILE_M; row++) {
                const int g_row = row_base + row;
                if (g_row < M) {
                    const block_q4_K& blk = W[(int64_t)g_row * n_blocks + b];
                    const float dall = raw_fp16_to_float(blk.d_raw);
                    const float dmin = raw_fp16_to_float(blk.dmin_raw);
                    uint8_t sc, mn;
                    get_scale_min_k4(sb, blk.scales, sc, mn);
                    const float d  = dall * sc;
                    const float dm = dmin * mn;
                    const uint8_t qb = blk.qs[32 * il + col];
                    const float val  = parity
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

                // B fragment: load activations from x for tokens
                // [tok_base + groupID] at K positions
                // [k_block_base + ka + 2*tinG ..+1, +8..+9].
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

    // ── Store D fragment back to y (FP16, row-major [N × M]) ──
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

// ── Host scalar reference ────────────────────────────────────────────────

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

// ── Helpers ──────────────────────────────────────────────────────────────

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

// ── Test 1: correctness on small problem ─────────────────────────────────

static bool run_correctness_test() {
    constexpr int M = 128;
    constexpr int N = 16;
    constexpr int K = 512;
    constexpr int n_blocks = K / QK_K;

    printf("\n── Correctness: M=%d N=%d K=%d ────────────────\n", M, N, K);

    std::vector<block_q4_K> h_W(M * n_blocks);
    std::vector<half>       h_X(N * K);
    std::vector<half>       h_Y(N * M);
    std::vector<half>       h_Y_ref(N * M);
    fill_random_q4k(h_W,  0xC0FFEE);
    fill_random_half(h_X, 0xFEEDBEEF, -1.0f, 1.0f);

    auto t0 = std::chrono::steady_clock::now();
    host_reference(h_W, h_X, h_Y_ref, M, N, K);
    auto t1 = std::chrono::steady_clock::now();
    printf("  Host ref: %.2f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    block_q4_K* d_W; half* d_X; half* d_Y;
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(block_q4_K)));
    CHECK_CUDA(cudaMalloc(&d_X, h_X.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_Y, h_Y.size() * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(half),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_Y, 0,          h_Y.size() * sizeof(half)));

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);
    gemm_mmq_q4k_kernel<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);

    // Tolerance:
    //
    //   Each output is a 512-term FP32 dot product, rounded to FP16 at
    //   writeback. For the largest values we see (~50 in magnitude),
    //   FP16 ULP is 50 / 1024 ≈ 0.049. The MMA evaluates the sum as a
    //   tree-reduction inside each 16-wide K-tile and across 32 sub-tile
    //   chunks; the host reference sums linearly k = 0 → 511. These
    //   orderings produce up to ~4–5 ULP of relative error at this
    //   problem size — not a kernel bug, just float-add associativity.
    //   0.25 ≈ 5 ULP at the peak magnitude is the right floor.
    constexpr float kTol = 0.25f;

    auto s = compare_half(h_Y, h_Y_ref, kTol);
    printf("  Sample: Y[0][0] ref=%g mma=%g | Y[%d][%d] ref=%g mma=%g\n",
           __half2float(h_Y_ref[0]), __half2float(h_Y[0]),
           N - 1, M - 1,
           __half2float(h_Y_ref[N * M - 1]),
           __half2float(h_Y[N * M - 1]));
    printf("  Max abs error: %g | Max rel error: %g | %d / %d outside %g\n",
           s.max_abs, s.max_rel, s.bad, N * M, kTol);

    if (s.bad > 0) {
        printf("  FAIL: correctness check did not pass.\n");
        return false;
    }
    printf("  PASS: correctness within %.2f abs (≤ 5 FP16 ULP at output magnitude).\n", kTol);
    return true;
}

// ── Test 2: throughput on Qwen3-4B Wo shape ──────────────────────────────

static void run_throughput_test() {
    constexpr int M = 2560;
    constexpr int N = 33;
    constexpr int K = 2560;
    constexpr int n_blocks = K / QK_K;

    printf("\n── Throughput: M=%d N=%d K=%d (Qwen3-4B Wo @ N=33) ──\n", M, N, K);

    std::vector<block_q4_K> h_W(M * n_blocks);
    std::vector<half>       h_X(N * K);
    fill_random_q4k(h_W,  0xBEEF);
    fill_random_half(h_X, 0xCAFE, -1.0f, 1.0f);

    block_q4_K* d_W; half* d_X; half* d_Y;
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(block_q4_K)));
    CHECK_CUDA(cudaMalloc(&d_X, h_X.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_Y, (size_t)N * M * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(half),       cudaMemcpyHostToDevice));

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);
    printf("  Grid: (%d, %d, 1), Block: 32 threads.\n", grid.x, grid.y);

    // Warm-up
    for (int i = 0; i < 5; i++) {
        gemm_mmq_q4k_kernel<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    constexpr int ITERS = 100;
    cudaEventRecord(e0);
    for (int i = 0; i < ITERS; i++) {
        gemm_mmq_q4k_kernel<<<grid, 32>>>(d_Y, d_W, d_X, M, N, K);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, e0, e1);
    const float ms_per_call = ms_total / ITERS;

    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = flops / (ms_per_call * 1.0e6);

    printf("  Kernel time: %.3f ms / call (avg of %d)\n", ms_per_call, ITERS);
    printf("  Throughput: %.1f GFLOPS (2 * M * N * K / time)\n", gflops);
    printf("  Reference: Orin Nano Super FP16 tensor-core peak ≈ 15 TFLOPS;\n"
           "  Q4_K dequant + mma utilization ratio = %.1f%%\n",
           100.0 * gflops / 15000.0);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
}

int main(int, char**) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SM %d.%d, %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    bool ok = run_correctness_test();
    // Always run throughput so we still get a GFLOPS reading even if
    // the correctness tolerance flagged something — useful for triaging
    // (a fast-but-wrong kernel is still informative about the ceiling).
    run_throughput_test();
    return ok ? 0 : 1;
}
