// test_kv_int8_convert.cu — Path I phase I2
//
// Validates the FP16 → INT8 conversion path the engine uses for INT8
// KV cache writes:
//
//   1. fp16_to_int8 called with rows = n_kv_heads, cols = head_dim
//      (i.e. one absmax scale per kv_head, not one scale for the whole
//      KV row).
//   2. The scale_out pointer is the per-(layer, pos, K-or-V) slot
//      returned by KVCachePool::kv_scale_ptr (I1, #63).
//   3. Round-trip: dequantize the stored INT8 with the captured scales,
//      compare to the original FP16. INT8 has ~1/127 relative precision
//      so per-element error should be bounded by ~0.8 % of max-abs per
//      head.
//   4. Different (layer, pos) slots are independent — writing into
//      slot A doesn't smear scales into slot B (no aliasing).
//
// Engine integration of INT8 KV at the attention kernels lands in I3.
// I2 just makes sure the conversion + scale-capture half is correct.

#include "jllm_memory.h"
#include "jllm_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

using jllm::KVCachePool;
using jllm::fp16_to_int8;

static KVCachePool::Config make_qwen3_4b_int8() {
    KVCachePool::Config cfg{};
    cfg.n_layers         = 36;
    cfg.n_kv_heads       = 8;
    cfg.head_dim         = 128;
    cfg.max_context      = 1024;
    cfg.overflow_context = 0;
    cfg.kv_type_bytes    = 1;          // INT8
    return cfg;
}

// Fill `n` FP16 values with a per-head-distinguishable pattern so we can
// verify each head's scale is computed independently.
static void fill_distinguishable(std::vector<half>& host, int n_heads, int head_dim,
                                 uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
    for (int h = 0; h < n_heads; h++) {
        // Each head gets a different magnitude scale so the per-head
        // absmax math has to compute different scales per row.
        const float head_scale = 0.1f * (float)(h + 1);   // h=0 → 0.1, h=7 → 0.8
        for (int c = 0; c < head_dim; c++) {
            float v = head_scale * (std::cos((float)c * 0.07f + (float)h)
                                    + jitter(rng));
            host[h * head_dim + c] = __float2half(v);
        }
    }
}

static int run_one_slot_round_trip() {
    printf("\n── Round-trip: one (layer, pos, K) slot ──\n");

    auto cfg = make_qwen3_4b_int8();
    KVCachePool pool;
    if (!pool.init(cfg)) { fprintf(stderr, "FAIL: pool.init\n"); return 1; }

    const int total = cfg.n_kv_heads * cfg.head_dim;     // 8 × 128 = 1024
    std::vector<half> h_src(total);
    fill_distinguishable(h_src, cfg.n_kv_heads, cfg.head_dim, 0xC0FFEE);

    half* d_src;
    CHECK_CUDA(cudaMalloc(&d_src, total * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), total * sizeof(half), cudaMemcpyHostToDevice));

    const int layer = 17;
    const int pos   = 42;
    int8_t* d_dst   = (int8_t*)pool.key_ptr(layer, pos);
    float*  d_scale = pool.kv_scale_ptr(layer, pos, /*is_value=*/false);
    if (!d_dst || !d_scale) {
        fprintf(stderr, "FAIL: pool returned null for layer=%d pos=%d\n", layer, pos);
        cudaFree(d_src); pool.destroy(); return 1;
    }

    fp16_to_int8(d_dst, d_scale, d_src, cfg.n_kv_heads, cfg.head_dim, 0);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int8_t> h_int8(total);
    std::vector<float>  h_scales(cfg.n_kv_heads);
    CHECK_CUDA(cudaMemcpy(h_int8.data(),   d_dst,   total * sizeof(int8_t),  cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_scales.data(), d_scale, cfg.n_kv_heads * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify each head has a distinct, monotonically-increasing scale —
    // input magnitudes go 0.1, 0.2, … 0.8 (per-head head_scale =
    // 0.1 × (h+1)), so each head's absmax-derived scale should be
    // ≈ head_scale / 127 and strictly larger than the previous head's.
    // (Earlier draft of the test asserted a 1.5× growth ratio between
    // adjacent heads — that's wrong for a linear input progression
    // where the ratios are 2, 1.5, 1.33, 1.25, 1.2, 1.17, 1.14.)
    printf("  per-head scales:");
    for (int h = 0; h < cfg.n_kv_heads; h++) printf(" %.4g", h_scales[h]);
    printf("\n");
    for (int h = 0; h < cfg.n_kv_heads; h++) {
        const float expected = (0.1f * (float)(h + 1)) / 127.0f;
        const float ratio    = h_scales[h] / expected;
        if (ratio < 0.5f || ratio > 1.5f) {
            fprintf(stderr, "FAIL: head %d scale %.4g not within 0.5–1.5× of "
                            "expected %.4g (ratio %.3f)\n",
                    h, h_scales[h], expected, ratio);
            cudaFree(d_src); pool.destroy(); return 1;
        }
        if (h > 0 && h_scales[h] <= h_scales[h - 1]) {
            fprintf(stderr, "FAIL: head %d scale %.4g not strictly > head %d %.4g\n",
                    h, h_scales[h], h - 1, h_scales[h - 1]);
            cudaFree(d_src); pool.destroy(); return 1;
        }
    }

    // Dequantize INT8 back to FP32 and compare to the original FP16.
    // Per-element error bound: scale = max_abs / 127, so |err| ≤ scale.
    // Relative error ≤ ~1/127 = 0.79 % of the row's max-abs value.
    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    int worst_h = 0;
    for (int h = 0; h < cfg.n_kv_heads; h++) {
        const float s = h_scales[h];
        for (int c = 0; c < cfg.head_dim; c++) {
            const float orig    = __half2float(h_src[h * cfg.head_dim + c]);
            const float dequant = (float)h_int8[h * cfg.head_dim + c] * s;
            const float e       = std::fabs(orig - dequant);
            if (e > max_abs_err) { max_abs_err = e; worst_h = h; }
            const float head_scale = 0.1f * (float)(h + 1);
            const float r = e / (head_scale + 1e-6f);
            if (r > max_rel_err) max_rel_err = r;
        }
    }
    printf("  max abs err: %g (worst head %d, that head's max_abs ≈ %.3g)\n",
           max_abs_err, worst_h, 0.1f * (float)(worst_h + 1));
    printf("  max rel err: %g (vs ~0.79 %% INT8 precision floor)\n",
           max_rel_err);
    if (max_rel_err > 0.02f) {   // generous: 2 % vs the 0.79 % floor
        fprintf(stderr, "FAIL: round-trip relative error %.4f exceeds 2 %%\n",
                max_rel_err);
        cudaFree(d_src); pool.destroy(); return 1;
    }

    cudaFree(d_src);
    pool.destroy();
    printf("  PASS: round-trip within INT8 precision floor.\n");
    return 0;
}

static int run_no_aliasing() {
    printf("\n── No aliasing: writes to different (layer, pos, K/V) slots independent ──\n");

    auto cfg = make_qwen3_4b_int8();
    KVCachePool pool;
    if (!pool.init(cfg)) { fprintf(stderr, "FAIL: pool.init\n"); return 1; }

    const int total = cfg.n_kv_heads * cfg.head_dim;

    // Three writes to different slots with different magnitudes.
    struct Site { int layer; int pos; bool is_value; uint32_t seed; float head_floor; };
    const Site sites[] = {
        {  0,    0, false, 0xAAAA, 0.05f },
        {  0,    0, true,  0xBBBB, 0.30f },
        { 35, 1023, false, 0xCCCC, 0.80f },
    };

    half* d_src;
    CHECK_CUDA(cudaMalloc(&d_src, total * sizeof(half)));
    for (const auto& s : sites) {
        std::vector<half> h_src(total);
        // Use distinct base magnitudes per site.
        std::mt19937 rng(s.seed);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int i = 0; i < total; i++) h_src[i] = __float2half(s.head_floor * d(rng));
        CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), total * sizeof(half), cudaMemcpyHostToDevice));

        int8_t* d_dst   = (int8_t*)(s.is_value ? pool.val_ptr(s.layer, s.pos) : pool.key_ptr(s.layer, s.pos));
        float*  d_scale = pool.kv_scale_ptr(s.layer, s.pos, s.is_value);
        fp16_to_int8(d_dst, d_scale, d_src, cfg.n_kv_heads, cfg.head_dim, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back the three sites' scales and verify each is in the
    // expected magnitude band (since head_floor is the per-element
    // magnitude bound, the per-head absmax should be ≤ head_floor and
    // > head_floor * 0.5 for random data).
    for (const auto& s : sites) {
        std::vector<float> h_scales(cfg.n_kv_heads);
        float* d_scale = pool.kv_scale_ptr(s.layer, s.pos, s.is_value);
        CHECK_CUDA(cudaMemcpy(h_scales.data(), d_scale,
                              cfg.n_kv_heads * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float min_s = h_scales[0], max_s = h_scales[0];
        for (int h = 1; h < cfg.n_kv_heads; h++) {
            if (h_scales[h] < min_s) min_s = h_scales[h];
            if (h_scales[h] > max_s) max_s = h_scales[h];
        }
        const float expected_upper = s.head_floor / 127.0f;
        const float expected_lower = expected_upper * 0.2f;   // generous lower bound for random data
        printf("  site (layer=%d, pos=%d, %s): per-head scale range [%.3g, %.3g] "
               "(expect ≈ [%.3g, %.3g])\n",
               s.layer, s.pos, s.is_value ? "V" : "K",
               min_s, max_s, expected_lower, expected_upper);
        if (max_s > expected_upper * 1.1f || min_s < expected_lower * 0.5f) {
            fprintf(stderr, "FAIL: site scales out of expected band — possible aliasing\n");
            cudaFree(d_src); pool.destroy(); return 1;
        }
    }

    cudaFree(d_src);
    pool.destroy();
    printf("  PASS: writes to different slots have independent scales.\n");
    return 0;
}

int main(int, char**) {
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);

    int rc = run_one_slot_round_trip();
    rc |= run_no_aliasing();
    if (rc != 0) { printf("\nFAIL: one or more cases did not pass.\n"); return 1; }
    printf("\nPASS: FP16 → INT8 conversion + KVCachePool scale capture works correctly.\n");
    return 0;
}
