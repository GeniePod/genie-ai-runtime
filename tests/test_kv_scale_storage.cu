// test_kv_scale_storage.cu — Path I phase I1
//
// Validates per-head INT8 scale storage in KVCachePool:
//   1. Allocates the pool in INT8 mode (kv_type_bytes=1) and confirms
//      kv_scales_bytes() reports the expected size.
//   2. kv_scale_ptr(layer, pos, is_value) returns non-null device
//      pointers within the allocated region, with the right per-layer
//      / per-half-block / per-pos strides.
//   3. Writes a deterministic pattern through each pointer, reads it
//      back, verifies layout boundaries don't overlap or alias.
//   4. Allocates the pool in FP16 mode (kv_type_bytes=2) and confirms
//      kv_scales_bytes() = 0 and kv_scale_ptr() returns nullptr.

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

using jllm::KVCachePool;

static int test_int8_mode() {
    printf("\n── INT8 mode: allocate, write/read, layout-overlap check ──\n");

    KVCachePool::Config cfg{};
    cfg.n_layers         = 36;
    cfg.n_kv_heads       = 8;
    cfg.head_dim         = 128;
    cfg.max_context      = 1024;
    cfg.overflow_context = 0;
    cfg.kv_type_bytes    = 1;          // INT8

    KVCachePool pool;
    if (!pool.init(cfg)) {
        fprintf(stderr, "FAIL: pool.init returned false\n");
        return 1;
    }

    const int64_t expected_bytes =
        2LL * cfg.n_layers * cfg.max_context * cfg.n_kv_heads * (int64_t)sizeof(float);
    const int64_t got_bytes = pool.kv_scales_bytes();
    printf("  kv_scales_bytes: expected %ld, got %ld\n", expected_bytes, got_bytes);
    if (got_bytes != expected_bytes) {
        fprintf(stderr, "FAIL: kv_scales_bytes mismatch\n");
        pool.destroy(); return 1;
    }

    // Probe pointers at corners (first/last layer, first/last pos, K/V).
    struct Probe { int layer; int pos; bool is_value; };
    const Probe probes[] = {
        {  0,    0, false},  // layer 0, pos 0, K
        {  0,    0, true},   // layer 0, pos 0, V
        {  0, 1023, false},  // layer 0, pos 1023, K
        { 35,    0, true},   // layer 35, pos 0, V
        { 35, 1023, true},   // layer 35, pos 1023, V
    };
    std::vector<float*> ptrs;
    for (const auto& p : probes) {
        float* sp = pool.kv_scale_ptr(p.layer, p.pos, p.is_value);
        if (!sp) {
            fprintf(stderr, "FAIL: kv_scale_ptr(layer=%d, pos=%d, is_value=%d) returned nullptr\n",
                    p.layer, p.pos, (int)p.is_value);
            pool.destroy(); return 1;
        }
        ptrs.push_back(sp);
    }
    printf("  5 corner pointers all non-null ✓\n");

    // Each probe's scale row is n_kv_heads floats wide. Write a
    // distinct pattern at each, then read all back and confirm no
    // overlap or aliasing.
    for (size_t i = 0; i < ptrs.size(); i++) {
        std::vector<float> host(cfg.n_kv_heads, (float)(i + 1) * 1000.0f);
        CHECK_CUDA(cudaMemcpy(ptrs[i], host.data(),
                              cfg.n_kv_heads * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    for (size_t i = 0; i < ptrs.size(); i++) {
        std::vector<float> host(cfg.n_kv_heads, 0.0f);
        CHECK_CUDA(cudaMemcpy(host.data(), ptrs[i],
                              cfg.n_kv_heads * sizeof(float),
                              cudaMemcpyDeviceToHost));
        const float want = (float)(i + 1) * 1000.0f;
        for (int h = 0; h < cfg.n_kv_heads; h++) {
            if (host[h] != want) {
                fprintf(stderr, "FAIL: probe %zu, head %d: read %g, expected %g "
                                "(possible aliasing in layout)\n",
                        i, h, host[h], want);
                pool.destroy(); return 1;
            }
        }
    }
    printf("  5 probes round-trip with distinct values, no aliasing ✓\n");

    // Cross-check stride math: kv_scale_ptr(layer=1, pos=0, K) should be
    // exactly per_layer floats after kv_scale_ptr(0, 0, K).
    float* l0_k0 = pool.kv_scale_ptr(0, 0, false);
    float* l1_k0 = pool.kv_scale_ptr(1, 0, false);
    const int64_t per_layer_floats = 2LL * cfg.max_context * cfg.n_kv_heads;
    const int64_t got_layer_stride = l1_k0 - l0_k0;
    if (got_layer_stride != per_layer_floats) {
        fprintf(stderr, "FAIL: per-layer stride: got %ld floats, expected %ld\n",
                got_layer_stride, per_layer_floats);
        pool.destroy(); return 1;
    }
    // Same for K → V within a layer: exactly max_context × n_kv_heads floats apart.
    float* l0_v0 = pool.kv_scale_ptr(0, 0, true);
    const int64_t got_kv_stride = l0_v0 - l0_k0;
    const int64_t expect_kv_stride = (int64_t)cfg.max_context * cfg.n_kv_heads;
    if (got_kv_stride != expect_kv_stride) {
        fprintf(stderr, "FAIL: K→V stride: got %ld floats, expected %ld\n",
                got_kv_stride, expect_kv_stride);
        pool.destroy(); return 1;
    }
    printf("  layer stride %ld floats ✓, K→V stride %ld floats ✓\n",
           got_layer_stride, got_kv_stride);

    pool.destroy();
    printf("  PASS: INT8 mode allocates and lays out scales correctly.\n");
    return 0;
}

static int test_fp16_mode() {
    printf("\n── FP16 mode: kv_scales should NOT be allocated ──\n");

    KVCachePool::Config cfg{};
    cfg.n_layers         = 36;
    cfg.n_kv_heads       = 8;
    cfg.head_dim         = 128;
    cfg.max_context      = 1024;
    cfg.overflow_context = 0;
    cfg.kv_type_bytes    = 2;          // FP16 — the alpha.11 default

    KVCachePool pool;
    if (!pool.init(cfg)) {
        fprintf(stderr, "FAIL: pool.init returned false\n");
        return 1;
    }

    if (pool.kv_scales_bytes() != 0) {
        fprintf(stderr, "FAIL: FP16 mode reports kv_scales_bytes=%ld (expected 0)\n",
                pool.kv_scales_bytes());
        pool.destroy(); return 1;
    }
    if (pool.kv_scale_ptr(0, 0, false) != nullptr ||
        pool.kv_scale_ptr(0, 0, true)  != nullptr)
    {
        fprintf(stderr, "FAIL: FP16 mode kv_scale_ptr returned non-null\n");
        pool.destroy(); return 1;
    }
    printf("  kv_scales_bytes=0 ✓, kv_scale_ptr=nullptr ✓\n");
    pool.destroy();
    printf("  PASS: FP16 mode skips scale allocation.\n");
    return 0;
}

int main(int, char**) {
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);

    int rc = test_int8_mode();
    rc |= test_fp16_mode();
    if (rc != 0) {
        printf("\nFAIL: one or more cases did not pass.\n");
        return 1;
    }
    printf("\nPASS: KVCachePool INT8 scale storage works correctly.\n");
    return 0;
}
