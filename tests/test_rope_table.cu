// test_rope_table.cu — Validate that the precomputed RoPE cos/sin table
// produces bit-identical results to the original on-the-fly trig path.
//
// Approach: for each test case apply RoPE twice on identical Q buffers:
//   - reference: tables cleared → rope_inplace falls back to powf+cosf+sinf
//   - table path: table built → rope_inplace uses the precomputed lookup
// Both use the same CUDA fast-math path; expect zero max-abs-diff.
//
// Covers: Qwen3-4B (hd=128), Gemma4 local (hd=256), Gemma4 global (hd=512).

#include "../include/jllm_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void fill_random_fp16(half* d, int n, unsigned seed = 42) {
    srand(seed);
    std::vector<half> h(n);
    for (int i = 0; i < n; i++)
        h[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    cudaMemcpy(d, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

static float max_abs_diff(const half* a, const half* b, int n) {
    std::vector<half> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(half), cudaMemcpyDeviceToHost);
    float mx = 0.0f;
    for (int i = 0; i < n; i++)
        mx = fmaxf(mx, fabsf(__half2float(ha[i]) - __half2float(hb[i])));
    return mx;
}

struct Case {
    const char* name;
    int n_heads, n_kv_heads, head_dim, position;
    float theta; bool neox;
};

int main() {
    const int MAX_SEQ = 1024;
    bool ok = true;

    const Case cases[] = {
        { "Qwen3-4B  hd=128 pos=77",  32, 8,  128, 77,       10000.0f, true },
        { "Gemma4 local hd=256 pos=77", 16, 1, 256, 77,       10000.0f, true },
        { "Gemma4 global hd=512 pos=77",16, 1, 512, 77,     1000000.0f, true },
        { "Qwen3-4B hd=128 pos=0",     32, 8,  128,  0,       10000.0f, true },
        { "Qwen3-4B hd=128 pos=511",   32, 8,  128, 511,      10000.0f, true },
        { "Gemma4 local  pos=512",      16, 1,  256, 512,      10000.0f, true },
    };

    for (const auto& tc : cases) {
        const int q_dim = tc.n_heads * tc.head_dim;

        half *d_ref = nullptr, *d_tbl = nullptr;
        cudaMalloc(&d_ref, q_dim * sizeof(half));
        cudaMalloc(&d_tbl, q_dim * sizeof(half));

        fill_random_fp16(d_ref, q_dim, 42);
        cudaMemcpy(d_tbl, d_ref, q_dim * sizeof(half), cudaMemcpyDeviceToDevice);

        // Reference: clear any existing table → forces trig path.
        jllm::rope_clear_tables();
        jllm::rope_inplace(d_ref, nullptr, tc.n_heads, 0, tc.head_dim,
                           tc.position, tc.theta, tc.neox, nullptr);

        // Table path: precompute, then invoke.
        jllm::rope_precompute_table(MAX_SEQ, tc.head_dim, tc.theta);
        jllm::rope_inplace(d_tbl, nullptr, tc.n_heads, 0, tc.head_dim,
                           tc.position, tc.theta, tc.neox, nullptr);

        cudaDeviceSynchronize();
        const float diff = max_abs_diff(d_ref, d_tbl, q_dim);
        const bool pass  = (diff == 0.0f);
        printf("[%s]  diff=%.2e  %s\n", tc.name, (double)diff, pass ? "PASS" : "FAIL");
        if (!pass) ok = false;

        cudaFree(d_ref);
        cudaFree(d_tbl);
    }

    // Also test the store-KV-fp16 fused variant.
    {
        const int n_heads = 16, n_kv_heads = 1, head_dim = 256, pos = 100;
        const float theta = 10000.0f;
        const int q_dim  = n_heads    * head_dim;
        const int kv_dim = n_kv_heads * head_dim;

        half *d_q_ref, *d_q_tbl, *d_k_ref, *d_k_tbl, *d_v;
        half *d_kc_ref, *d_kc_tbl, *d_vc_ref, *d_vc_tbl;
        cudaMalloc(&d_q_ref, q_dim  * sizeof(half));
        cudaMalloc(&d_q_tbl, q_dim  * sizeof(half));
        cudaMalloc(&d_k_ref, kv_dim * sizeof(half));
        cudaMalloc(&d_k_tbl, kv_dim * sizeof(half));
        cudaMalloc(&d_v,     kv_dim * sizeof(half));
        cudaMalloc(&d_kc_ref, kv_dim * sizeof(half));
        cudaMalloc(&d_kc_tbl, kv_dim * sizeof(half));
        cudaMalloc(&d_vc_ref, kv_dim * sizeof(half));
        cudaMalloc(&d_vc_tbl, kv_dim * sizeof(half));

        fill_random_fp16(d_q_ref, q_dim,  99);
        fill_random_fp16(d_k_ref, kv_dim, 77);
        fill_random_fp16(d_v,     kv_dim, 55);
        cudaMemcpy(d_q_tbl, d_q_ref, q_dim  * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_k_tbl, d_k_ref, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice);

        jllm::rope_clear_tables();
        jllm::rope_inplace_store_kv_fp16(d_q_ref, d_k_ref, d_v, d_kc_ref, d_vc_ref,
                                          n_heads, n_kv_heads, head_dim,
                                          pos, theta, true, nullptr);

        jllm::rope_precompute_table(MAX_SEQ, head_dim, theta);
        jllm::rope_inplace_store_kv_fp16(d_q_tbl, d_k_tbl, d_v, d_kc_tbl, d_vc_tbl,
                                          n_heads, n_kv_heads, head_dim,
                                          pos, theta, true, nullptr);

        cudaDeviceSynchronize();
        const float dq = max_abs_diff(d_q_ref, d_q_tbl, q_dim);
        const float dk = max_abs_diff(d_k_ref, d_k_tbl, kv_dim);
        const float dkc = max_abs_diff(d_kc_ref, d_kc_tbl, kv_dim);
        const float dvc = max_abs_diff(d_vc_ref, d_vc_tbl, kv_dim);
        const bool pass = (dq == 0.0f && dk == 0.0f && dkc == 0.0f && dvc == 0.0f);
        printf("[store_kv Gemma4 local pos=%d]  dQ=%.2e dK=%.2e dKc=%.2e dVc=%.2e  %s\n",
               pos, (double)dq, (double)dk, (double)dkc, (double)dvc, pass ? "PASS" : "FAIL");
        if (!pass) ok = false;

        cudaFree(d_q_ref); cudaFree(d_q_tbl); cudaFree(d_k_ref); cudaFree(d_k_tbl);
        cudaFree(d_v); cudaFree(d_kc_ref); cudaFree(d_kc_tbl);
        cudaFree(d_vc_ref); cudaFree(d_vc_tbl);
    }

    jllm::rope_clear_tables();
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
