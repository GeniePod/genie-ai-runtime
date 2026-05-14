// jllm_kernels.h — Orin-tuned CUDA kernels (SM 8.7 ONLY)
//
// Design rules:
//   1. Every kernel minimizes DRAM traffic (102 GB/s is the bottleneck)
//   2. Tile sizes tuned for 48 KB shared memory per SM
//   3. Thread blocks sized for the 8-SM Orin Nano Super baseline
//   4. Fuse everything possible (fewer kernels = fewer DRAM round-trips)
//   5. INT4 dequant fused into compute (never write dequantized weights to DRAM)
//
// No desktop GPU paths. No Volta. No Turing. Only SM 8.7.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Orin SM 8.7 constants (duplicated here to avoid circular include with jllm.h)
#ifndef JLLM_SHARED_MEM_SM
#define JLLM_SHARED_MEM_SM   (48 * 1024)   // 48 KB per SM
#endif
#ifndef JLLM_WARP_SIZE
#define JLLM_WARP_SIZE       32
#endif

namespace jllm {

// Orin-optimal tile sizes (smaller than desktop — 48 KB shared mem)
constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int ATTENTION_TILE_Q  = 32;   // query tokens per tile
constexpr int ATTENTION_TILE_KV = 64;   // KV tokens per tile
constexpr int BLOCK_SIZE = 128;         // threads per block (4 warps, good occupancy)

// The GGUF loader keeps quantized tensors in the mmap'd model file.  CUDA
// kernels cannot safely dereference those host pointers directly; the loader
// registers the mmap as mapped host memory and records the device-visible base
// address here. Returns nullptr when no mapped alias is available.
const void* resolve_mapped_weight_device_ptr(const void* host_ptr);
void register_mapped_weight_region(const void* host_base, const void* device_base,
                                   int64_t bytes);
void clear_mapped_weight_region(const void* host_base);

// ── Quantized GEMV (decode: batch=1, the hot path) ──────────────────────
// Q4_K dequant fused into GEMV — never writes dequantized weights to DRAM
//
// y[M] = W[M×K] (Q4) × x[K] (FP16)
// W stored as: 4-bit weights + FP16 scales per group of 32
//
void gemv_q4(
    half*          y,          // output [M]
    const void*    W_q4,       // quantized weights [M × K / 2] (4-bit packed)
    const half*    scales,     // per-group scales [M × K / group_size]
    const half*    x,          // input [K]
    int            M,
    int            K,
    int            group_size, // typically 32 or 128
    cudaStream_t   stream
);

// GGML K-quant GEMV dispatcher. Supports Q4_K, Q5_K and Q6_K, which are
// mixed within llama.cpp's *_K_M quantization presets.
void gemv_quant(
    half*          y,
    const void*    W,
    int            ggml_type,
    const half*    x,
    int            M,
    int            K,
    cudaStream_t   stream
);

// Combined K-quant GEMV dispatchers for projections that share the same
// activation vector. These reduce decode launch overhead for Q/K/V and
// gate/up without changing math.
void gemv_quant_pair(
    half*          y0,
    const void*    W0,
    int            ggml_type0,
    int            M0,
    half*          y1,
    const void*    W1,
    int            ggml_type1,
    int            M1,
    const half*    x,
    int            K,
    cudaStream_t   stream
);

void gemv_quant_triple(
    half*          y0,
    const void*    W0,
    int            ggml_type0,
    int            M0,
    half*          y1,
    const void*    W1,
    int            ggml_type1,
    int            M1,
    half*          y2,
    const void*    W2,
    int            ggml_type2,
    int            M2,
    const half*    x,
    int            K,
    cudaStream_t   stream
);

// Batched K-quant GEMM. Computes y[N×M] = x[N×K] · Wᵀ[K×M] where W is
// stored in GGUF K-quant layout (Q4_K / Q5_K / Q6_K — same types as the
// gemv_quant family). Used during prefill to amortize the weight read
// across the N prompt tokens — one DRAM load per weight value, N FMAs
// per load. For N=1 this is just gemv_quant (and the dispatcher routes
// there to keep the decode path unchanged).
//
// Layouts:
//   x : [N × K] half, row-major (token-outer)
//   y : [N × M] half, row-major
//   W : [M × K] K-quant blocks, row-major over output rows
void gemm_quant_batched(
    half*          y,
    const void*    W,
    int            ggml_type,
    const half*    x,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream
);

// K-quant GEMV that writes FP32 output. Used for logits to avoid an
// intermediate FP16 projection plus FP16->FP32 conversion.
void gemv_quant_f32(
    float*         y,
    const void*    W,
    int            ggml_type,
    const half*    x,
    int            M,
    int            K,
    cudaStream_t   stream
);

// Dequantize one token embedding row directly on the GPU. Supports the same
// GGML K-quant formats used by token_embd.weight in Q*_K_M GGUF files.
bool dequant_embedding_row(
    half*          dst,
    const void*    W,
    int            ggml_type,
    int            token_id,
    int            hidden_dim,
    cudaStream_t   stream
);

// ── Fused RMSNorm + Residual Add ────────────────────────────────────────
// input = RMSNorm(x + residual) * weight
// One read of x, one read of residual, one write of output.
// 3× less DRAM traffic than separate kernels.
//
void fused_rmsnorm_residual(
    half*          output,
    const half*    x,
    const half*    residual,
    const void*    weight,     // norm weight [hidden_dim] — FP32 or FP16
    int            rows,
    int            hidden_dim,
    float          eps,
    bool           weight_fp32,// true if weight is float*, false if half*
    cudaStream_t   stream
);

// ── Fused SwiGLU ─────────────────────────────────────────────────────────
// output = silu(gate) * up
// gate and up are computed from the same input via two GEMV calls,
// but this kernel fuses the activation: no intermediate write to DRAM.
//
void fused_swiglu(
    half*          output,
    const half*    gate,       // [rows × intermediate_dim]
    const half*    up,         // [rows × intermediate_dim]
    int            rows,
    int            intermediate_dim,
    cudaStream_t   stream
);

// ── Rotary Position Embedding ────────────────────────────────────────────
// Applied in-place to Q and K before attention.
//
void rope_inplace(
    half*          q,          // [n_heads × head_dim]
    half*          k,          // [n_kv_heads × head_dim]
    int            n_heads,
    int            n_kv_heads,
    int            head_dim,
    int            position,   // token position in sequence
    float          theta_base, // RoPE base (typically 10000 or 500000)
    bool           neox,       // true: pair i with i + head_dim/2
    cudaStream_t   stream
);

// RoPE plus FP16 KV cache store for the common decode path. This replaces
// rope_inplace() followed by two tiny K/V cudaMemcpyAsync calls.
void rope_inplace_store_kv_fp16(
    half*          q,
    half*          k,
    const half*    v,
    half*          k_cache_dst,
    half*          v_cache_dst,
    int            n_heads,
    int            n_kv_heads,
    int            head_dim,
    int            position,
    float          theta_base,
    bool           neox,
    cudaStream_t   stream
);

// ── Flash Attention (single query, decode path) ──────────────────────────
// Fused: Q×K^T → scale → softmax → ×V, all in shared memory + registers.
// Tuned for 48 KB shared mem: TILE_Q=1 (decode), TILE_KV=64.
// KV cache can be FP16 or INT8.
//
void flash_attention_decode(
    half*          output,     // [n_heads × head_dim]
    const half*    q,          // [n_heads × head_dim] (single query)
    const void*    k_cache,    // [seq_len × n_kv_heads × head_dim] (FP16 or INT8)
    const void*    v_cache,    // [seq_len × n_kv_heads × head_dim]
    int            n_heads,
    int            n_kv_heads, // GQA: n_heads / n_kv_heads = group size
    int            head_dim,
    int            seq_len,    // current sequence length
    float          scale,      // 1/sqrt(head_dim)
    bool           kv_int8,    // true = INT8 KV cache
    const float*   kv_scales,  // per-head scales if kv_int8 (else nullptr)
    cudaStream_t   stream
);

// Flash Attention for prefill — N queries against the same K/V cache,
// each with a causal-mask seq_len of `start_pos + token + 1`. One launch
// covers (n_heads × N) blocks; otherwise the per-(head, token) inner
// loop is identical to flash_attention_decode_kernel.
//
// Saves N-1 kernel launches per layer and improves SM occupancy when
// n_heads alone is below the Orin Nano's 8-SM × ~16-block sweet spot.
//
// All K/V positions in [0, start_pos + N) must be written into the cache
// before this call — call the per-token RoPE+KV-store sequence first.
void flash_attention_prefill_batched(
    half*          output,     // [N × n_heads × head_dim]
    const half*    q,          // [N × n_heads × head_dim]
    const void*    k_cache,    // [≥(start_pos+N) × n_kv_heads × head_dim]
    const void*    v_cache,
    int            n_heads,
    int            n_kv_heads,
    int            head_dim,
    int            N,          // number of query tokens
    int            start_pos,  // absolute position of token 0 in the cache
    float          scale,
    bool           kv_int8,
    const float*   kv_scales,
    cudaStream_t   stream
);

// ── Softmax (standalone, for logits) ─────────────────────────────────────
void softmax_inplace(
    float*         x,
    int            n,
    cudaStream_t   stream
);

// ── FP16 ↔ INT8 conversion (for KV cache quantization) ──────────────────
void fp16_to_int8(
    int8_t*        dst,
    float*         scale_out,  // one scale per row
    const half*    src,
    int            rows,
    int            cols,
    cudaStream_t   stream
);

}  // namespace jllm
