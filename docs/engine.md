# Engine

## Lifecycle

```
Engine engine;
engine.load("model.gguf", params);   // parse GGUF, mmap, allocate pools
engine.generate("Hello", params, cb); // tokenize → prefill → decode → stream
engine.unload();                      // free everything
```

## load()

1. Probe system memory (`probe_system_memory()`)
2. Parse GGUF config (`load_gguf_config()`)
3. Load and map weights (`load_and_map_weights()`) — mmap + cudaHostRegister + tensor name matching
4. Auto-calculate max context from remaining memory
5. Allocate KV cache pool (pinned fast + unpinned overflow)
6. Allocate scratch pool (bump allocator)
7. Create CUDA stream
8. Load tokenizer from GGUF
9. Print memory budget

## Transformer Layer

`transformer_layer(layer, pos, x)` — 12 operations per layer:

```
Input: x [hidden_dim] — hidden state from previous layer

┌─ Attention Block ────────────────────────────────────┐
│  1. normed = RMSNorm(x) × attn_weight               │
│  2. Q = gemv_q4(W_q, normed)                         │
│  3. K = gemv_q4(W_k, normed)                         │
│  4. V = gemv_q4(W_v, normed)                         │
│  5. RoPE(Q, K, position)                              │
│  6. KV cache store (INT8 quantize if enabled)         │
│  7. attn_out = flash_attention(Q, K_cache, V_cache)   │
│  8. attn_proj = gemv_q4(W_o, attn_out)                │
│  9. x2 = x + attn_proj              ← residual #1    │
└──────────────────────────────────────────────────────┘

┌─ FFN Block ──────────────────────────────────────────┐
│  10. normed2 = RMSNorm(x2) × ffn_weight              │
│  11. gate = gemv_q4(W_gate, normed2)                  │
│  12. up = gemv_q4(W_up, normed2)                      │
│  13. swiglu_out = silu(gate) × up                     │
│  14. ffn_out = gemv_q4(W_down, swiglu_out)            │
│  15. x = x2 + ffn_out               ← residual #2    │
└──────────────────────────────────────────────────────┘

Output: x [hidden_dim] — input to next layer
```

All intermediate buffers allocated from `ScratchPool` (reset each decode step).

## Decode Step

`decode_step(pos)` — one token generation:

1. Get hidden state buffer from scratch
2. Embedding lookup: `cudaMemcpyAsync(x, tok_embd + token_id × hidden_dim)`
3. Run `transformer_layer()` for all N layers
4. Final RMSNorm
5. Logit projection: `gemv_q4(W_output, normed)` → FP16
6. Convert FP16 → FP32 on GPU (`fp16_to_fp32` kernel)
7. Copy FP32 logits to CPU (`cudaMemcpy D2H`)
8. Sample: `sample_token(logits, vocab_size, params)`
9. Update recent tokens (for repeat penalty)
10. Return token ID

## Generation Loop

`generate(prompt, params, callback)`:

```
Tokenize prompt → token IDs
│
├── Prefill phase (Path B, default):
│   Allocate x_batch[N × hidden_dim] in scratch
│   Dequant all prompt embeddings into x_batch rows
│   For each layer l in [0, n_layers):
│     transformer_prefill(l, start_pos=0, N, x_batch)
│   Path A: sample first token from x_batch[N-1] (skip a forward pass)
│
├── Decode phase:
│   For each output token (up to max_tokens):
│     check_memory_and_thermal()  ← OOM guard + thermal
│     scratch.reset()
│     token = decode_step(pos)
│     callback(detokenized_text, is_eos)
│     if EOS: break
│
└── Return GenStats
```

Set `JLLM_BATCHED_PREFILL=0` to fall back to the original per-token
prefill (one token at a time, calling `transformer_layer` directly).

## transformer_prefill — Layer-major Batched Prefill

`transformer_prefill(layer, start_pos, n_tokens, x_batch)` — one layer
across all N prompt tokens. Maintains a set of persistent batched
buffers (allocated at the function's scratch snapshot) and combines
batched kernels with two short per-token loops:

```
Persistent batched buffers in scratch (rewound on exit):
  normed_batch    [N × H]
  q_batch         [N × Q_DIM]
  k_batch         [N × KV_DIM]
  v_batch         [N × KV_DIM]
  attn_out_batch  [N × Q_DIM]
  attn_proj_batch [N × H]
  x_attn_batch    [N × H]
  normed2_batch   [N × H]
  gate_batch      [N × I]
  up_batch        [N × I]
  swiglu_batch    [N × I]
  ffn_out_batch   [N × H]

┌─ Attention ──────────────────────────────────────────────────────┐
│  1. fused_rmsnorm_residual  (rows=N)                             │
│  2. gemm_quant_batched × 3  → q_batch, k_batch, v_batch          │
│  3. fused_rmsnorm_residual  Qwen3 QK-norm (rows=N × n_heads /    │
│     N × n_kv_heads)                                              │
│  4. Per-token loop (RoPE + KV store — sequential, causal):       │
│       rope_inplace_store_kv_fp16(q[i], k[i], v[i], cache@pos+i)  │
│  5. flash_attention_prefill_batched  one launch, grid=(n_heads,N)│
│     → attn_out_batch                                             │
│  6. gemm_quant_batched(Wo)  → attn_proj_batch                    │
│  7. vec_add(x_attn_batch ← x_batch + attn_proj_batch, n=N×H)     │
└──────────────────────────────────────────────────────────────────┘

┌─ FFN ────────────────────────────────────────────────────────────┐
│  8.  fused_rmsnorm_residual  (rows=N)                            │
│  9.  gemm_quant_batched × 2  → gate_batch, up_batch              │
│  10. fused_swiglu  (rows=N)                                      │
│  11. gemm_quant_batched(W_down)  → ffn_out_batch                 │
│  12. vec_add(x_batch ← x_attn_batch + ffn_out_batch, n=N×H)      │
└──────────────────────────────────────────────────────────────────┘
```

Every weight matrix in the layer runs through one `gemm_quant_batched`
call — each weight value is read from DRAM **once** and re-used across
all `N` query tokens, instead of being re-streamed `N` times by
`gemv_quant`.

The only per-token work that remains is RoPE + KV store: token `i`'s
K/V must be written to the cache before token `i+1` reads it via the
causal attention mask. The attention itself is then batched into one
launch via `flash_attention_prefill_batched`.

## transformer_layer — Per-token (decode + opt-out prefill)

`transformer_layer(layer, pos, x)` runs the same logic for one token.
Used by `decode_step` and by the `JLLM_BATCHED_PREFILL=0` prefill
fallback. Composed of:

```
1. RMSNorm
2. gemv_quant_triple  → q, k, v
3. transformer_layer_attn_block:
     a. attn_compute: QK-norm (cond) + RoPE + KV store + attention
     b. gemv_quant(Wo) + vec_add(first residual)
4. RMSNorm + gemv_quant_pair(gate/up) + swiglu (per-token)
5. transformer_layer_ffn_block:
     gemv_quant(W_down) + vec_add(second residual)
```

`transformer_layer_attn_compute` is a private helper called from both
`attn_block` (per-token) and `transformer_prefill` (batched dispatch).

## CUDA Graphs

`build_cuda_graph(pos)` captures the GPU-side decode step as a CUDA graph:

- All transformer layers + final norm + logit projection captured
- Replayed with `cudaGraphLaunch()` for subsequent tokens
- Reduces kernel launch overhead from ~1ms to ~5μs per token
- Not captured: embedding lookup (host→device), sampling (host-side)
- Graph must be rebuilt if KV cache structure changes

## Sampling

`sample_token()` (`src/engine/sample.cpp`) — CPU-side token selection:

1. Apply repeat penalty (penalize recent tokens)
2. Apply temperature (divide logits by T)
3. If T=0: greedy (argmax)
4. Softmax on CPU (logits are small: vocab_size × 4 bytes)
5. Top-K filter (keep K highest, partial sort)
6. Top-P filter (keep until cumulative probability > P)
7. Random sample from filtered distribution

## Tokenizer

`Tokenizer` (`src/engine/tokenizer.cpp`):

### Encoding

Uses hash map `token_to_id_` for O(max_token_len) per position:
1. At each position, try longest match first (decreasing length)
2. Hash map lookup for each candidate substring
3. If no match: byte fallback (`<0x41>` → byte token)

### Decoding

Direct lookup: `vocab[token_id]`. Handles byte tokens (`<0xNN>` → actual byte).

## Stop Mechanism

`engine.stop()` sets `stop_flag_ = true`. The decode loop checks this every iteration and breaks cleanly. Used for:
- SIGINT handler (Ctrl+C in CLI)
- HTTP request cancellation
- Timeout

## GenStats

Returned after generation:

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Number of prompt tokens processed |
| `completion_tokens` | Number of tokens generated |
| `prompt_ms` | Time for prefill phase |
| `decode_ms` | Time for decode phase |
| `ttft_ms` | Time-to-first-token: wall-clock from generate() entry to first sampled token delivered (tokenization + prefill + first decode step) |
| `prompt_tok_per_sec` | Prefill throughput |
| `decode_tok_per_sec` | Decode throughput |
| `peak_memory_mb` | Maximum memory usage observed |
| `peak_thermal_c` | Maximum GPU temperature observed |
| `oom_stops` | Times OOM guard stopped generation |
| `thermal_pauses` | Times thermal backoff triggered |
