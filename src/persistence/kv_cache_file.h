// kv_cache_file.h — Path F (#45) persistent KV cache I/O.
//
// On-disk format finalized in F1 / PR #46. The save / load helpers are
// lifted verbatim from tests/test_kv_serialize.cu; F3 hooks save into
// Engine::generate()'s end-of-turn path, F4 will add the hydrate hook
// at the start of generate().

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jllm {

constexpr uint32_t KVCACHE_MAGIC   = 0x4D4C4C4Au;   // 'JLLM' LE

// Format versions:
//   v1 — F3 / F3b. Layout: [header 128 B][KV body body_bytes]. KV body
//        packed per layer (used_tokens × eb keys + values). No token IDs.
//   v2 — F4a. Layout: [header 128 B][tokens used_tokens × 4 B]
//        [KV body body_bytes]. Token IDs added so F4b can find the
//        longest common prefix between a cached conversation and a
//        new prompt, and skip prefill for the matched range.
//
// Old v1 files cannot be loaded by v2 readers; they fail the size check
// (file size disagrees with sizeof(header) + tokens_bytes + body_bytes).
// Path F is alpha-only so far, no migration path needed.
constexpr uint32_t KVCACHE_VERSION = 2u;

#pragma pack(push, 1)
struct KVCacheFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_type_bytes;       // 1 = INT8, 2 = FP16
    uint32_t max_context;
    uint32_t used_tokens;         // also = number of token IDs in the tokens region
    uint64_t body_bytes;          // KV portion only; tokens region is used_tokens × 4 B
    char     model_hash[32];      // FNV-1a-64 of (size + first 256 B of GGUF), zero-padded
    uint8_t  reserved[56];
};
#pragma pack(pop)
static_assert(sizeof(KVCacheFileHeader) == 128, "header must be 128 bytes");

// Atomic save: writes <path>.tmp, fsyncs, renames to <path>. Returns
// false on any I/O failure. host_kv must already contain the KV body
// bytes the caller wants persisted (gather from device → host is the
// caller's responsibility, typically via KVCachePool::gather_used_to_host).
// tokens points to hdr.used_tokens uint32 token IDs; the caller is
// responsible for collecting these (engine has them in tokenizer.encode
// output). Pass tokens=nullptr / n_tokens=0 only for tests that don't
// need the v2 prefix-match logic — production should always pass them.
bool save_kv_to_file(const std::string& path,
                     const KVCacheFileHeader& hdr,
                     const uint32_t* tokens,
                     size_t n_tokens,
                     const void* host_kv,
                     size_t body_bytes);

// Load: validates magic, version, and fstat'd size vs
// sizeof(header) + n_tokens * 4 + body_bytes. Refuses truncated files.
// Reads token IDs into *out_tokens (resized to hdr.used_tokens) and KV
// body into host_kv (must be at least hdr.body_bytes in capacity).
bool load_kv_from_file(const std::string& path,
                       KVCacheFileHeader& hdr,
                       std::vector<uint32_t>* out_tokens,
                       void* host_kv,
                       size_t kv_capacity);

// Peek-only: read and validate just the header. Used by callers that
// need to size a body buffer before calling load_kv_from_file. Returns
// false if the file doesn't exist, isn't a valid v2 KV cache, or has
// an on-disk size that disagrees with what the header advertises.
bool peek_kv_header(const std::string& path, KVCacheFileHeader* out_hdr);

// FNV-1a-64 over (file_size as 8 bytes || first 256 B of file). The
// first 256 B of a GGUF file covers magic + version + metadata count
// + the start of the metadata KV pairs, which is sufficient to detect
// almost any model swap without re-reading the multi-GB body. Written
// into the first 8 bytes of hdr.model_hash, remaining 24 bytes zeroed.
//
// Returns false if the file cannot be opened or is shorter than 256 B.
bool compute_model_fingerprint(const std::string& gguf_path,
                               char out_hash[32]);

// Returns env JLLM_KV_CACHE_DIR if set, else "/opt/jllm/data/kv-cache".
std::string default_kv_cache_dir();

// mkdir -p equivalent. Returns true if dir exists or was created.
bool ensure_dir(const std::string& dir);

// Joins dir + "/" + conv_id + ".bin". Assumes conv_id has already been
// validated against jllm::validate_conversation_id().
std::string path_for_conv_id(const std::string& dir, const std::string& conv_id);

// Path F5: per-process budget for `dir`'s *.bin files, in MB. Reads
// JLLM_KV_CACHE_MAX_MB; defaults to 1024 (1 GB). 0 disables eviction.
int64_t default_kv_cache_max_mb();

// Path F5: age threshold for stale .tmp cleanup, in seconds. Reads
// JLLM_KV_CACHE_STALE_TMP_S; defaults to 60.
int default_kv_cache_stale_tmp_s();

// Path F5: walk `dir`, delete oldest *.bin files by mtime until the
// total size of remaining *.bin files is ≤ max_bytes. Never deletes
// `keep_path` (the file we just saved). Logs each eviction. No-op if
// max_bytes <= 0 (eviction disabled).
void enforce_kv_cache_budget(const std::string& dir,
                             int64_t max_bytes,
                             const std::string& keep_path);

// Path F5: unlink any *.tmp file in `dir` whose mtime is older than
// `max_age_seconds`. Cleans up after a previous run that crashed
// mid-save before the .tmp → final rename. No-op if max_age_seconds <= 0.
void cleanup_stale_tmp_files(const std::string& dir, int max_age_seconds);

}  // namespace jllm
