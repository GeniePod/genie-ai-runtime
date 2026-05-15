// kv_cache_file.h — Path F (#45) persistent KV cache I/O.
//
// On-disk format finalized in F1 / PR #46. The save / load helpers are
// lifted verbatim from tests/test_kv_serialize.cu; F3 hooks save into
// Engine::generate()'s end-of-turn path, F4 will add the hydrate hook
// at the start of generate().

#pragma once

#include <cstdint>
#include <string>

namespace jllm {

constexpr uint32_t KVCACHE_MAGIC   = 0x4D4C4C4Au;   // 'JLLM' LE
constexpr uint32_t KVCACHE_VERSION = 1u;

#pragma pack(push, 1)
struct KVCacheFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_type_bytes;       // 1 = INT8, 2 = FP16
    uint32_t max_context;
    uint32_t used_tokens;
    uint64_t body_bytes;          // defense against truncated writes
    char     model_hash[32];      // FNV-1a-64 of (size + first 256 B of GGUF), zero-padded
    uint8_t  reserved[56];
};
#pragma pack(pop)
static_assert(sizeof(KVCacheFileHeader) == 128, "header must be 128 bytes");

// Atomic save: writes <path>.tmp, fsyncs, renames to <path>. Returns
// false on any I/O failure. host_buffer must already contain the body
// bytes the caller wants persisted (the persistence layer is purely
// I/O — gather from device → host is the caller's responsibility,
// typically via KVCachePool::gather_used_to_host).
bool save_kv_to_file(const std::string& path,
                     const KVCacheFileHeader& hdr,
                     const void* host_buffer,
                     size_t body_bytes);

// Load: validates magic, version, and fstat'd size vs hdr.body_bytes.
// Refuses truncated files. Copies body into host_buffer (must be at
// least hdr.body_bytes in capacity). Scattering to the device pool is
// the caller's responsibility, typically via KVCachePool::scatter_from_host.
bool load_kv_from_file(const std::string& path,
                       KVCacheFileHeader& hdr,
                       void* host_buffer,
                       size_t buffer_capacity);

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

}  // namespace jllm
