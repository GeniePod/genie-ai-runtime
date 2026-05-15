// test_kv_serialize.cu — Path F phase F1
//
// Validate the KV cache serialization round-trip: device buffer →
// disk file → separate device buffer → memcmp byte-identical.
//
// No engine integration. This file defines the v1 header layout and
// the save/load helpers as static functions; F3 will lift them into
// `src/persistence/` and wire them into the decode loop. F1 only
// proves that:
//
//   1. The header struct round-trips correctly (magic, version, shape).
//   2. The body round-trips with byte-equality across realistic KV
//      sizes (Qwen3-4B-class: 36 layers × 8 KV heads × 128 head_dim
//      × FP16 × 1024 max_context = ~144 MB).
//   3. The atomic-rename + body-bytes-check pattern rejects truncated
//      files instead of loading garbage.
//
// Issue: #45. PR: path-f/01-kv-serialize-roundtrip.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

// ── KV cache file format v1 ─────────────────────────────────────────────

static constexpr uint32_t KVCACHE_MAGIC   = 0x4D4C4C4Au;   // 'JLLM' LE
static constexpr uint32_t KVCACHE_VERSION = 1u;

#pragma pack(push, 1)
struct KVCacheFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_type_bytes;
    uint32_t max_context;
    uint32_t used_tokens;
    uint64_t body_bytes;
    char     model_hash[32];
    uint8_t  reserved[56];   // pad to 128:
                             //   8 × uint32 (32) + uint64 (8) + 32 (hash) = 72
                             //   + reserved 56 = 128
};
#pragma pack(pop)
static_assert(sizeof(KVCacheFileHeader) == 128, "header must be 128 bytes");

// ── Save / load helpers (lifted into src/persistence/ in F3) ────────────

// Atomic save: write to <path>.tmp, fsync, rename.
static bool save_kv_to_file(const char* path,
                            const KVCacheFileHeader& hdr,
                            const void* d_buffer,
                            size_t body_bytes)
{
    if (hdr.body_bytes != body_bytes) {
        fprintf(stderr, "save_kv_to_file: header.body_bytes (%lu) "
                        "!= body_bytes (%lu)\n",
                (unsigned long)hdr.body_bytes, (unsigned long)body_bytes);
        return false;
    }

    std::string tmp_path = std::string(path) + ".tmp";

    std::vector<uint8_t> host_staging(body_bytes);
    CHECK_CUDA(cudaMemcpy(host_staging.data(), d_buffer, body_bytes,
                          cudaMemcpyDeviceToHost));

    int fd = ::open(tmp_path.c_str(),
                    O_WRONLY | O_CREAT | O_TRUNC,
                    S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        fprintf(stderr, "save_kv_to_file: open(%s) failed\n", tmp_path.c_str());
        return false;
    }

    if (::write(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "save_kv_to_file: header write short\n");
        ::close(fd); ::unlink(tmp_path.c_str()); return false;
    }
    size_t written = 0;
    while (written < body_bytes) {
        ssize_t n = ::write(fd, host_staging.data() + written,
                            body_bytes - written);
        if (n <= 0) {
            fprintf(stderr, "save_kv_to_file: body write failed at %zu/%zu\n",
                    written, body_bytes);
            ::close(fd); ::unlink(tmp_path.c_str()); return false;
        }
        written += (size_t)n;
    }
    if (::fsync(fd) != 0) {
        fprintf(stderr, "save_kv_to_file: fsync failed\n");
        ::close(fd); ::unlink(tmp_path.c_str()); return false;
    }
    ::close(fd);

    if (::rename(tmp_path.c_str(), path) != 0) {
        fprintf(stderr, "save_kv_to_file: rename failed\n");
        ::unlink(tmp_path.c_str()); return false;
    }
    return true;
}

// Load: validates header, refuses on truncated body. On success, copies
// body into d_buffer (which must be at least header.body_bytes long).
static bool load_kv_from_file(const char* path,
                              KVCacheFileHeader& hdr,
                              void* d_buffer,
                              size_t buffer_capacity)
{
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "load_kv_from_file: open(%s) failed\n", path);
        return false;
    }

    if (::read(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "load_kv_from_file: short header\n");
        ::close(fd); return false;
    }
    if (hdr.magic != KVCACHE_MAGIC) {
        fprintf(stderr, "load_kv_from_file: bad magic 0x%x\n", hdr.magic);
        ::close(fd); return false;
    }
    if (hdr.version != KVCACHE_VERSION) {
        fprintf(stderr, "load_kv_from_file: unsupported version %u\n",
                hdr.version);
        ::close(fd); return false;
    }
    if (hdr.body_bytes > buffer_capacity) {
        fprintf(stderr, "load_kv_from_file: body_bytes (%lu) > "
                        "buffer_capacity (%zu)\n",
                (unsigned long)hdr.body_bytes, buffer_capacity);
        ::close(fd); return false;
    }

    // Defense against truncated writes: stat the file and refuse if the
    // on-disk size doesn't match header + body_bytes.
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        fprintf(stderr, "load_kv_from_file: fstat failed\n");
        ::close(fd); return false;
    }
    size_t expected = sizeof(hdr) + (size_t)hdr.body_bytes;
    if ((size_t)st.st_size != expected) {
        fprintf(stderr, "load_kv_from_file: size mismatch on disk %ld "
                        "vs expected %zu\n",
                (long)st.st_size, expected);
        ::close(fd); return false;
    }

    std::vector<uint8_t> host_staging(hdr.body_bytes);
    size_t read = 0;
    while (read < hdr.body_bytes) {
        ssize_t n = ::read(fd, host_staging.data() + read,
                           hdr.body_bytes - read);
        if (n <= 0) {
            fprintf(stderr, "load_kv_from_file: body read failed at %zu/%lu\n",
                    read, (unsigned long)hdr.body_bytes);
            ::close(fd); return false;
        }
        read += (size_t)n;
    }
    ::close(fd);

    CHECK_CUDA(cudaMemcpy(d_buffer, host_staging.data(), hdr.body_bytes,
                          cudaMemcpyHostToDevice));
    return true;
}

// ── Test scaffolding ────────────────────────────────────────────────────

// Deterministic pattern keyed on (layer, pos, byte_offset_within_token).
// Using a simple xorshift-style hash so the data isn't trivially zero
// and any layout slip-up shows up as non-equality.
static void fill_pattern(std::vector<uint8_t>& host, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (auto& b : host) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        b = (uint8_t)(s & 0xFF);
    }
}

struct ShapeCase {
    const char* label;
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int kv_type_bytes;
    int max_context;
};

static bool run_roundtrip(const ShapeCase& s) {
    const int64_t entry_bytes  = 2LL * s.n_kv_heads * s.head_dim * s.kv_type_bytes;
    const int64_t body_bytes   = (int64_t)s.n_layers * s.max_context * entry_bytes;
    const int64_t body_mb      = body_bytes / (1024 * 1024);

    printf("\n── %s (n_layers=%d kv_heads=%d head_dim=%d kv_bytes=%d "
           "max_ctx=%d → %ld MB) ──\n",
           s.label, s.n_layers, s.n_kv_heads, s.head_dim,
           s.kv_type_bytes, s.max_context, (long)body_mb);

    // Source-of-truth host pattern, copy to device A.
    std::vector<uint8_t> host_src(body_bytes);
    fill_pattern(host_src, 0xC0FFEEu + (uint32_t)s.max_context);

    void* d_src = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, body_bytes));
    CHECK_CUDA(cudaMemcpy(d_src, host_src.data(), body_bytes,
                          cudaMemcpyHostToDevice));

    KVCacheFileHeader hdr{};
    hdr.magic         = KVCACHE_MAGIC;
    hdr.version       = KVCACHE_VERSION;
    hdr.n_layers      = (uint32_t)s.n_layers;
    hdr.n_kv_heads    = (uint32_t)s.n_kv_heads;
    hdr.head_dim      = (uint32_t)s.head_dim;
    hdr.kv_type_bytes = (uint32_t)s.kv_type_bytes;
    hdr.max_context   = (uint32_t)s.max_context;
    hdr.used_tokens   = (uint32_t)(s.max_context / 4);   // partial-fill realism
    hdr.body_bytes    = (uint64_t)body_bytes;
    std::memset(hdr.model_hash, 0, sizeof(hdr.model_hash));
    std::memset(hdr.reserved,   0, sizeof(hdr.reserved));

    const char* path = "/tmp/jllm_kv_roundtrip.bin";

    if (!save_kv_to_file(path, hdr, d_src, (size_t)body_bytes)) {
        fprintf(stderr, "FAIL: save_kv_to_file returned false\n");
        cudaFree(d_src); return false;
    }

    void* d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_dst, body_bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0xAA, body_bytes));   // poison

    KVCacheFileHeader hdr2{};
    if (!load_kv_from_file(path, hdr2, d_dst, (size_t)body_bytes)) {
        fprintf(stderr, "FAIL: load_kv_from_file returned false\n");
        cudaFree(d_src); cudaFree(d_dst); return false;
    }

    // Header field-by-field check.
    if (hdr2.magic         != hdr.magic         ||
        hdr2.version       != hdr.version       ||
        hdr2.n_layers      != hdr.n_layers      ||
        hdr2.n_kv_heads    != hdr.n_kv_heads    ||
        hdr2.head_dim      != hdr.head_dim      ||
        hdr2.kv_type_bytes != hdr.kv_type_bytes ||
        hdr2.max_context   != hdr.max_context   ||
        hdr2.used_tokens   != hdr.used_tokens   ||
        hdr2.body_bytes    != hdr.body_bytes)
    {
        fprintf(stderr, "FAIL: header field mismatch\n");
        cudaFree(d_src); cudaFree(d_dst); return false;
    }

    // Body byte-identity.
    std::vector<uint8_t> host_dst(body_bytes);
    CHECK_CUDA(cudaMemcpy(host_dst.data(), d_dst, body_bytes,
                          cudaMemcpyDeviceToHost));
    if (std::memcmp(host_src.data(), host_dst.data(), (size_t)body_bytes) != 0) {
        // Find first divergence for diagnosis.
        for (size_t i = 0; i < (size_t)body_bytes; i++) {
            if (host_src[i] != host_dst[i]) {
                fprintf(stderr,
                        "FAIL: body diverges at offset %zu (src=0x%02x dst=0x%02x)\n",
                        i, host_src[i], host_dst[i]);
                break;
            }
        }
        cudaFree(d_src); cudaFree(d_dst); return false;
    }

    printf("  PASS: %ld MB round-tripped byte-identical, header preserved.\n",
           (long)body_mb);

    cudaFree(d_src);
    cudaFree(d_dst);
    ::unlink(path);
    return true;
}

// Truncated-file rejection test: write a real KV file, truncate it, expect
// load to reject rather than load garbage.
static bool run_truncated_rejection() {
    printf("\n── Truncated-file rejection ──\n");

    ShapeCase s{"truncate", 4, 8, 64, 2, 128};
    const int64_t entry_bytes = 2LL * s.n_kv_heads * s.head_dim * s.kv_type_bytes;
    const int64_t body_bytes  = (int64_t)s.n_layers * s.max_context * entry_bytes;

    std::vector<uint8_t> host_src(body_bytes);
    fill_pattern(host_src, 0xBADBEEFu);

    void* d_src = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, body_bytes));
    CHECK_CUDA(cudaMemcpy(d_src, host_src.data(), body_bytes,
                          cudaMemcpyHostToDevice));

    KVCacheFileHeader hdr{};
    hdr.magic = KVCACHE_MAGIC; hdr.version = KVCACHE_VERSION;
    hdr.n_layers = s.n_layers; hdr.n_kv_heads = s.n_kv_heads;
    hdr.head_dim = s.head_dim; hdr.kv_type_bytes = s.kv_type_bytes;
    hdr.max_context = s.max_context; hdr.used_tokens = 0;
    hdr.body_bytes = (uint64_t)body_bytes;

    const char* path = "/tmp/jllm_kv_truncated.bin";
    if (!save_kv_to_file(path, hdr, d_src, (size_t)body_bytes)) {
        cudaFree(d_src); return false;
    }

    // Truncate to half the body. Header is intact but on-disk size will
    // disagree with header.body_bytes, so load must reject.
    if (::truncate(path, sizeof(KVCacheFileHeader) + body_bytes / 2) != 0) {
        fprintf(stderr, "truncate failed\n");
        cudaFree(d_src); return false;
    }

    void* d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_dst, body_bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0xAA, body_bytes));

    KVCacheFileHeader hdr2{};
    bool loaded = load_kv_from_file(path, hdr2, d_dst, (size_t)body_bytes);

    cudaFree(d_src);
    cudaFree(d_dst);
    ::unlink(path);

    if (loaded) {
        fprintf(stderr, "FAIL: load accepted a truncated file\n");
        return false;
    }
    printf("  PASS: truncated file correctly rejected.\n");
    return true;
}

int main(int, char**) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);
    printf("Header size: %zu bytes (must be 128).\n",
           sizeof(KVCacheFileHeader));

    // Cover a small synthetic case, a realistic Qwen3-4B-class case, and
    // a wide one to exercise the size handling.
    const ShapeCase cases[] = {
        {"small",        4, 8, 64,  2, 128},                  // ~256 KB
        {"qwen3-4b",    36, 8, 128, 2, 1024},                 // ~144 MB
        {"int8_kv",     36, 8, 128, 1, 1024},                 // ~72 MB
    };

    bool ok = true;
    for (const auto& c : cases) {
        ok = run_roundtrip(c) && ok;
    }
    ok = run_truncated_rejection() && ok;

    if (!ok) {
        printf("\nFAIL: at least one case did not round-trip.\n");
        return 1;
    }
    printf("\nPASS: all KV cache serialization round-trips succeeded.\n");
    return 0;
}
