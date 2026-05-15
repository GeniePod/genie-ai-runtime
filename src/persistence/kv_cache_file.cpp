// kv_cache_file.cpp — Path F (#45) persistent KV cache I/O.
//
// Save / load helpers lifted from tests/test_kv_serialize.cu after F1
// (#46) validated the on-disk format round-trips byte-identical on
// realistic Qwen3-4B-class shapes (144 MB) and rejects truncated files
// via the fstat'd-size-vs-body_bytes check.

#include "kv_cache_file.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <ctime>

namespace jllm {

static bool write_all(int fd, const void* buf, size_t bytes,
                      const char* what, const std::string& path)
{
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t written = 0;
    while (written < bytes) {
        ssize_t n = ::write(fd, p + written, bytes - written);
        if (n <= 0) {
            fprintf(stderr, "[kv_cache] save: %s write failed at %zu/%zu in %s\n",
                    what, written, bytes, path.c_str());
            return false;
        }
        written += (size_t)n;
    }
    return true;
}

bool save_kv_to_file(const std::string& path,
                     const KVCacheFileHeader& hdr,
                     const uint32_t* tokens,
                     size_t n_tokens,
                     const void* host_kv,
                     size_t body_bytes)
{
    if (hdr.body_bytes != body_bytes) {
        fprintf(stderr, "[kv_cache] save: header.body_bytes (%lu) != "
                        "body_bytes (%lu)\n",
                (unsigned long)hdr.body_bytes, (unsigned long)body_bytes);
        return false;
    }
    if (hdr.used_tokens != n_tokens) {
        fprintf(stderr, "[kv_cache] save: header.used_tokens (%u) != "
                        "n_tokens (%zu)\n",
                hdr.used_tokens, n_tokens);
        return false;
    }
    if (n_tokens > 0 && tokens == nullptr) {
        fprintf(stderr, "[kv_cache] save: n_tokens > 0 but tokens=null\n");
        return false;
    }

    std::string tmp_path = path + ".tmp";

    int fd = ::open(tmp_path.c_str(),
                    O_WRONLY | O_CREAT | O_TRUNC,
                    S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        fprintf(stderr, "[kv_cache] save: open(%s) failed: %s\n",
                tmp_path.c_str(), std::strerror(errno));
        return false;
    }

    if (::write(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "[kv_cache] save: header write short\n");
        ::close(fd); ::unlink(tmp_path.c_str()); return false;
    }
    if (n_tokens > 0) {
        const size_t tokens_bytes = n_tokens * sizeof(uint32_t);
        if (!write_all(fd, tokens, tokens_bytes, "tokens", tmp_path)) {
            ::close(fd); ::unlink(tmp_path.c_str()); return false;
        }
    }
    if (!write_all(fd, host_kv, body_bytes, "kv body", tmp_path)) {
        ::close(fd); ::unlink(tmp_path.c_str()); return false;
    }
    if (::fsync(fd) != 0) {
        fprintf(stderr, "[kv_cache] save: fsync failed\n");
        ::close(fd); ::unlink(tmp_path.c_str()); return false;
    }
    ::close(fd);

    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "[kv_cache] save: rename failed: %s\n",
                std::strerror(errno));
        ::unlink(tmp_path.c_str()); return false;
    }
    return true;
}

static bool read_all(int fd, void* buf, size_t bytes,
                     const char* what, const std::string& path)
{
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < bytes) {
        ssize_t n = ::read(fd, p + got, bytes - got);
        if (n <= 0) {
            fprintf(stderr, "[kv_cache] load: %s read failed at %zu/%zu in %s\n",
                    what, got, bytes, path.c_str());
            return false;
        }
        got += (size_t)n;
    }
    return true;
}

bool load_kv_from_file(const std::string& path,
                       KVCacheFileHeader& hdr,
                       std::vector<uint32_t>* out_tokens,
                       void* host_kv,
                       size_t kv_capacity)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        // Caller treats this as "no cache yet" — no log noise here.
        return false;
    }

    if (::read(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "[kv_cache] load: short header in %s\n", path.c_str());
        ::close(fd); return false;
    }
    if (hdr.magic != KVCACHE_MAGIC) {
        fprintf(stderr, "[kv_cache] load: bad magic 0x%x in %s\n",
                hdr.magic, path.c_str());
        ::close(fd); return false;
    }
    if (hdr.version != KVCACHE_VERSION) {
        fprintf(stderr, "[kv_cache] load: unsupported version %u in %s\n",
                hdr.version, path.c_str());
        ::close(fd); return false;
    }
    if (hdr.body_bytes > kv_capacity) {
        fprintf(stderr, "[kv_cache] load: body_bytes (%lu) > "
                        "kv_capacity (%zu) in %s\n",
                (unsigned long)hdr.body_bytes, kv_capacity, path.c_str());
        ::close(fd); return false;
    }

    const size_t tokens_bytes = (size_t)hdr.used_tokens * sizeof(uint32_t);

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        fprintf(stderr, "[kv_cache] load: fstat failed on %s\n", path.c_str());
        ::close(fd); return false;
    }
    size_t expected = sizeof(hdr) + tokens_bytes + (size_t)hdr.body_bytes;
    if ((size_t)st.st_size != expected) {
        fprintf(stderr, "[kv_cache] load: size mismatch on %s: "
                        "disk=%ld expected=%zu (truncated write?)\n",
                path.c_str(), (long)st.st_size, expected);
        ::close(fd); return false;
    }

    if (out_tokens) {
        out_tokens->resize(hdr.used_tokens);
        if (hdr.used_tokens > 0 &&
            !read_all(fd, out_tokens->data(), tokens_bytes, "tokens", path))
        {
            ::close(fd); return false;
        }
    } else if (tokens_bytes > 0) {
        if (::lseek(fd, (off_t)tokens_bytes, SEEK_CUR) == (off_t)-1) {
            ::close(fd); return false;
        }
    }

    if (!read_all(fd, host_kv, (size_t)hdr.body_bytes, "kv body", path)) {
        ::close(fd); return false;
    }
    ::close(fd);
    return true;
}

// FNV-1a-64 fingerprint: (file_size as 8 bytes || first 256 B of file).
// Not cryptographic — purpose is "don't load cache built against a
// different model file". Cheap to compute (256 B + an 8-byte length).
bool compute_model_fingerprint(const std::string& gguf_path,
                               char out_hash[32])
{
    std::memset(out_hash, 0, 32);

    int fd = ::open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[kv_cache] fingerprint: open(%s) failed\n",
                gguf_path.c_str());
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        fprintf(stderr, "[kv_cache] fingerprint: fstat failed on %s\n",
                gguf_path.c_str());
        return false;
    }
    if (st.st_size < 256) {
        ::close(fd);
        fprintf(stderr, "[kv_cache] fingerprint: file too small (%ld B)\n",
                (long)st.st_size);
        return false;
    }

    uint8_t head[256];
    if (::read(fd, head, sizeof(head)) != (ssize_t)sizeof(head)) {
        ::close(fd);
        fprintf(stderr, "[kv_cache] fingerprint: short read on %s\n",
                gguf_path.c_str());
        return false;
    }
    ::close(fd);

    // FNV-1a 64-bit.
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ull;
    uint64_t h = FNV_OFFSET;
    // Mix file size first (network-order-independent — we only ever
    // round-trip on the same machine, so endianness is not a concern).
    uint64_t sz = (uint64_t)st.st_size;
    for (int i = 0; i < 8; i++) {
        h ^= (sz >> (i * 8)) & 0xFFu;
        h *= FNV_PRIME;
    }
    for (int i = 0; i < 256; i++) {
        h ^= head[i];
        h *= FNV_PRIME;
    }

    std::memcpy(out_hash, &h, sizeof(h));   // first 8 bytes; rest stay zero
    return true;
}

bool peek_kv_header(const std::string& path, KVCacheFileHeader* out_hdr) {
    if (!out_hdr) return false;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    KVCacheFileHeader hdr{};
    if (::read(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        ::close(fd); return false;
    }
    if (hdr.magic != KVCACHE_MAGIC) {
        fprintf(stderr, "[kv_cache] peek: bad magic 0x%x in %s\n",
                hdr.magic, path.c_str());
        ::close(fd); return false;
    }
    if (hdr.version != KVCACHE_VERSION) {
        fprintf(stderr, "[kv_cache] peek: unsupported version %u in %s\n",
                hdr.version, path.c_str());
        ::close(fd); return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    ::close(fd);

    const size_t tokens_bytes = (size_t)hdr.used_tokens * sizeof(uint32_t);
    const size_t expected = sizeof(hdr) + tokens_bytes + (size_t)hdr.body_bytes;
    if ((size_t)st.st_size != expected) {
        fprintf(stderr, "[kv_cache] peek: size mismatch on %s "
                        "(disk=%ld expected=%zu)\n",
                path.c_str(), (long)st.st_size, expected);
        return false;
    }
    *out_hdr = hdr;
    return true;
}

std::string default_kv_cache_dir() {
    const char* env = std::getenv("JLLM_KV_CACHE_DIR");
    if (env && *env) return std::string(env);
    return std::string("/opt/jllm/data/kv-cache");
}

bool ensure_dir(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "[kv_cache] ensure_dir(%s) failed: %s\n",
                dir.c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

std::string path_for_conv_id(const std::string& dir, const std::string& conv_id) {
    if (dir.empty() || conv_id.empty()) return {};
    if (dir.back() == '/') return dir + conv_id + ".bin";
    return dir + "/" + conv_id + ".bin";
}

int64_t default_kv_cache_max_mb() {
    const char* v = std::getenv("JLLM_KV_CACHE_MAX_MB");
    if (!v || !*v) return 1024;
    char* end = nullptr;
    long long n = std::strtoll(v, &end, 10);
    if (n < 0) return 0;
    return (int64_t)n;
}

int default_kv_cache_stale_tmp_s() {
    const char* v = std::getenv("JLLM_KV_CACHE_STALE_TMP_S");
    if (!v || !*v) return 60;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (n < 0) return 0;
    return (int)n;
}

namespace {
struct CacheEntry {
    std::string path;
    std::time_t mtime;
    int64_t     size_bytes;
};

// Returns *.bin entries in `dir` with their mtimes and sizes.
static std::vector<CacheEntry> list_kv_files(const std::string& dir,
                                             const char* suffix)
{
    std::vector<CacheEntry> out;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) return out;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!de.is_regular_file(ec) || ec) continue;
        const auto& p = de.path();
        if (p.extension() != suffix) continue;
        struct stat st{};
        if (::stat(p.c_str(), &st) != 0) continue;
        out.push_back({p.string(), st.st_mtime, (int64_t)st.st_size});
    }
    return out;
}
}  // namespace

void enforce_kv_cache_budget(const std::string& dir,
                             int64_t max_bytes,
                             const std::string& keep_path)
{
    if (max_bytes <= 0) return;   // 0 disables eviction
    auto entries = list_kv_files(dir, ".bin");
    int64_t total = 0;
    for (const auto& e : entries) total += e.size_bytes;
    if (total <= max_bytes) return;

    // Oldest first, but skip keep_path even if it would otherwise be evicted.
    std::sort(entries.begin(), entries.end(),
              [](const CacheEntry& a, const CacheEntry& b) {
                  return a.mtime < b.mtime;
              });

    int evicted = 0;
    int64_t freed = 0;
    for (const auto& e : entries) {
        if (total <= max_bytes) break;
        if (e.path == keep_path) continue;
        if (::unlink(e.path.c_str()) == 0) {
            total  -= e.size_bytes;
            freed  += e.size_bytes;
            evicted++;
            fprintf(stderr, "[kv_cache] evicted %s (%ld B, mtime %ld)\n",
                    e.path.c_str(), (long)e.size_bytes, (long)e.mtime);
        }
    }
    if (evicted > 0) {
        fprintf(stderr, "[kv_cache] eviction: %d file(s), %ld B freed, "
                        "total now %ld B / %ld B budget\n",
                evicted, (long)freed, (long)total, (long)max_bytes);
    }
}

void cleanup_stale_tmp_files(const std::string& dir, int max_age_seconds) {
    if (max_age_seconds <= 0) return;
    auto entries = list_kv_files(dir, ".tmp");
    if (entries.empty()) return;

    const std::time_t now = std::time(nullptr);
    for (const auto& e : entries) {
        if ((now - e.mtime) < (std::time_t)max_age_seconds) continue;
        if (::unlink(e.path.c_str()) == 0) {
            fprintf(stderr, "[kv_cache] cleaned stale tmp %s (age %lds)\n",
                    e.path.c_str(), (long)(now - e.mtime));
        }
    }
}

}  // namespace jllm
