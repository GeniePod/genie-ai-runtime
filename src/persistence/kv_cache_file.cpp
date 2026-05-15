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

namespace jllm {

bool save_kv_to_file(const std::string& path,
                     const KVCacheFileHeader& hdr,
                     const void* d_buffer,
                     size_t body_bytes)
{
    if (hdr.body_bytes != body_bytes) {
        fprintf(stderr, "[kv_cache] save: header.body_bytes (%lu) != "
                        "body_bytes (%lu)\n",
                (unsigned long)hdr.body_bytes, (unsigned long)body_bytes);
        return false;
    }

    std::string tmp_path = path + ".tmp";

    std::vector<uint8_t> host_staging(body_bytes);
    cudaError_t err = cudaMemcpy(host_staging.data(), d_buffer, body_bytes,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[kv_cache] save: cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

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
    size_t written = 0;
    while (written < body_bytes) {
        ssize_t n = ::write(fd, host_staging.data() + written,
                            body_bytes - written);
        if (n <= 0) {
            fprintf(stderr, "[kv_cache] save: body write failed at %zu/%zu\n",
                    written, body_bytes);
            ::close(fd); ::unlink(tmp_path.c_str()); return false;
        }
        written += (size_t)n;
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

bool load_kv_from_file(const std::string& path,
                       KVCacheFileHeader& hdr,
                       void* d_buffer,
                       size_t buffer_capacity)
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
    if (hdr.body_bytes > buffer_capacity) {
        fprintf(stderr, "[kv_cache] load: body_bytes (%lu) > "
                        "buffer_capacity (%zu) in %s\n",
                (unsigned long)hdr.body_bytes, buffer_capacity, path.c_str());
        ::close(fd); return false;
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        fprintf(stderr, "[kv_cache] load: fstat failed on %s\n", path.c_str());
        ::close(fd); return false;
    }
    size_t expected = sizeof(hdr) + (size_t)hdr.body_bytes;
    if ((size_t)st.st_size != expected) {
        fprintf(stderr, "[kv_cache] load: size mismatch on %s: "
                        "disk=%ld expected=%zu (truncated write?)\n",
                path.c_str(), (long)st.st_size, expected);
        ::close(fd); return false;
    }

    std::vector<uint8_t> host_staging(hdr.body_bytes);
    size_t read = 0;
    while (read < hdr.body_bytes) {
        ssize_t n = ::read(fd, host_staging.data() + read,
                           hdr.body_bytes - read);
        if (n <= 0) {
            fprintf(stderr, "[kv_cache] load: body read failed at %zu/%lu\n",
                    read, (unsigned long)hdr.body_bytes);
            ::close(fd); return false;
        }
        read += (size_t)n;
    }
    ::close(fd);

    cudaError_t err = cudaMemcpy(d_buffer, host_staging.data(), hdr.body_bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[kv_cache] load: cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
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

}  // namespace jllm
