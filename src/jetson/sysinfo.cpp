// sysinfo.cpp — One-time Jetson system probe + live stats

#include "jllm_jetson.h"
#include "jllm_memory.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace jllm {

static int read_sysfs_int(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int val = -1;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

static int read_first_sysfs_int(const char* const* paths) {
    for (int i = 0; paths[i]; ++i) {
        int val = read_sysfs_int(paths[i]);
        if (val >= 0) return val;
    }
    return -1;
}

static int read_first_sysfs_mhz(const char* const* paths, int divisor) {
    int val = read_first_sysfs_int(paths);
    return val >= 0 ? val / divisor : 0;
}

static const char* gpu_load_paths[] = {
    "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/load",
    "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/load",
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/load",
    "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/load",
    "/sys/devices/platform/host1x/17000000.gpu/devfreq/17000000.gpu/load",
    "/sys/devices/gpu.0/devfreq/17000000.gp10b/load",
    nullptr
};

static const char* gpu_cur_freq_paths[] = {
    "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
    "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/platform/host1x/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/gpu.0/devfreq/17000000.gp10b/cur_freq",
    nullptr
};

JetsonInfo probe_jetson() {
    JetsonInfo info = {};

    // L4T version
    FILE* f = fopen("/etc/nv_tegra_release", "r");
    if (f) {
        char line[256];
        if (fgets(line, sizeof(line), f)) {
            // Parse "# R36 (release), REVISION: 4.0"
            int major = 0, minor = 0;
            if (sscanf(line, "# R%d (release), REVISION: %d", &major, &minor) >= 1) {
                snprintf(info.l4t_version, sizeof(info.l4t_version), "%d.%d", major, minor);
            }
        }
        fclose(f);
    }

    // CUDA version from runtime
    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    info.cuda_major = runtime_ver / 1000;
    info.cuda_minor = (runtime_ver % 1000) / 10;

    // GPU info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        info.gpu_sm_count = prop.multiProcessorCount;
        // Ampere SM 8.7: 128 CUDA cores per SM (4 warps × 32 threads)
        info.gpu_cuda_cores = prop.multiProcessorCount * 128;
    }

    // RAM
    FILE* mi = fopen("/proc/meminfo", "r");
    if (mi) {
        char line[256];
        while (fgets(line, sizeof(line), mi)) {
            long val;
            if (sscanf(line, "MemTotal: %ld kB", &val) == 1) info.total_ram_mb = val / 1024;
            if (sscanf(line, "CmaTotal: %ld kB", &val) == 1) info.cma_total_mb = val / 1024;
        }
        fclose(mi);
    }

    // NVMe free space
    f = popen("df /dev/nvme0n1p1 2>/dev/null | tail -1 | awk '{print $4}'", "r");
    if (f) {
        long kb = 0;
        fscanf(f, "%ld", &kb);
        info.nvme_free_mb = kb / 1024;
        pclose(f);
    }

    return info;
}

void print_jetson_info(const JetsonInfo& info) {
#ifndef JLLM_VERSION
#define JLLM_VERSION "dev"
#endif
    fprintf(stderr,
        "╔══════════════════════════════════════╗\n"
        "║   genie-ai-runtime v%-16s ║\n"
        "╠══════════════════════════════════════╣\n",
        JLLM_VERSION);
    fprintf(stderr,
        "║ L4T:    %-10s  CUDA: %d.%d       ║\n"
        "║ SMs:    %-3d         Cores: %-5d     ║\n"
        "║ RAM:    %-5ld MB    CMA: %-4ld MB    ║\n"
        "║ NVMe:   %-5ld MB free               ║\n"
        "╚══════════════════════════════════════╝\n",
        info.l4t_version, info.cuda_major, info.cuda_minor,
        info.gpu_sm_count, info.gpu_cuda_cores,
        info.total_ram_mb, info.cma_total_mb,
        info.nvme_free_mb);
}

LiveStats read_live_stats() {
    LiveStats s = {};

    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            long val;
            if (sscanf(line, "MemTotal: %ld kB", &val) == 1) s.ram_total_mb = val / 1024;
            if (sscanf(line, "MemAvailable: %ld kB", &val) == 1)
                s.ram_used_mb = s.ram_total_mb - val / 1024;
        }
        fclose(f);
    }

    // GPU utilization from devfreq load
    s.gpu_util_pct = read_first_sysfs_int(gpu_load_paths);
    if (s.gpu_util_pct < 0) s.gpu_util_pct = 0;

    // GPU frequency
    s.gpu_freq_mhz = read_first_sysfs_mhz(gpu_cur_freq_paths, 1000000);

    // Temperature
    auto ts = read_thermal();
    s.gpu_temp_c = ts.gpu_temp_c;

    return s;
}

void print_live_stats(const LiveStats& s) {
    fprintf(stderr, "\r[RAM %ld/%ld MB | GPU %d%% @ %d MHz | %.1f°C | %.1f tok/s]",
            s.ram_used_mb, s.ram_total_mb,
            s.gpu_util_pct, s.gpu_freq_mhz,
            s.gpu_temp_c, s.tokens_per_sec);
}

}  // namespace jllm
