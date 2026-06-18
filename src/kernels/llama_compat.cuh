// llama_compat.cuh — minimal ggml-cuda compatibility shim.
//
// genie vendors llama.cpp's MMA abstraction (mma.cuh) and its Q4_K int8 MMQ
// kernel verbatim, to reproduce llama's tuned tensor-core GEMM on Jetson Orin
// (SM 8.7). Those files normally pull ggml's common.cuh; this shim provides
// only the handful of macros/helpers they actually reference, so the vendored
// code compiles standalone inside genie with -arch=sm_87.
//
// Definitions are copied faithfully from llama.cpp ggml/src/ggml-cuda/common.cuh
// and ggml/include/ggml.h (MIT licensed). Only the NVIDIA (non-HIP) path is kept.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---- warp / compute-capability constants (common.cuh) ----
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define GGML_CUDA_CC_VOLTA   700
#define GGML_CUDA_CC_TURING  750
#define GGML_CUDA_CC_AMPERE  800

// ---- tensor-core availability gates (common.cuh, NVIDIA path only) ----
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif

// ---- diagnostics / unused (ggml.h, common.cuh) ----
// NO_DEVICE_CODE only guards branches for unsupported arches; on sm_87 those
// branches are dead, so a no-op is safe.
#define NO_DEVICE_CODE
#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)
#endif
#ifndef GGML_UNUSED_VARS
template <typename... Ts> __host__ __device__ inline void ggml_unused_vars_impl(Ts &&...) {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)
#endif

// ---- coalesced small-copy helpers (common.cuh) ----
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "misusing alignment parameter for ggml_cuda_memcpy_1");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char  *) dst)[i] = ((const char  *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int   *) dst)[i] = ((const int   *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2  *) dst)[i] = ((const int2  *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4  *) dst)[i] = ((const int4  *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
