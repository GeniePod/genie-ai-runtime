cmake_minimum_required(VERSION 3.20)
project(jetson-llm LANGUAGES C CXX CUDA)

# ── Jetson ONLY — no x86, no discrete GPU, no Windows ────────────────────
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(FATAL_ERROR "This project targets Jetson (aarch64) ONLY.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)   # SM 8.7 = Orin Nano/NX/AGX (Ampere)

# Default to Release without FORCE-locking it. Developers can pass
# -DCMAKE_BUILD_TYPE=Debug for nsight-compute / cuda-gdb sessions.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the build type: Release | Debug | RelWithDebInfo | MinSizeRel" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Release Debug RelWithDebInfo MinSizeRel)
endif()

# Orin-specific CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --ptxas-options=-v")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.2-a+fp16 -ffast-math -Wno-format-truncation -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=177")

# Host-side loop unrolling. Helps the per-block dequant loops in
# src/engine/model.cpp::dequant_q4k_row / dequant_q6k_row, which run
# once per prompt token during prefill. -O3 will unroll some loops on
# its own but -funroll-loops is more aggressive and well-tested on aarch64.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")

# Enable link-time optimization (IPO) for Release builds where the
# toolchain supports it. Cross-translation-unit inlining typically buys
# 2-5% on host code with zero correctness risk. CUDA device LTO is
# deliberately not enabled here — that path is fussier and the recent
# kernel revert (65a3d08) showed cross-kernel inlining can mask layout
# bugs. Host-only LTO covers the engine glue, sampling, tokenizer, etc.
include(CheckIPOSupported)
check_ipo_supported(RESULT JLLM_IPO_SUPPORTED OUTPUT JLLM_IPO_ERROR LANGUAGES CXX)
if(JLLM_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Host LTO/IPO: enabled")
else()
    message(STATUS "Host LTO/IPO: disabled (${JLLM_IPO_ERROR})")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Make CUDA headers visible to .cpp files (they include cuda_runtime.h)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ── Sources ──────────────────────────────────────────────────────────────
set(MEMORY_SRC
    src/memory/budget.cpp
    src/memory/kv_cache.cpp
    src/memory/pool.cpp
)

set(JETSON_SRC
    src/jetson/power.cpp
    src/jetson/thermal.cpp
    src/jetson/sysinfo.cpp
)

set(KERNEL_SRC
    src/kernels/gemv_q4.cu
    src/kernels/fused_norm.cu
    src/kernels/attention.cu
    src/kernels/rope.cu
    src/kernels/softmax.cu
    src/kernels/convert.cu
)

set(ENGINE_SRC
    src/engine/model.cpp
    src/engine/decode.cu
    src/engine/sample.cpp
    src/engine/tokenizer.cpp
)

set(SERVER_SRC
    src/server/http_server.cpp
)

# ── Library ──────────────────────────────────────────────────────────────
add_library(jetson_llm_core STATIC
    ${MEMORY_SRC} ${JETSON_SRC} ${KERNEL_SRC} ${ENGINE_SRC}
)
target_include_directories(jetson_llm_core PUBLIC include)
target_link_libraries(jetson_llm_core PUBLIC
    CUDA::cudart CUDA::cublas Threads::Threads
)

# Apply host LTO only when supported. Skipping for CUDA targets
# (gemv_q4.cu et al.) intentionally — see the IPO comment above.
if(JLLM_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET jetson_llm_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ── CLI binary ───────────────────────────────────────────────────────────
add_executable(jetson-llm src/main.cpp)
target_link_libraries(jetson-llm PRIVATE jetson_llm_core)
if(JLLM_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET jetson-llm PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ── Server binary ────────────────────────────────────────────────────────
add_executable(jetson-llm-server src/main_server.cpp ${SERVER_SRC})
target_link_libraries(jetson-llm-server PRIVATE jetson_llm_core)
if(JLLM_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET jetson-llm-server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ── Tests ────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_memory tests/test_memory.cpp)
target_link_libraries(test_memory PRIVATE jetson_llm_core)
add_test(NAME memory COMMAND test_memory)

add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE jetson_llm_core)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_model_load tests/test_model_load.cpp)
target_link_libraries(test_model_load PRIVATE jetson_llm_core)
# Run with: ./build/test_model_load model.gguf
