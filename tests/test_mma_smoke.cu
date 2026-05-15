// test_mma_smoke.cu — Path E phase E1
//
// Smoke test: prove `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
// compiles, launches, and produces the right result on SM 8.7 with our
// current toolchain (nvcc 12.6, CUDA 12.6, L4T R36).
//
// Goal: cheap go/no-go gate before investing in the Path E Q4_K MMQ
// kernel rewrite. If this fails at compile or runtime we know not to
// proceed; if it passes we have the foundational PTX wired up and a
// reference fragment-layout we can re-use in E2.
//
// Issue: #33. PR: path-e/01-mma-smoke-test.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

// ── Shape constants for the smoke instruction ────────────────────────────
//
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32:
//   A is M × K = 16 × 16  FP16, row-major
//   B is K × N = 16 ×  8  FP16, col-major
//   C is M × N = 16 ×  8  FP32 (accumulator, also output D here)
//
// Fragment layout per thread in the 32-thread warp (from PTX ISA 8.4):
//   groupID = lane >> 2       (0..7)
//   tinG    = lane & 3        (0..3)
//
//   A (4 × .b32, each .b32 packs 2 × FP16):
//     a0 = A[groupID    ][tinG*2 + 0..1]
//     a1 = A[groupID + 8][tinG*2 + 0..1]
//     a2 = A[groupID    ][tinG*2 + 8..9]
//     a3 = A[groupID + 8][tinG*2 + 8..9]
//
//   B (2 × .b32, col-major load):
//     b0 = B[tinG*2 + 0..1 ][groupID]
//     b1 = B[tinG*2 + 8..9 ][groupID]
//
//   D (4 × .f32):
//     d0 = D[groupID    ][tinG*2 + 0]
//     d1 = D[groupID    ][tinG*2 + 1]
//     d2 = D[groupID + 8][tinG*2 + 0]
//     d3 = D[groupID + 8][tinG*2 + 1]

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 16;

__global__ void mma_smoke_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float*      __restrict__ D)
{
    const int lane    = threadIdx.x & 31;
    const int groupID = lane >> 2;
    const int tinG    = lane &  3;

    // ── Load A fragment ───────────────────────────────────────────────
    // Pack two FP16 into a .b32 by aliasing half2.
    half2 a0_h = *reinterpret_cast<const half2*>(&A[(groupID    ) * K + (tinG * 2)]);
    half2 a1_h = *reinterpret_cast<const half2*>(&A[(groupID + 8) * K + (tinG * 2)]);
    half2 a2_h = *reinterpret_cast<const half2*>(&A[(groupID    ) * K + (tinG * 2 + 8)]);
    half2 a3_h = *reinterpret_cast<const half2*>(&A[(groupID + 8) * K + (tinG * 2 + 8)]);

    uint32_t a0 = *reinterpret_cast<uint32_t*>(&a0_h);
    uint32_t a1 = *reinterpret_cast<uint32_t*>(&a1_h);
    uint32_t a2 = *reinterpret_cast<uint32_t*>(&a2_h);
    uint32_t a3 = *reinterpret_cast<uint32_t*>(&a3_h);

    // ── Load B fragment (col-major: B has 8 columns of 16 elements each) ──
    // For col-major storage, B[row][col] lives at B[col * K + row].
    half2 b0_h = *reinterpret_cast<const half2*>(&B[groupID * K + (tinG * 2)]);
    half2 b1_h = *reinterpret_cast<const half2*>(&B[groupID * K + (tinG * 2 + 8)]);

    uint32_t b0 = *reinterpret_cast<uint32_t*>(&b0_h);
    uint32_t b1 = *reinterpret_cast<uint32_t*>(&b1_h);

    // ── Zero accumulator ──────────────────────────────────────────────
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // ── Issue the tensor-core MMA ────────────────────────────────────
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );

    // ── Store D fragment ──────────────────────────────────────────────
    D[(groupID    ) * N + (tinG * 2 + 0)] = d0;
    D[(groupID    ) * N + (tinG * 2 + 1)] = d1;
    D[(groupID + 8) * N + (tinG * 2 + 0)] = d2;
    D[(groupID + 8) * N + (tinG * 2 + 1)] = d3;
}

int main(int argc, char** argv) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 8 || (prop.major == 8 && prop.minor < 0)) {
        fprintf(stderr, "FAIL: SM %d.%d does not support mma.sync.m16n8k16.\n",
                prop.major, prop.minor);
        return 1;
    }

    // ── Build test matrices with bounded small ints ──────────────────
    // Values in [-4, 4]; max partial sum has |K| × 16 = 256 magnitude.
    // All operations exact in FP32, exact-or-near-exact in FP16.
    std::vector<half>  h_A(M * K);
    std::vector<half>  h_B(K * N);
    std::vector<float> h_D(M * N);
    std::vector<float> h_D_ref(M * N);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = __float2half(float(((i * 7 + j * 3) % 9) - 4));

    // B col-major: B[row][col] at B[col*K + row]
    for (int r = 0; r < K; r++)
        for (int c = 0; c < N; c++)
            h_B[c * K + r] = __float2half(float(((r * 5 + c * 2) % 9) - 4));

    // Reference matmul (FP32 throughout)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += __half2float(h_A[i * K + k]) *
                     __half2float(h_B[j * K + k]);          // B is col-major
            }
            h_D_ref[i * N + j] = s;
        }
    }

    half  *d_A = nullptr, *d_B = nullptr;
    float *d_D = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_D, h_D.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    mma_smoke_kernel<<<1, 32>>>(d_A, d_B, d_D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, h_D.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ── Compare ──────────────────────────────────────────────────────
    float max_abs = 0.0f;
    int   bad     = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float a = h_D[i * N + j];
            float b = h_D_ref[i * N + j];
            float e = std::fabs(a - b);
            if (e > max_abs) max_abs = e;
            if (e > 1e-2f) bad++;
        }
    }

    printf("Reference D[0][0]=%g  D[15][7]=%g\n",
           h_D_ref[0], h_D_ref[M * N - 1]);
    printf("MMA       D[0][0]=%g  D[15][7]=%g\n",
           h_D[0], h_D[M * N - 1]);
    printf("Max abs error: %g (over %d × %d = %d outputs, %d outside 1e-2)\n",
           max_abs, M, N, M * N, bad);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    if (bad > 0) {
        printf("FAIL: mma.sync.m16n8k16 produced wrong output.\n");
        return 1;
    }
    printf("PASS: mma.sync.m16n8k16.row.col.f32.f16.f16.f32 works on SM %d.%d.\n",
           prop.major, prop.minor);
    return 0;
}
