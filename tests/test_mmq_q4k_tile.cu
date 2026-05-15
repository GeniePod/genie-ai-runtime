// test_mmq_q4k_tile.cu — Path E phase E2
//
// Minimal Q4_K matrix-multiply skeleton built on tensor cores. Tests the
// pipeline we'll productionize in E3+:
//
//   1. Dequantize a Q4_K sub-block (32 quants) for 16 weight-matrix rows
//      into a 16×32 FP16 staging tile in shared memory.
//   2. Issue two m16n8k16 MMAs across that 32-wide tile.
//   3. Loop over all 8 sub-blocks of the Q4_K block (full K=256).
//   4. Compare a single 16×8 FP32 output tile to a host reference.
//
// Problem size is deliberately tiny (M=16, N=8, K=256 — exactly one
// block_q4_K per output row). No engine integration, no perf claim.
// The point is to validate the math + fragment layout + dequant-into-
// shared-memory flow before we wire any of this into the prefill path.
//
// Pass criterion: max abs error < 1e-2 between device MMA output and
// host scalar reference. Both paths use the same FP16 dequantization,
// so the only remaining source of difference is FP32 accumulator order
// inside the MMA — which should round-trip to zero at this problem size.
//
// Issue: #33. PR: path-e/02-q4k-mmq-tile.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

// ── Q4_K block (matches src/kernels/gemv_q4.cu) ──────────────────────────
//
// 144 B per block. 256 quants, organized as 8 sub-blocks of 32. Each
// sub-block has a 6-bit scale and 6-bit min packed across the 12-byte
// scales[] field via get_scale_min_k4.

constexpr int QK_K = 256;

struct __attribute__((packed)) block_q4_K {
    uint16_t d_raw;
    uint16_t dmin_raw;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0)         result = std::ldexp((float)mant, -24);
    else if (exp == 31)   result = mant ? NAN : INFINITY;
    else                  result = std::ldexp((float)(mant + 1024), (int)exp - 25);
    return sign ? -result : result;
#endif
}

__host__ __device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ── E2 skeleton kernel ──────────────────────────────────────────────────
//
// One warp computes a single M=16, N=8 output tile from M=16 Q4_K block
// rows × K=256 columns of weights and a (K=256, N=8) col-major
// activation matrix. Eight sub-blocks × 2 MMA-K tiles = 16 MMAs total.

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = QK_K;     // 256, one Q4_K block per row

__global__ void mmq_q4k_tile_kernel(const block_q4_K* __restrict__ W,
                                    const half*       __restrict__ X,
                                    float*            __restrict__ D)
{
    const int lane    = threadIdx.x & 31;
    const int groupID = lane >> 2;
    const int tinG    = lane &  3;

    // Persistent D accumulator across all 16 MMAs.
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Shared-mem staging tile: 16 rows × 32 cols of FP16. One sub-block
    // at a time. 16 × 32 × 2 = 1 KiB per warp; fits trivially.
    __shared__ half A_tile[M][32];

    for (int sb = 0; sb < 8; sb++) {
        const int il     = sb >> 1;
        const int parity = sb &  1;

        // ── Dequant: each lane fills one column across all 16 rows ──
        //
        // Lane `l` (0..31) handles column `l` of the 16×32 tile, which
        // corresponds to qs byte `32*il + l` in each of the 16 weight
        // blocks. Low nibble → sub-block sb=2*il, high nibble → sb=2*il+1.
        const int col = lane;
        for (int row = 0; row < M; row++) {
            const block_q4_K& blk = W[row];
            const float dall = raw_fp16_to_float(blk.d_raw);
            const float dmin = raw_fp16_to_float(blk.dmin_raw);

            uint8_t sc, mn;
            get_scale_min_k4(sb, blk.scales, sc, mn);

            const float d  = dall * sc;
            const float dm = dmin * mn;

            const uint8_t qb = blk.qs[32 * il + col];
            const float val  = parity
                ? (d * (qb >> 4)  - dm)
                : (d * (qb & 0xF) - dm);

            A_tile[row][col] = __float2half(val);
        }
        __syncwarp();

        // ── 2 MMAs across the K=32 tile, k=16 each ──
        #pragma unroll
        for (int km = 0; km < 2; km++) {
            const int ka = km * 16;

            half2 a0_h = *reinterpret_cast<const half2*>(&A_tile[groupID    ][ka + tinG * 2]);
            half2 a1_h = *reinterpret_cast<const half2*>(&A_tile[groupID + 8][ka + tinG * 2]);
            half2 a2_h = *reinterpret_cast<const half2*>(&A_tile[groupID    ][ka + tinG * 2 + 8]);
            half2 a3_h = *reinterpret_cast<const half2*>(&A_tile[groupID + 8][ka + tinG * 2 + 8]);

            uint32_t a0 = *reinterpret_cast<uint32_t*>(&a0_h);
            uint32_t a1 = *reinterpret_cast<uint32_t*>(&a1_h);
            uint32_t a2 = *reinterpret_cast<uint32_t*>(&a2_h);
            uint32_t a3 = *reinterpret_cast<uint32_t*>(&a3_h);

            const int k_base = sb * 32 + ka;
            half2 b0_h = *reinterpret_cast<const half2*>(&X[groupID * K + k_base + tinG * 2]);
            half2 b1_h = *reinterpret_cast<const half2*>(&X[groupID * K + k_base + tinG * 2 + 8]);

            uint32_t b0 = *reinterpret_cast<uint32_t*>(&b0_h);
            uint32_t b1 = *reinterpret_cast<uint32_t*>(&b1_h);

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
        }
        __syncwarp();
    }

    // ── Store D fragment ──
    D[(groupID    ) * N + (tinG * 2 + 0)] = d0;
    D[(groupID    ) * N + (tinG * 2 + 1)] = d1;
    D[(groupID + 8) * N + (tinG * 2 + 0)] = d2;
    D[(groupID + 8) * N + (tinG * 2 + 1)] = d3;
}

// ── Host reference ──────────────────────────────────────────────────────
//
// Dequantize every weight to FP16, then do an FP32 GEMM. Uses the same
// __float2half conversion as the device path so the FP16 weight tile is
// bit-identical between host and device. Float-add order in the GEMM
// differs from the MMA, but for our problem size and small values that
// produces near-zero error.

static void host_reference(const std::vector<block_q4_K>& W,
                           const std::vector<half>& X,
                           std::vector<float>& D)
{
    std::vector<half> A(M * K);
    for (int row = 0; row < M; row++) {
        const block_q4_K& blk = W[row];
        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        for (int sb = 0; sb < 8; sb++) {
            const int il     = sb >> 1;
            const int parity = sb &  1;
            uint8_t sc, mn;
            get_scale_min_k4(sb, blk.scales, sc, mn);
            const float d  = dall * sc;
            const float dm = dmin * mn;
            for (int col = 0; col < 32; col++) {
                const uint8_t qb = blk.qs[32 * il + col];
                const float val  = parity
                    ? (d * (qb >> 4)  - dm)
                    : (d * (qb & 0xF) - dm);
                A[row * K + sb * 32 + col] = __float2half(val);
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += __half2float(A[i * K + k]) *
                     __half2float(X[j * K + k]);
            }
            D[i * N + j] = s;
        }
    }
}

int main(int, char**) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SM %d.%d\n", prop.name, prop.major, prop.minor);

    // ── Build M Q4_K blocks with random byte content ──
    std::mt19937 rng(42);
    std::uniform_int_distribution<int>  d_byte(0, 255);
    std::uniform_real_distribution<float> d_dall(0.005f, 0.020f);
    std::uniform_real_distribution<float> d_dmin(0.001f, 0.004f);

    std::vector<block_q4_K> h_W(M);
    for (int r = 0; r < M; r++) {
        h_W[r].d_raw    = __half_as_ushort(__float2half(d_dall(rng)));
        h_W[r].dmin_raw = __half_as_ushort(__float2half(d_dmin(rng)));
        for (int i = 0; i < 12;        i++) h_W[r].scales[i] = (uint8_t)d_byte(rng);
        for (int i = 0; i < QK_K / 2;  i++) h_W[r].qs[i]     = (uint8_t)d_byte(rng);
    }

    // Activation X (K × N) col-major, FP16 values in [-1, 1].
    std::uniform_real_distribution<float> d_act(-1.0f, 1.0f);
    std::vector<half> h_X(K * N);
    for (size_t i = 0; i < h_X.size(); i++) h_X[i] = __float2half(d_act(rng));

    std::vector<float> h_D_ref(M * N);
    std::vector<float> h_D    (M * N, 0.0f);
    host_reference(h_W, h_X, h_D_ref);

    block_q4_K* d_W = nullptr;
    half*       d_X = nullptr;
    float*      d_D = nullptr;
    CHECK_CUDA(cudaMalloc(&d_W, M * sizeof(block_q4_K)));
    CHECK_CUDA(cudaMalloc(&d_X, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_D, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), M * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), K * N * sizeof(half),   cudaMemcpyHostToDevice));

    mmq_q4k_tile_kernel<<<1, 32>>>(d_W, d_X, d_D);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Compare ──
    float max_abs = 0.0f, max_rel = 0.0f;
    int   bad     = 0;
    for (int i = 0; i < M * N; i++) {
        const float a = h_D[i];
        const float b = h_D_ref[i];
        const float e = std::fabs(a - b);
        const float r = e / (std::fabs(b) + 1e-6f);
        if (e > max_abs) max_abs = e;
        if (r > max_rel) max_rel = r;
        if (e > 1e-2f) bad++;
    }

    printf("Sample: D[0][0] ref=%g mma=%g | D[15][7] ref=%g mma=%g\n",
           h_D_ref[0], h_D[0],
           h_D_ref[M * N - 1], h_D[M * N - 1]);
    printf("Max abs error: %g | Max rel error: %g | %d / %d outside 1e-2\n",
           max_abs, max_rel, bad, M * N);

    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_D);

    if (bad > 0) {
        printf("FAIL: Q4_K MMQ tile does not match host reference.\n");
        return 1;
    }
    printf("PASS: Q4_K dequant + tensor-core MMQ tile matches reference.\n");
    return 0;
}
