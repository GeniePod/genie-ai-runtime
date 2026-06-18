// test_mmq_q6k_int8.cu — int8 tensor-core MMQ for the Q6_K prefill GEMM
// (the ffn_down projection — the #1 prefill cost, ~35%, scalar today).
//
// Q6_K has per-16-element scales (scales[16] int8 * super-block d), so this
// uses mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 — one MMA per 16-wide
// scale group — feeding (q6-32) s8 weights x q8_1 activations to the int8
// tensor cores, with the d*scale*d8 scale applied in a float epilogue. Q6_K
// is symmetric (value = d*scale*(q6-32)), so there is NO min/sum term (unlike
// Q4_K). The 6-bit unpack (ql nibble | qh 2-bit, -32) reuses genie's decode
// q6_unpack4 (PR #106), the m16n8k16 A-fragment layout is the "low half" of
// the validated m16n8k32 Q4_K layout.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} }while(0)

constexpr int QK_K = 256;

struct __attribute__((packed)) block_q6_K {
    uint8_t  ql[QK_K / 2];     // 128 — low 4 bits
    uint8_t  qh[QK_K / 4];     // 64  — high 2 bits
    int8_t   scales[QK_K / 16];// 16  — per-16-element scales
    uint16_t d_raw;            // super-block fp16 scale
};
static_assert(sizeof(block_q6_K) == 210, "Q6_K block must be 210 bytes");

struct __attribute__((packed)) block_q8_1 {
    __half d;
    __half s;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 block must be 36 bytes");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign=(h>>15)&1,exp=(h>>10)&0x1F,mant=h&0x3FF; float r;
    if(exp==0) r=std::ldexp((float)mant,-24);
    else if(exp==31) r=mant?NAN:INFINITY;
    else r=std::ldexp((float)(mant+1024),(int)exp-25);
    return sign?-r:r;
#endif
}

// Two-uint16 unaligned int load (block_q6_K is 2- but not 4-aligned).
__host__ __device__ __forceinline__ int load_int_u16(const void* p) {
    const uint16_t* s = (const uint16_t*)p;
    return (int)((uint32_t)s[0] | ((uint32_t)s[1] << 16));
}

#ifndef __CUDA_ARCH__
static int host_vsubss4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; i++) {
        int x = (int8_t)((a >> (8*i)) & 0xFF);
        int y = (int8_t)((b >> (8*i)) & 0xFF);
        int s = x - y; if (s > 127) s = 127; if (s < -128) s = -128;
        r |= (s & 0xFF) << (8*i);
    }
    return r;
}
#endif

// 4 packed (q6-32) int8 for positions p=4*gi..4*gi+3 (mirrors genie decode).
__host__ __device__ __forceinline__ int q6_unpack4(const block_q6_K& blk, int gi) {
    const int p = 4 * gi, n = (p >= 128) ? 128 : 0, ph = p - n;
    const int g = ph >> 5, r = ph & 31;
    const uint8_t* ql_h = blk.ql + (n >> 1);
    const uint8_t* qh_h = blk.qh + (n >> 2);
    const int qlbyte = (ph < 64) ? ph : (ph - 64);
    const int ql_int = load_int_u16(ql_h + qlbyte);
    const int vil = (ph >= 64) ? ((ql_int >> 4) & 0x0F0F0F0F) : (ql_int & 0x0F0F0F0F);
    const int qh_int = load_int_u16(qh_h + r);
    const int shift = 2 * g;
    const int sel = qh_int & (0x03030303 << shift);
    const int vih = (shift <= 4 ? (sel << (4 - shift)) : (sel >> (shift - 4))) & 0x30303030;
#ifdef __CUDA_ARCH__
    return __vsubss4(vil | vih, 0x20202020);
#else
    return host_vsubss4(vil | vih, 0x20202020);
#endif
}

// q8_1 quantize of one 32-elt group.
__global__ void quantize_q8_1(const half* __restrict__ x, block_q8_1* __restrict__ y, int ng) {
    const int gx = blockIdx.x; if (gx >= ng) return;
    const int lane = threadIdx.x;
    const float v = __half2float(x[(int64_t)gx * 32 + lane]);
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    const float d = a / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
    int q = max(-127, min(127, __float2int_rn(v * id)));
    int sum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    y[gx].qs[lane] = (int8_t)q;
    if (lane == 0) { y[gx].d = __float2half(d); y[gx].s = __float2half(d * (float)sum); }
}

constexpr int TILE_M = 16, TILE_N = 8, N_WARPS = 4, BLOCK_N = TILE_N * N_WARPS;

__global__ void gemm_mmq_q6k_i8_kernel(half* __restrict__ y,
                                       const block_q6_K* __restrict__ W,
                                       const block_q8_1* __restrict__ XQ,
                                       int M, int N, int K) {
    const int row_base = blockIdx.y * TILE_M;
    const int blk_tok_base = blockIdx.x * BLOCK_N;
    if (row_base >= M) return;
    const int t_id = threadIdx.x, warp_id = t_id >> 5, lane = t_id & 31;
    const int groupID = lane >> 2, tinG = lane & 3;
    const int tok_base = blk_tok_base + warp_id * TILE_N;
    const int n_blocks = K / QK_K;
    const int nsb = n_blocks * 8;           // q8_1 sub-blocks (32-elt) along K
    const int tok0 = tok_base + tinG * 2, tok1 = tok0 + 1;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    __shared__ int8_t A_tile[TILE_M][16];
    __shared__ float  per_dsc[TILE_M][16];  // d_row * scales_row[j]

    for (int b = 0; b < n_blocks; b++) {
        // per-(row, group) scale = d * scales[j]
        {
            const int row = t_id >> 3;       // 0..15
            const int two = t_id & 7;        // fills 2 of the 16 groups
            const int g_row = row_base + row;
            float d = 0.f; const block_q6_K* bp = nullptr;
            if (g_row < M) { bp = &W[(int64_t)g_row * n_blocks + b]; d = raw_fp16_to_float(bp->d_raw); }
            #pragma unroll
            for (int hpart = 0; hpart < 2; hpart++) {
                const int j = two + 8 * hpart;            // 0..15
                per_dsc[row][j] = (g_row < M) ? d * (float)bp->scales[j] : 0.f;
            }
        }
        __syncthreads();

        for (int j = 0; j < 16; j++) {       // 16 scale groups (16 k each)
            // Fill A_tile[16][16] = (q6-32) for group j. 64 (row,gi) units.
            if (t_id < 64) {
                const int row = t_id >> 2;               // 0..15
                const int gilocal = t_id & 3;            // 0..3
                const int g_row = row_base + row;
                int vi = 0;
                if (g_row < M) vi = q6_unpack4(W[(int64_t)g_row * n_blocks + b], 4 * j + gilocal);
                *reinterpret_cast<int*>(&A_tile[row][4 * gilocal]) = vi;
            }
            __syncthreads();

            // m16n8k16 A fragment: a0=(gid,k 4t..+3), a1=(gid+8,k 4t..+3).
            const int a0 = *reinterpret_cast<const int*>(&A_tile[groupID    ][4 * tinG]);
            const int a1 = *reinterpret_cast<const int*>(&A_tile[groupID + 8][4 * tinG]);

            // B: q8_1 of token (tok_base+groupID). group j -> q8 sub-block
            // b*8 + j/2, byte offset (j&1)*16.
            const int g_tok = tok_base + groupID;
            const int qsb = b * 8 + (j >> 1);
            const int qoff = (j & 1) * 16;
            int bb0 = 0;
            if (g_tok < N)
                bb0 = *reinterpret_cast<const int*>(XQ[(int64_t)g_tok * nsb + qsb].qs + qoff + 4 * tinG);

            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(bb0));

            const float dscA = per_dsc[groupID    ][j];
            const float dscB = per_dsc[groupID + 8][j];
            float d8_0 = 0.f, d8_1 = 0.f;
            if (tok0 < N) d8_0 = __half2float(XQ[(int64_t)tok0 * nsb + qsb].d);
            if (tok1 < N) d8_1 = __half2float(XQ[(int64_t)tok1 * nsb + qsb].d);
            d0 += dscA * d8_0 * (float)c0;
            d1 += dscA * d8_1 * (float)c1;
            d2 += dscB * d8_0 * (float)c2;
            d3 += dscB * d8_1 * (float)c3;
            __syncthreads();
        }
    }

    const int row_a = row_base + groupID, row_b = row_base + groupID + 8;
    if (row_a < M) {
        if (tok0 < N) y[(int64_t)tok0 * M + row_a] = __float2half(d0);
        if (tok1 < N) y[(int64_t)tok1 * M + row_a] = __float2half(d1);
    }
    if (row_b < M) {
        if (tok0 < N) y[(int64_t)tok0 * M + row_b] = __float2half(d2);
        if (tok1 < N) y[(int64_t)tok1 * M + row_b] = __float2half(d3);
    }
}

// ── references ──────────────────────────────────────────────────────────
static void host_ref(const std::vector<block_q6_K>& W, const std::vector<half>& X,
                     std::vector<half>& Y, int M, int N, int K) {
    const int nb = K / QK_K;
    for (int t = 0; t < N; t++) for (int r = 0; r < M; r++) {
        float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q6_K& blk = W[(int64_t)r * nb + b];
            const float d = raw_fp16_to_float(blk.d_raw);
            for (int gi = 0; gi < 64; gi++) {
                const int vi = q6_unpack4(blk, gi);
                for (int m = 0; m < 4; m++) {
                    const int k = 4 * gi + m;
                    const int q = (int8_t)((vi >> (8 * m)) & 0xFF);
                    acc += d * (float)blk.scales[k >> 4] * (float)q
                         * __half2float(X[(int64_t)t * K + b * QK_K + k]);
                }
            }
        }
        Y[(int64_t)t * M + r] = __float2half(acc);
    }
}

static void host_q8_1(const half* x, int8_t* qs, float& d8) {
    float a = 0.f; for (int i = 0; i < 32; i++) a = std::fmax(a, std::fabs(__half2float(x[i])));
    const float d = a / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; i++) { int q = (int)std::lrint(__half2float(x[i]) * id); qs[i] = (int8_t)std::max(-127,std::min(127,q)); }
    d8 = d;
}
static void host_int8_ref(const std::vector<block_q6_K>& W, const std::vector<half>& X,
                          std::vector<half>& Y, int M, int N, int K) {
    const int nb = K / QK_K;
    for (int t = 0; t < N; t++) for (int r = 0; r < M; r++) {
        float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q6_K& blk = W[(int64_t)r * nb + b];
            const float d = raw_fp16_to_float(blk.d_raw);
            for (int j = 0; j < 16; j++) {           // 16-elt scale groups
                int8_t q8[32]; float d8;             // q8 covers 32; group is 16
                const int qsb_k = (16 * j) & ~31;    // 32-block start
                host_q8_1(&X[(int64_t)t * K + b * QK_K + qsb_k], q8, d8);
                const int qoff = (16 * j) & 31;
                int dot = 0;
                for (int m = 0; m < 16; m++) {
                    const int k = 16 * j + m;
                    const int gi = k >> 2, mm = k & 3;
                    const int q6 = (int8_t)((q6_unpack4(blk, gi) >> (8 * mm)) & 0xFF);
                    dot += q6 * (int)q8[qoff + m];
                }
                acc += d * (float)blk.scales[j] * d8 * (float)dot;
            }
        }
        Y[(int64_t)t * M + r] = __float2half(acc);
    }
}

static void fill_q6k(std::vector<block_q6_K>& W, uint32_t seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> sc(-32, 32);
    std::uniform_real_distribution<float> dd(0.001f, 0.01f);
    for (auto& b : W) {
        for (int i = 0; i < 128; i++) b.ql[i] = byte(rng);
        for (int i = 0; i < 64; i++)  b.qh[i] = byte(rng);
        for (int i = 0; i < 16; i++)  b.scales[i] = (int8_t)sc(rng);
        b.d_raw = __half_as_ushort(__float2half(dd(rng)));
    }
}
static void fill_half(std::vector<half>& X, uint32_t seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1, 1);
    for (auto& v : X) v = __float2half(d(rng));
}

static block_q8_1* quantize_X(const std::vector<half>& hX, int N, int K) {
    const int ng = N * K / 32; half* dX; block_q8_1* dXQ;
    CK(cudaMalloc(&dX, hX.size() * sizeof(half)));
    CK(cudaMalloc(&dXQ, (size_t)ng * sizeof(block_q8_1)));
    CK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(half), cudaMemcpyHostToDevice));
    quantize_q8_1<<<ng, 32>>>(dX, dXQ, ng);
    CK(cudaDeviceSynchronize()); cudaFree(dX); return dXQ;
}

static bool correctness() {
    const int M = 128, N = 16, K = 512, nb = K / QK_K;
    printf("\n-- Q6_K int8-MMA correctness: M=%d N=%d K=%d --\n", M, N, K);
    std::vector<block_q6_K> hW(M * nb); std::vector<half> hX(N * K), hY(N * M, __float2half(0.f)), hYr(N * M), hYi(N * M);
    fill_q6k(hW, 0xABCDEF); fill_half(hX, 0x12345);
    host_ref(hW, hX, hYr, M, N, K); host_int8_ref(hW, hX, hYi, M, N, K);
    block_q6_K* dW; half* dY;
    CK(cudaMalloc(&dW, hW.size() * sizeof(block_q6_K))); CK(cudaMalloc(&dY, hY.size() * sizeof(half)));
    CK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(block_q6_K), cudaMemcpyHostToDevice));
    CK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
    block_q8_1* dXQ = quantize_X(hX, N, K);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + TILE_M - 1) / TILE_M, 1);
    gemm_mmq_q6k_i8_kernel<<<grid, N_WARPS * 32>>>(dY, dW, dXQ, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(dW); cudaFree(dY); cudaFree(dXQ);
    float mi = 0, mr = 0; int bad = 0;
    for (int i = 0; i < N * M; i++) {
        float a = __half2float(hY[i]), bi = __half2float(hYi[i]), br = __half2float(hYr[i]);
        mi = std::fmax(mi, std::fabs(a - bi)); mr = std::fmax(mr, std::fabs(a - br));
        if (std::fabs(a - bi) > 0.5f) bad++;
    }
    printf("  Y[0]=%g ref=%g i8ref=%g | Y[last]=%g ref=%g\n",
           __half2float(hY[0]), __half2float(hYr[0]), __half2float(hYi[0]),
           __half2float(hY[N*M-1]), __half2float(hYr[N*M-1]));
    printf("  [MMA vs host_int8] max_abs=%g  %d/%d outside 0.5\n", mi, bad, N*M);
    printf("  [MMA vs host_ref ] max_abs=%g (= q8_1 quant error)\n", mr);
    if (bad) { printf("  FAIL.\n"); return false; }
    printf("  PASS.\n"); return true;
}

static void bench(const char* lbl, int M, int N, int K) {
    const int nb = K / QK_K;
    std::vector<block_q6_K> hW(M * nb); std::vector<half> hX(N * K);
    fill_q6k(hW, 0x999 + M + K); fill_half(hX, 0x111 + N);
    block_q6_K* dW; half* dY;
    CK(cudaMalloc(&dW, hW.size() * sizeof(block_q6_K))); CK(cudaMalloc(&dY, (size_t)N * M * sizeof(half)));
    CK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(block_q6_K), cudaMemcpyHostToDevice));
    block_q8_1* dXQ = quantize_X(hX, N, K);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + TILE_M - 1) / TILE_M, 1);
    for (int i = 0; i < 5; i++) gemm_mmq_q6k_i8_kernel<<<grid, N_WARPS * 32>>>(dY, dW, dXQ, M, N, K);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 100; i++) gemm_mmq_q6k_i8_kernel<<<grid, N_WARPS * 32>>>(dY, dW, dXQ, M, N, K);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= 100;
    printf("  %-10s M=%5d N=%3d K=%5d  ms=%6.3f  GFLOPS=%6.1f\n",
           lbl, M, N, K, ms, 2.0 * M * N * K / (ms * 1e6));
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(dW); cudaFree(dY); cudaFree(dXQ);
}

int main() {
    CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s SM %d.%d\n", p.name, p.major, p.minor);
    bool ok = correctness();
    printf("\n-- throughput (Gemma4 down: M=1536 K=12288) --\n");
    bench("down",    1536, 256, 12288);
    bench("down2",   2048, 256,  9728);
    return ok ? 0 : 1;
}
