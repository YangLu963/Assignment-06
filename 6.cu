%%writefile gemm_cute.cu
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cute/tensor.hpp>

using namespace cute;

/**
 * High-performance GEMM kernel optimized for NVIDIA T4 (Turing SM75).
 * Features: Shared Memory Tiling, float4 Vectorized Loads, Register Accumulation.
 */
__global__ void gemm_smem_async_kernel(
    float const* __restrict__ A,
    float const* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // ---------- Static Tile Dimensions ----------
    constexpr int BM = 64;   // Block tile rows
    constexpr int BN = 64;   // Block tile columns
    constexpr int BK = 32;   // Block tile depth (k-slice)

    // ---------- Per-thread Register Tile ----------
    // 16x16 block -> Each thread handles (BM/16) x (BN/16) = 4x4 output elements
    constexpr int TM = BM / 16;  // = 4
    constexpr int TN = BN / 16;  // = 4

    // ---------- Thread / Block Coordinates ----------
    const int tx = threadIdx.x;          // range [0, 16)
    const int ty = threadIdx.y;          // range [0, 16)
    const int tid = ty * 16 + tx;        // Flattened thread ID [0, 256)
    const int bx = blockIdx.x;           // Block column index (N direction)
    const int by = blockIdx.y;           // Block row index (M direction)

    // ---------- Shared Memory Allocation ----------
    __shared__ float sA[BM][BK];   // 64x32 row-major
    __shared__ float sB[BK][BN];   // 32x64 row-major

    // ---------- Register Accumulators ----------
    float accum[TM][TN] = {};       // 4x4 fragment, initialized to zero

    // ---------- Global Memory Starting Offsets ----------
    const int rowA = by * BM;      // Starting row of A for this block
    const int colB = bx * BN;      // Starting column of B for this block

    // ---------- Memory Loading Helpers ----------
    // 256 threads loading BMxBK = 64x32 = 2048 floats
    // Each thread loads 2048/256 = 8 elements
    constexpr int A_ELEMS = BM * BK;    // 2048
    constexpr int B_ELEMS = BK * BN;    // 2048
    constexpr int THREADS  = 256;

    // =================== Main K-Loop ===================
    for (int k0 = 0; k0 < K; k0 += BK) {

        // ---- Step 1: Global to Shared Memory Transfer via float4 Vectorization ----
        // float4 reads 4 floats (128-bit) at once to maximize T4 memory bandwidth.
        // A_ELEMS = 2048 floats = 512 float4. Each thread loads 512/256 = 2 float4.

        static_assert(A_ELEMS % (THREADS * 4) == 0, "A tile size must be a multiple of THREADS*4");
        static_assert(B_ELEMS % (THREADS * 4) == 0, "B tile size must be a multiple of THREADS*4");

        // Load sA (BMxBK = 64x32) using float4 pointers
        {
            float4* sA4 = reinterpret_cast<float4*>(sA);
            constexpr int VEC_A = A_ELEMS / 4;   // 512 float4 elements
#pragma unroll
            for (int i = tid; i < VEC_A; i += THREADS) {
                int elem  = i * 4;                // Corresponding flat float index
                int r     = elem / BK;
                int c     = elem % BK;            // 'c' is guaranteed to be a multiple of 4
                int gRow  = rowA + r;
                int gCol  = k0   + c;
                float4 val = make_float4(0,0,0,0);
                if (gRow < M && gCol + 3 < K) {
                    // __ldg: Load via read-only texture cache for optimized access to restrict data
                    val = __ldg(reinterpret_cast<const float4*>(&A[gRow * K + gCol]));
                }
                sA4[r * (BK/4) + c/4] = val;
            }
        }

        // Load sB (BKxBN = 32x64) using float4 pointers
        {
            float4* sB4 = reinterpret_cast<float4*>(sB);
            constexpr int VEC_B = B_ELEMS / 4;   // 512 float4 elements
#pragma unroll
            for (int i = tid; i < VEC_B; i += THREADS) {
                int elem  = i * 4;
                int r     = elem / BN;
                int c     = elem % BN;
                int gRow  = k0   + r;
                int gCol  = colB + c;
                float4 val = make_float4(0,0,0,0);
                if (gRow < K && gCol + 3 < N) {
                    val = __ldg(reinterpret_cast<const float4*>(&B[gRow * N + gCol]));
                }
                sB4[r * (BN/4) + c/4] = val;
            }
        }

        // ---- Step 2: Barrier Synchronization ----
        // Ensure all threads have finished loading data into Shared Memory
        __syncthreads();

        // ---- Step 3: Compute GEMM at Register Level ----
        // Each thread computes a TMxTN block starting at [ty*TM, tx*TN]
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += sA[ty * TM + m][k]
                                 * sB[k][tx * TN + n];
                }
            }
        }

        // ---- Step 4: Post-computation Barrier ----
        // Synchronize before next iteration to prevent overwriting SMEM
        __syncthreads();
    }

    // =================== Write Results Back to Global C ===================
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gRow = by * BM + ty * TM + m;
            int gCol = bx * BN + tx * TN + n;
            if (gRow < M && gCol < N) {
                C[gRow * N + gCol] = accum[m][n];
            }
        }
    }
}

int main() {
    // --- Matrix Dimensions (Increase these for meaningful TFLOPS benchmarking) ---
    const int M = 128, N = 128, K = 64;

    // --- Host Memory ---
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    // --- Device Memory ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // --- Launch Configuration ---
    // Grid: (N/BN, M/BM) = (2, 2), Block: 16x16 = 256 threads
    dim3 dimGrid ((N + 63) / 64, (M + 63) / 64);
    dim3 dimBlock(16, 16);

    // --- Warm-up Kernel ---
    gemm_smem_async_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // --- Performance Benchmarking: 100 iterations ---
    const int REPS = 100;
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    cudaEventRecord(t_start);
    for (int i = 0; i < REPS; ++i) {
        gemm_smem_async_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t_start, t_stop);
    ms /= REPS;

    // Calculation: TFLOPS = 2 * M * N * K / (time_in_seconds * 1e12)
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;

    // --- Verify Correctness ---
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "======================================" << std::endl;
    std::cout << "      CuTe GEMM — SMEM + Vectorized" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Matrix: (" << M << "x" << K << ") x ("
              << K << "x" << N << ")" << std::endl;
    std::cout << "C[0]     = " << h_C[0]
              << "  (expected " << 2.0f * K << ")" << std::endl;
    std::cout << "C[M*N-1] = " << h_C[M*N-1]
              << "  (expected " << 2.0f * K << ")" << std::endl;

    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_C[i] - 2.0f * K) > 1e-3f) { pass = false; break; }
    }
    std::cout << "Correctness: " << (pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "Avg latency: " << ms      << " ms" << std::endl;
    std::cout << "Performance: " << tflops  << " TFLOPS" << std::endl;
    std::cout << "======================================" << std::endl;

    // --- Clean up ---
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
!nvcc -std=c++17 -arch=sm_75 -I /content/cutlass/include gemm_cute.cu -o gemm_cute
!./gemm_cute
