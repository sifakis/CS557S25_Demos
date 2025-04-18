#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>

#include "Laplacian.h"
#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include "Utilities.h"

bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

__global__
__launch_bounds__(512)
void computeLaplacianKernel(float *uRaw,float *LuRaw)
{
    // Halo array for input ... indexed [-1 .. 8]^3
    __shared__ float uLocalShifted[10][10][10];
    using padded_array_t = float (&)[10][10][10];
    auto& uLocal = reinterpret_cast<padded_array_t>(uLocalShifted[1][1][1]);

    using array_t = float (&) [XDIM][YDIM][ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    int tid = (threadIdx.x << 6) + (threadIdx.y << 3) + threadIdx.z;

    for (int eID = tid; eID < 1000; eID += 512) {
        int ii = eID / 100;
        int jj = (eID % 100) / 10;
        int kk = eID % 10;
        int bi = (blockIdx.x << 3);
        int bj = (blockIdx.y << 3);
        int bk = (blockIdx.z << 3);
        if (eID < 1000) {
            int i = bi + ii - 1;
            int j = bj + jj - 1;
            int k = bk + kk - 1;
            bool onBoundary = (i <= 0) || (i >= XDIM-1) || (j <= 0) || (j >= YDIM-1) || (k <= 0) || (k >= ZDIM-1);
            uLocal[ii-1][jj-1][kk-1] = onBoundary ? u[i][j][k] : 0.;
        }
    }

    __syncthreads();

    int i = (blockIdx.x << 3) + threadIdx.x;
    int j = (blockIdx.y << 3) + threadIdx.y;
    int k = (blockIdx.z << 3) + threadIdx.z;
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    int kk = threadIdx.z;
    bool onBoundary = (i == 0) || (i == XDIM-1) || (j == 0) || (j == YDIM-1) || (k == 0) || (k == ZDIM-1);

    if (!onBoundary)
         Lu[i][j][k] =
             -6 * uLocal[ii][jj][kk]
             + uLocal[ii+1][jj][kk]
             + uLocal[ii-1][jj][kk]
             + uLocal[ii][jj+1][kk]
             + uLocal[ii][jj-1][kk]
             + uLocal[ii][jj][kk+1]
             + uLocal[ii][jj][kk-1];
       
}

float computeLaplacianGPU( float *uRaw, float *LuRaw)
{
    cudaEvent_t start, stop;

    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start));

    dim3 gridDim(XDIM/8,YDIM/8,ZDIM/8);
    dim3 blockDim(8,8,8);

    check_cuda(cudaGetLastError());
    computeLaplacianKernel<<<gridDim, blockDim>>>(uRaw, LuRaw);
    check_cuda(cudaGetLastError());

    check_cuda(cudaEventRecord(stop));
    check_cuda(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[Elapsed time : %fms]\n", milliseconds);
    return milliseconds;
}
