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
    int i = (blockIdx.x << 3) + threadIdx.x;
    int j = (blockIdx.y << 3) + threadIdx.y;
    int k = (blockIdx.z << 3) + threadIdx.z;

    using array_t = float (&) [XDIM][YDIM][ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    bool onBoundary = (i == 0) || (i == XDIM-1) || (j == 0) || (j == YDIM-1) || (k == 0) || (k == ZDIM-1);

    if (!onBoundary)
         Lu[i][j][k] =
            -6 * u[i][j][k]
            + u[i+1][j][k]
            + u[i-1][j][k]
            + u[i][j+1][k]
            + u[i][j-1][k]
            + u[i][j][k+1]
            + u[i][j][k-1];
       
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
