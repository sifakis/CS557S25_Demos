#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>
#include <mma.h>

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include "Utilities.h"

using namespace nvcuda;

bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

__global__
void GEMMKernel(const float (&matrixArrayA)[][16][8], const float (&matrixArrayB)[][8][16], float (&matrixArrayC)[][16][16])
{
    int matrixID = blockIdx.x;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, &matrixA[0][0], 8);
   wmma::load_matrix_sync(b_frag, &matrixB[0][0], 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(&matrixC[0][0], c_frag, 16, wmma::mem_row_major);

    
}
__global__
void GEMMKernelSerial(const float (&matrixArrayA)[][16][8], const float (&matrixArrayB)[][8][16], float (&matrixArrayC)[][16][16])
{
    int matrixID = blockIdx.x;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++) {
        matrixC[i][j] = 0.;
        for (int k = 0; k < 8; k++)
            matrixC[i][j] += matrixA[i][k] * matrixB[k][j]; }
    
}
float computeGEMMtestGPU(const float (&matrixArrayA)[][16][8], const float (&matrixArrayB)[][8][16],
                         float (&matrixArrayC)[][16][16], int numMatrices)
{
     cudaEvent_t start, stop;

     check_cuda(cudaEventCreate(&start));
     check_cuda(cudaEventCreate(&stop));
     check_cuda(cudaEventRecord(start));

     check_cuda(cudaGetLastError());
     // GEMMKernelSerial<<<numMatrices, 1>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     GEMMKernel<<<numMatrices,32>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     check_cuda(cudaGetLastError());

     check_cuda(cudaEventRecord(stop));
     check_cuda(cudaEventSynchronize(stop));

     float milliseconds = 0;
     cudaEventElapsedTime(&milliseconds, start, stop);
     printf("[Elapsed time : %fms]\n", milliseconds);
     return milliseconds;
 }
