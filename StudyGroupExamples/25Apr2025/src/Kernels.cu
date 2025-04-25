#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>
#include <mma.h>

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include "Utilities.h"

#include "Kernels.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace nvcuda;
using namespace cute;

bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

__global__
void GEMMKernel_v5(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;
    int threadID = threadIdx.x;

    using _M = _128;
    using _N = _128;
    using _K = _64;
    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2, _2, _1>>>;
    
    Tensor gA = make_tensor(make_gmem_ptr(&matrixArrayA[matrixID][0][0]),
        make_layout(make_shape(_M{}, _K{}), GenRowMajor{}));
    Tensor gB = make_tensor(make_gmem_ptr(&matrixArrayB[matrixID][0][0]),
        make_layout(make_shape(_N{}, _K{}), GenColMajor{}));
    Tensor gC = make_tensor(make_gmem_ptr(&matrixArrayC[matrixID][0][0]),
        make_layout(make_shape(_M{}, _N{}), GenRowMajor{}));

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, Shape<_M, _N>{});

    auto thr_mma = tiled_mma.get_thread_slice(threadID);

    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCrA = thr_mma.partition_fragment_A(gA);
    Tensor tCrB = thr_mma.partition_fragment_B(gB);

    copy(tCgA, tCrA);
    copy(tCgB, tCrB);

    __syncthreads();

    gemm(tiled_mma, accum, tCrA, tCrB, accum);

    Tensor tCgC = thr_mma.partition_C(gC);
    copy(accum, tCgC);
}

__global__
void GEMMKernel_v4(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;
    int threadID = threadIdx.x;
    int warpID = threadID >> 5;
    int warpX = warpID & 0x1;
    int warpY = warpID >> 1;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag00;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag01;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag10;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag11;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag00;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag01;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag10;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag11;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag00;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag01;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag10;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag11;

    for (int iOff = warpX*32; iOff < SIZE_M; iOff += 64)
    for (int jOff = warpY*32; jOff < SIZE_N; jOff += 64) {

        // Initialize the output to zero
        wmma::fill_fragment(c_frag00, 0.0f);
        wmma::fill_fragment(c_frag01, 0.0f);
        wmma::fill_fragment(c_frag10, 0.0f);
        wmma::fill_fragment(c_frag11, 0.0f);

        for (int kOff = 0; kOff < SIZE_K; kOff += 16) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag00, &matrixA[iOff   ][kOff  ], SIZE_K);
            wmma::load_matrix_sync(a_frag01, &matrixA[iOff   ][kOff+8], SIZE_K);
            wmma::load_matrix_sync(a_frag10, &matrixA[iOff+16][kOff  ], SIZE_K);
            wmma::load_matrix_sync(a_frag11, &matrixA[iOff+16][kOff+8], SIZE_K);
            wmma::load_matrix_sync(b_frag00, &matrixB[kOff  ][jOff   ], SIZE_N);
            wmma::load_matrix_sync(b_frag01, &matrixB[kOff  ][jOff+16], SIZE_N);
            wmma::load_matrix_sync(b_frag10, &matrixB[kOff+8][jOff   ], SIZE_N);
            wmma::load_matrix_sync(b_frag11, &matrixB[kOff+8][jOff+16], SIZE_N);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag00, a_frag00, b_frag00, c_frag00);
            wmma::mma_sync(c_frag00, a_frag01, b_frag10, c_frag00);
            wmma::mma_sync(c_frag01, a_frag00, b_frag01, c_frag01);
            wmma::mma_sync(c_frag01, a_frag01, b_frag11, c_frag01);
            wmma::mma_sync(c_frag10, a_frag10, b_frag00, c_frag10);
            wmma::mma_sync(c_frag10, a_frag11, b_frag10, c_frag10);
            wmma::mma_sync(c_frag11, a_frag10, b_frag01, c_frag11);
            wmma::mma_sync(c_frag11, a_frag11, b_frag11, c_frag11);
        }

        // Store the output
        wmma::store_matrix_sync(&matrixC[iOff   ][jOff   ], c_frag00, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff   ][jOff+16], c_frag01, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff+16][jOff   ], c_frag10, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff+16][jOff+16], c_frag11, SIZE_N, wmma::mem_row_major);
    }

}

__global__
void GEMMKernel_v3(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag00;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag01;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag10;
    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag11;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag00;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag01;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag10;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag11;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag00;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag01;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag10;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag11;

    for (int iOff = 0; iOff < SIZE_M; iOff += 32)
    for (int jOff = 0; jOff < SIZE_N; jOff += 32) {

        // Initialize the output to zero
        wmma::fill_fragment(c_frag00, 0.0f);
        wmma::fill_fragment(c_frag01, 0.0f);
        wmma::fill_fragment(c_frag10, 0.0f);
        wmma::fill_fragment(c_frag11, 0.0f);

        for (int kOff = 0; kOff < SIZE_K; kOff += 16) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag00, &matrixA[iOff   ][kOff  ], SIZE_K);
            wmma::load_matrix_sync(a_frag01, &matrixA[iOff   ][kOff+8], SIZE_K);
            wmma::load_matrix_sync(a_frag10, &matrixA[iOff+16][kOff  ], SIZE_K);
            wmma::load_matrix_sync(a_frag11, &matrixA[iOff+16][kOff+8], SIZE_K);
            wmma::load_matrix_sync(b_frag00, &matrixB[kOff  ][jOff   ], SIZE_N);
            wmma::load_matrix_sync(b_frag01, &matrixB[kOff  ][jOff+16], SIZE_N);
            wmma::load_matrix_sync(b_frag10, &matrixB[kOff+8][jOff   ], SIZE_N);
            wmma::load_matrix_sync(b_frag11, &matrixB[kOff+8][jOff+16], SIZE_N);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag00, a_frag00, b_frag00, c_frag00);
            wmma::mma_sync(c_frag00, a_frag01, b_frag10, c_frag00);
            wmma::mma_sync(c_frag01, a_frag00, b_frag01, c_frag01);
            wmma::mma_sync(c_frag01, a_frag01, b_frag11, c_frag01);
            wmma::mma_sync(c_frag10, a_frag10, b_frag00, c_frag10);
            wmma::mma_sync(c_frag10, a_frag11, b_frag10, c_frag10);
            wmma::mma_sync(c_frag11, a_frag10, b_frag01, c_frag11);
            wmma::mma_sync(c_frag11, a_frag11, b_frag11, c_frag11);
        }

        // Store the output
        wmma::store_matrix_sync(&matrixC[iOff   ][jOff   ], c_frag00, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff   ][jOff+16], c_frag01, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff+16][jOff   ], c_frag10, SIZE_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&matrixC[iOff+16][jOff+16], c_frag11, SIZE_N, wmma::mem_row_major);
    }

}

__global__
void GEMMKernel_v2(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;
    int threadID = threadIdx.x;
    int warpID = threadID >> 5;
    int warpX = warpID & 0x1;
    int warpY = warpID >> 1;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    for (int iOff = warpX*16; iOff < SIZE_M; iOff += 32)
    for (int jOff = warpY*16; jOff < SIZE_N; jOff += 32) {

        // Initialize the output to zero
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kOff = 0; kOff < SIZE_K; kOff += 8) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, &matrixA[iOff][kOff], SIZE_K);
            wmma::load_matrix_sync(b_frag, &matrixB[kOff][jOff], SIZE_N);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store the output
        wmma::store_matrix_sync(&matrixC[iOff][jOff], c_frag, SIZE_N, wmma::mem_row_major);

    }

}

__global__
void GEMMKernel_v1(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    wmma::fragment<wmma::matrix_a,    16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    for (int iOff = 0; iOff < SIZE_M; iOff += 16)
    for (int jOff = 0; jOff < SIZE_N; jOff += 16) {

        // Initialize the output to zero
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kOff = 0; kOff < SIZE_K; kOff += 8) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, &matrixA[iOff][kOff], SIZE_K);
            wmma::load_matrix_sync(b_frag, &matrixB[kOff][jOff], SIZE_N);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store the output
        wmma::store_matrix_sync(&matrixC[iOff][jOff], c_frag, SIZE_N, wmma::mem_row_major);

    }

}

__global__
void GEMMKernelSerial(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N], float (&matrixArrayC)[][SIZE_M][SIZE_N])
{
    int matrixID = blockIdx.x;

    auto& matrixA = matrixArrayA[matrixID];
    auto& matrixB = matrixArrayB[matrixID];
    auto& matrixC = matrixArrayC[matrixID];

    for (int i = 0; i < SIZE_M; i++)
    for (int j = 0; j < SIZE_N; j++) {
        matrixC[i][j] = 0.;
        for (int k = 0; k < SIZE_K; k++)
            matrixC[i][j] += matrixA[i][k] * matrixB[k][j]; }
    
}
float computeGEMMtestGPU(const float (&matrixArrayA)[][SIZE_M][SIZE_K], const float (&matrixArrayB)[][SIZE_K][SIZE_N],
                         float (&matrixArrayC)[][SIZE_M][SIZE_N], int numMatrices)
{
     cudaEvent_t start, stop;

     check_cuda(cudaEventCreate(&start));
     check_cuda(cudaEventCreate(&stop));
     check_cuda(cudaEventRecord(start));

     check_cuda(cudaGetLastError());
     // GEMMKernelSerial<<<numMatrices, 1>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     // GEMMKernel_v1<<<numMatrices,32>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     // GEMMKernel_v2<<<numMatrices,128>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     // GEMMKernel_v3<<<numMatrices,32>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     // GEMMKernel_v4<<<numMatrices,128>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     GEMMKernel_v5<<<numMatrices,128>>>(matrixArrayA,matrixArrayB,matrixArrayC);
     check_cuda(cudaGetLastError());

     check_cuda(cudaEventRecord(stop));
     check_cuda(cudaEventSynchronize(stop));

     float milliseconds = 0;
     cudaEventElapsedTime(&milliseconds, start, stop);
     printf("[Elapsed time : %fms]\n", milliseconds);
     return milliseconds;
 }
