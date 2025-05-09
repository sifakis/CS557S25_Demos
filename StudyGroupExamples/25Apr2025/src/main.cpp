// Copyright Eftychios Sifakis
// SPDX-License-Identifier: BSD-2-Clause
//
/// @file main.cpp
///
/// @brief Demonstration of WMMA operations

#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>
#include "Timer.h"
#include "Kernels.h"
#include "Utilities.h"
#include <iostream>
#include <iomanip>
#include <random>

int main(int argc, char *argv[])
{
    Timer timer;

    // Allocate and reshape matrix arrays
    using arrayA_t = float (&)[][SIZE_M][SIZE_K];
    using arrayB_t = float (&)[][SIZE_K][SIZE_N];
    using arrayC_t = float (&)[][SIZE_M][SIZE_N];

    auto arrayAraw = new float [SIZE_M*SIZE_K*NUMBER_OF_MATRICES];
    auto arrayBraw = new float [SIZE_N*SIZE_K*NUMBER_OF_MATRICES];
    auto arrayCraw = new float [SIZE_M*SIZE_N*NUMBER_OF_MATRICES];
    auto arrayReferenceCraw = new float [SIZE_M*SIZE_N*NUMBER_OF_MATRICES];

    arrayA_t arrayA = reinterpret_cast<arrayA_t>(*arrayAraw);
    arrayB_t arrayB = reinterpret_cast<arrayB_t>(*arrayBraw);
    arrayC_t arrayC = reinterpret_cast<arrayC_t>(*arrayCraw);
    arrayC_t arrayReferenceC = reinterpret_cast<arrayC_t>(*arrayReferenceCraw);

    if (&arrayA[0][0][0] != arrayAraw) throw std::logic_error("barf");
    // Fill input arrays with random numbers, set output to zero

    timer.Start();
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-1., 1.);
    for (int m = 0; m < NUMBER_OF_MATRICES; m++) {
        for (int i = 0; i < SIZE_M; i++)
        for (int j = 0; j < SIZE_K; j++)
            arrayA[m][i][j] = uniform_dist(gen);
        for (int i = 0; i < SIZE_N; i++)
        for (int j = 0; j < SIZE_K; j++)
            arrayB[m][j][i] = uniform_dist(gen);
        for (int i = 0; i < SIZE_M; i++)
        for (int j = 0; j < SIZE_N; j++)
            arrayC[m][i][j] = arrayReferenceC[m][i][j] = 0.;            
    }
    timer.Stop("Initializing matrix arrays : ");

    // Computing reference result
    timer.Start();
    for (int m = 0; m < NUMBER_OF_MATRICES; m++)
        for (int i = 0; i < SIZE_M; i++)
        for (int j = 0; j < SIZE_N; j++) {
            arrayReferenceC[m][i][j] = 0.;
            for (int k = 0; k < SIZE_K; k++)
                arrayReferenceC[m][i][j] += arrayA[m][i][k] * arrayB[m][k][j]; }
    timer.Stop("Reference computation (CPU) : ");

    // Allocate 
    float *deviceArrayA;
    float *deviceArrayB;
    float *deviceArrayC;

    check_cuda(cudaGetLastError());

    check_cuda(cudaMalloc(&deviceArrayA, NUMBER_OF_MATRICES*SIZE_M*SIZE_K*sizeof(float)));
    check_cuda(cudaMemcpy(deviceArrayA, arrayA, NUMBER_OF_MATRICES*SIZE_M*SIZE_K*sizeof(float), cudaMemcpyHostToDevice));

    if (buffer_check(deviceArrayA, reinterpret_cast<float*>(arrayA), NUMBER_OF_MATRICES*SIZE_M*SIZE_K)) {
        std::cout << "deviceArrayA is correct!\n";
    } else {
        std::cout << "deviceArrayA has mismatch!\n";
    }

    check_cuda(cudaGetLastError());

    check_cuda(cudaMalloc(&deviceArrayB, NUMBER_OF_MATRICES*SIZE_N*SIZE_K*sizeof(float)));
    check_cuda(cudaMemcpy(deviceArrayB, arrayB, NUMBER_OF_MATRICES*SIZE_N*SIZE_K*sizeof(float), cudaMemcpyHostToDevice));
  
    if (buffer_check(deviceArrayB, reinterpret_cast<float*>(arrayB), NUMBER_OF_MATRICES*SIZE_N*SIZE_K)) {
        std::cout << "deviceArrayB is correct!\n";
    } else {
        std::cout << "deviceArrayB has mismatch!\n";
    }

    check_cuda(cudaMalloc(&deviceArrayC, NUMBER_OF_MATRICES*SIZE_M*SIZE_N*sizeof(float)));
    check_cuda(cudaMemset(deviceArrayC, 0, NUMBER_OF_MATRICES*SIZE_M*SIZE_N));

    for(int test = 1; test <= 10; test++) {
        std::cout << "Running GPU test iteration " << std::setw(2) << test << " ";
        computeGEMMtestGPU( reinterpret_cast<arrayA_t>(*deviceArrayA), reinterpret_cast<arrayB_t>(*deviceArrayB),
            reinterpret_cast<arrayC_t>(*deviceArrayC), NUMBER_OF_MATRICES);
    }
    
    check_cuda(cudaMemcpy(arrayC, deviceArrayC, NUMBER_OF_MATRICES*SIZE_M*SIZE_N*sizeof(float), cudaMemcpyDeviceToHost));
    if (buffer_check(deviceArrayC, reinterpret_cast<float*>(arrayC), NUMBER_OF_MATRICES*SIZE_M*SIZE_N)) {
        std::cout << "deviceArrayC is correct!\n";
    } else {
        std::cout << "deviceArrayC has mismatch!\n";
    }

    // Compute discrepancy

    float maxDiff = 0.;
    timer.Start();
    for (int m = 0; m < NUMBER_OF_MATRICES; m++)
        for (int i = 0; i < SIZE_M; i++)
        for (int j = 0; j < SIZE_N; j++)
            maxDiff = std::max( maxDiff, std::abs(arrayReferenceC[m][i][j] - arrayC[m][i][j]) );
    timer.Stop("Validation : ");
    std::cout << "Discrepancy = " << maxDiff << std::endl;

    return 0;
}
