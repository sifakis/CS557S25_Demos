// Copyright Eftychios Sifakis
// SPDX-License-Identifier: BSD-2-Clause
//
/// @file main.cpp
///
/// @brief Simple test Sparse Convolutional Operations

#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>
#include "Timer.h"
#include "Laplacian.h"
#include "Kernels.h"
#include "Utilities.h"
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{
    using array_t = float (&) [XDIM][YDIM][ZDIM];
    
    float *uRaw = new float [XDIM*YDIM*ZDIM];
    float *LuRaw = new float [XDIM*YDIM*ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++) {
        u[i][j][k] = (float) ((i+j*k)%991);
        Lu[i][j][k] = 0.;
    }

    Timer timer;

    for(int test = 1; test <= 10; test++)
    {
        std::cout << "Running CPU test iteration " << std::setw(2) << test << " ";
        timer.Start();
        ComputeLaplacian(u, Lu);
        timer.Stop("Elapsed time : ");
    }
    
    // Allocate 
    float *deviceURaw;
    float *deviceLuRaw;

    check_cuda(cudaGetLastError());

    check_cuda(cudaMalloc(&deviceURaw, XDIM*YDIM*ZDIM*sizeof(float)));
    check_cuda(cudaMemcpy(deviceURaw, uRaw, XDIM*YDIM*ZDIM*sizeof(float), cudaMemcpyHostToDevice));

    if (buffer_check(deviceURaw, uRaw, XDIM*YDIM*ZDIM)) {
        std::cout << "deviceURaw is correct!\n";
    } else {
        std::cout << "deviceURaw has mismatch!\n";
    }

    check_cuda(cudaMalloc(&deviceLuRaw, XDIM*YDIM*ZDIM*sizeof(float)));
    check_cuda(cudaMemset(deviceLuRaw, 0, XDIM*YDIM*ZDIM*sizeof(float)));

    for(int test = 1; test <= 0; test++)
    {
        std::cout << "Running GPU test iteration " << std::setw(2) << test << " ";
        computeLaplacianGPU(deviceURaw, deviceLuRaw);
    }
    
    if (buffer_check(deviceLuRaw, LuRaw, XDIM*YDIM*ZDIM)) {
        std::cout << "deviceLuRaw is correct!\n";
    } else {
        std::cout << "deviceLuRaw has mismatch!\n";
    }

    return 0;
}
