// Copyright Eftychios Sifakis
// SPDX-License-Identifier: BSD-2-Clause
//
/// @file Utilities.h
///
/// @brief Simple CUDA check tools

#pragma once

#define check_cuda(code) (check_cuda_result(code, __FILE__, __LINE__))

bool check_cuda_result(cudaError_t code, const char* file, int line);

template<typename T>
bool buffer_check(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    check_cuda(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));

    bool same = true;
    for (int i=0; same && i< elem_count; ++i) {
        same = (tmpBuffer[i] == hostBuffer[i]);
    }

    delete [] tmpBuffer;

    return same;
}
