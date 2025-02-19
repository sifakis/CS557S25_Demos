#pragma once

#if 0
float convolution(unsigned int leafCount,
                  nanovdb::NanoGrid<nanovdb::ValueIndex> *grid,
                  float *outTensor,
                  const float *inTensor,
                  const float *stencil_ptr);

float stencilConvolveLauncher(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *hostGrid, nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGrid,
    float *inputBuffer,
    float *stencil,
    float *outputBuffer);
#endif
float computeLaplacianGPU( float *uRaw, float *LuRaw);
