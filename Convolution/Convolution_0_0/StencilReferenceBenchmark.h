#pragma once

#include "StencilOps.h"

template<int _Nx, int _Ny, int _Nz, int _Di, int _Do>
struct StencilReferenceBenchmark
{
    static constexpr int Nx = _Nx;
    static constexpr int Ny = _Ny;
    static constexpr int Nz = _Nz;
    static constexpr int Di = _Di;
    static constexpr int Do = _Do;
    
    using in_tensor_array_t      = float (&) [Nx][Ny][Nz][Di];
    using out_tensor_array_t     = float (&) [Nx][Ny][Nz][Do];
    using stencil_array_t        = float (&) [3][3][3][Di][Do];

    static constexpr std::size_t offset = Di + Nz*Di + Ny*Nz*Di;
    static constexpr std::size_t stencilOffset = Di*Do + 3*Di*Do + 3*3*Di*Do;

    StencilReferenceBenchmark(float * const stencilRaw)
        : outRaw(new float [Nx*Ny*Nz*Do]),
         outReferenceRaw(new float [Nx*Ny*Nz*Do]),
         inRaw(new float [Nx*Ny*Nz*Di + 2*offset]),
         out(reinterpret_cast<out_tensor_array_t>(*outRaw)),
         outReference(reinterpret_cast<out_tensor_array_t>(*outReferenceRaw)),
         in(reinterpret_cast<in_tensor_array_t>(*(inRaw+offset))),
         stencil(reinterpret_cast<stencil_array_t>(*(stencilRaw+stencilOffset)))
    {}

    void runReference()
    {
#pragma omp parallel for
        for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
        for (int k = 0; k < Nz; k++)
            for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++)
                for (int outDim = 0; outDim < Do; outDim++)
                for (int inDim = 0; inDim < Di; inDim++)
                    outReference[i][j][k][outDim] +=
                        stencil[di][dj][dk][inDim][outDim] *
                        in[i+di][j+dj][k+dk][inDim];
    }

    void runPerformance()
    {
#pragma omp parallel for
        for (int i = 0; i < Nx; i += 8)
        for (int j = 0; j < Ny; j += 8)
        for (int k = 0; k < Nz; k += 8) {

            float inLocal[10][10][10][Di];

            for (int di = -1; di <= 8; di++)
            for (int dj = -1; dj <= 8; dj++)
            for (int dk = -1; dk <= 8; dk++)
                for (int inDim = 0; inDim < Di; inDim++)
                    inLocal[di+1][dj+1][dk+1][inDim] = in[i+di][j+dj][k+dk][inDim];
            
            using inLocal_tensor_array_t  = float (&) [10][10][10][Di];
            inLocal_tensor_array_t inLocalShifted = reinterpret_cast<inLocal_tensor_array_t>(inLocal[1][1][1][0]);
            float outLocal[8][8][8][Do] = { 0 };

            localStencilOp<Di,Do>(inLocalShifted, outLocal, stencil);

            for (int di = 0; di < 8; di++)
            for (int dj = 0; dj < 8; dj++)
            for (int dk = 0; dk < 8; dk++)
                for (int outDim = 0; outDim < Do; outDim++)
                    out[i+di][j+dj][k+dk][outDim] += outLocal[di][dj][dk][outDim];
        }         
    }

    float *outRaw, *outReferenceRaw, *inRaw;
    out_tensor_array_t out, outReference;
    in_tensor_array_t in;
    stencil_array_t stencil;

};
