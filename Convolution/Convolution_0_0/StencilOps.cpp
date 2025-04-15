#include "StencilOps.h"

template<int Di, int Do>
void localStencilOp(const float (&in)[10][10][10][Di],
    float (&out)[8][8][8][Do], const float (&stencil)[3][3][3][Di][Do])
{
    for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++)
    for (int dk = -1; dk <= 1; dk++)
        for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++)
            for (int inDim = 0; inDim < Di; inDim++)
                for (int outDim = 0; outDim < Do; outDim++)
                out[i][j][k][outDim] +=
                    stencil[di][dj][dk][inDim][outDim] *
                    in[i+di][j+dj][k+dk][inDim];
}

template
void localStencilOp<16,32>(const float (&in)[10][10][10][16],
    float (&out)[8][8][8][32], const float (&stencil)[3][3][3][16][32]);
