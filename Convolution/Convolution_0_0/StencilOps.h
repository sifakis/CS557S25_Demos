#pragma once

template<int Di, int Do>
void localStencilOp(const float (&in)[10][10][10][Di],
    float (&out)[8][8][8][Do], const float (&stencil)[3][3][3][Di][Do]);
