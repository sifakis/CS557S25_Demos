#pragma once

#define NUMBER_OF_MATRICES 1024*1024

float computeGEMMtestGPU(const float (&matrixA)[][16][8], const float (&matrixB)[][8][16],
    float (&matrixC)[][16][16], int numMatrices);
