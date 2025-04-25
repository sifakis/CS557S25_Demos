#pragma once

#define SIZE_M 128
#define SIZE_N 128
#define SIZE_K 64

#define NUMBER_OF_MATRICES 16*1024

float computeGEMMtestGPU(const float (&matrixA)[][SIZE_M][SIZE_K], const float (&matrixB)[][SIZE_K][SIZE_N],
    float (&matrixC)[][SIZE_M][SIZE_N], int numMatrices);
