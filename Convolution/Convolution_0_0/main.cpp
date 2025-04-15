#include "Parameters.h"
#include "Timer.h"
#include "StencilReferenceBenchmark.h"

#include <iomanip>
#include <random>

template<typename BenchmarkType>
float MaxDiff(const typename BenchmarkType::out_tensor_array_t result,
              const typename BenchmarkType::out_tensor_array_t resultReference)
{
    float maxDiff = 0.;
    for (int i = 0; i < BenchmarkType::Nx; i++)
    for (int j = 0; j < BenchmarkType::Ny; j++)
    for (int k = 0; k < BenchmarkType::Nz; k++)
        for (int outDim = 0; outDim < BenchmarkType::Do; outDim++)
            maxDiff = std::max(maxDiff, std::abs(result[i][j][k][outDim]-resultReference[i][j][k][outDim]));
    return maxDiff;
}

int main(int argc, char *argv[])
{
    using BenchmarkType = StencilReferenceBenchmark<Params::Nx, Params::Ny, Params::Nz, Params::Di, Params::Do>;
    using namespace Params;

    float *stencilRaw = new float[3*3*3*Params::Di*Params::Do];
    BenchmarkType benchmark(stencilRaw);
    Timer timer;
    
    timer.Start();
    for (int i = 0; i < Nx*Ny*Nz*Do; i++)
        benchmark.outRaw[i] = benchmark.outReferenceRaw[i] = 0.;
    for (int i = 0; i < Nx*Ny*Nz*Di+2*BenchmarkType::offset; i++)
        benchmark.inRaw[i] = 0.;
    for (int i = 0; i < 27*Di*Do; i++)
        stencilRaw[i] = 0.;
    timer.Stop("Zeroing out buffers : ");

    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-1., 1.);

    timer.Start();
    for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++)
    for (int dk = -1; dk <= 1; dk++)
        for (int inDim = 0; inDim < Di; inDim++)
        for (int outDim = 0; outDim < Do; outDim++)
            benchmark.stencil[di][dj][dk][inDim][outDim] = uniform_dist(gen);
    for (int i = 1; i < Nx-1; i++)
    for (int j = 1; j < Ny-1; j++)
    for (int k = 1; k < Nz-1; k++)
        for (int inDim = 0; inDim < Di; inDim++)
            benchmark.in[i][j][k][inDim] = uniform_dist(gen);
    
    timer.Stop("Initializing values : ");
    
    std::cout << "Max diff = " << MaxDiff<BenchmarkType>(benchmark.out, benchmark.outReference) << std::endl;

    std::cout << "Running reference " << std::flush;
    timer.Start();
    benchmark.runReference();
    timer.Stop("Elapsed time : ");
    
    std::cout << "Running test " << std::flush;
    timer.Start();
    benchmark.runPerformance();
    timer.Stop("Elapsed time : ");
    
    std::cout << "Max diff = " << MaxDiff<BenchmarkType>(benchmark.out, benchmark.outReference) << std::endl;

    std::cout << "Running performance test " << std::flush << std::endl;
    for (int run = 0; run < 10; run++) {    
        timer.Start();
        benchmark.runPerformance();
        timer.Stop("Elapsed time : ");
    }

    return 0;
}
