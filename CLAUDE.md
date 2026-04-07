# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS557 Spring 2025 course demos for high-performance computing. Each subdirectory is a self-contained example showing progressive optimization of numerical kernels: matrix multiplication (DenseAlgebra), finite-difference Laplacian solvers (LaplaceSolver, LaplacianStencil), convolution, sparse operations, and GPU/heterogeneous computing (StudyGroupExamples).

## Build Commands

Each example is a standalone directory. Navigate into it and run `make`.

**CPU examples (Intel compiler + MKL):**
```bash
cd DenseAlgebra/GEMM_Test_1_0_avx2
make          # build
make assembly # generate .s assembly output
make clean    # remove a.out, *.s, *.o
./a.out
```

**CPU examples (icc/g++ without explicit Makefile):**
```bash
# Typical flags pattern (see neighboring Makefiles for exact flags):
icc -O3 -std=c++14 -Wall -w -mavx2 -mfma -fopenmp main.cpp *.cpp
```

**CUDA examples** (require `CUTLASS_ROOT` env var for 25Apr2025 CUTE example):
```bash
cd StudyGroupExamples/25Apr2025
export CUTLASS_ROOT=/path/to/cutlass
make                       # builds into release/
./release/cudaTest
make clean
```

## Architecture & Conventions

### Directory numbering scheme
Within each domain, subdirectories are numbered `<domain>_<major>_<minor>`:
- `0.x` = baseline/reference implementations (plain C++, no SIMD, single-threaded or basic OpenMP)
- `1.x` = optimized variants (blocking, tiling, SIMD intrinsics, hand-written assembly)

### Common file roles (repeated in every example)
| File | Role |
|------|------|
| `Parameters.h` | `#define` constants for problem size, tile size, etc. — change here to reconfigure |
| `Timer.h` | RAII wall-clock timer for benchmarking |
| `Utilities.cpp/h` | Memory allocation with alignment, printing helpers |
| `main.cpp` | Driver: allocates data, calls kernels, times and validates them |

### CPU kernel patterns
- Kernels are split into a **reference** function and an **optimized** function within the same `.cpp`/`.h` pair.
- Vectorization uses AVX2 (`-mavx2 -mfma`) or AVX512 (`-mavx512f`); `icc`-specific flags `-xavx2`/`-xavx512` enable auto-vectorization with the respective ISA.
- `-fno-alias` is applied only to hot kernel files (e.g. `MatMatMultiplyBlockHelper.cpp`, `StencilOps.cpp`) to allow aggressive alias-free optimization.
- Assembly inspection: `make assembly` compiles with `-S`; output is the `.s` file for that translation unit.

### CUDA / GPU examples (StudyGroupExamples/)
- Source lives under `src/`; compiled objects go into `release/`.
- `Kernels.cu` contains GPU kernels; `Laplacian.cpp` / `main.cpp` contain CPU-side orchestration.
- Target architecture: `sm_86` (NVIDIA Ampere, RTX 30-series).
- The 25Apr2025 example uses CUTE (CUTLASS Tensor Universal template Engine) and requires `CUTLASS_ROOT` set to a CUTLASS install.
- Compilation flags of note: `--expt-relaxed-constexpr`, `-lineinfo`, `--resource-usage -Xptxas -v` (prints register/shared-memory usage).

### SparseDirect (PARDISO)
Uses Intel MKL PARDISO for sparse direct solve. Matrix stored in CSR format (`CSRMatrix.h`). Requires MKL to be available on the link path (linked via icc `-mkl`).

### LaplacianNanoVDB
Uses the NanoVDB header-only library (bundled under `nanovdb/`). No separate build step needed for the library itself.
