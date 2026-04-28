# NanoVDB — a guided introduction

Pedagogical companion for the CS557 parallel-programming course.
Each numbered subdirectory introduces one new NanoVDB concept and
builds on the previous ones.

## Bundled headers

`nanovdb/` is a trimmed, header-only subset of the NanoVDB library
from the [ASF openvdb repository](https://github.com/AcademySoftwareFoundation/openvdb)
(master branch).  Only the pieces actually used by these examples
were kept — examples, tests, CMake files, Python bindings, and the
C/portable variants (`CNanoVDB.h`, `PNanoVDB.h`) have been removed.

## Building

Each example is a self-contained subdirectory with its own
`Makefile` that includes the shared flags in `../config.mk`:

    cd NanoVDB_0_0
    make               # default: g++
    make CXX=icc       # alternative: Intel compiler
    ./NanoVDB_0_0
    make clean

## Examples

| Directory     | Introduces                                               |
|---------------|----------------------------------------------------------|
| `NanoVDB_0_0` | `build::Grid<float>`, accessor-based voxel insertion, `createNanoGrid()` bake step, read-back via `NanoGrid` accessor |
| `NanoVDB_0_1` | A genuinely sparse topology (spherical shell in a 512³ ambient domain); leaf-aligned coarse rejection; storing a scalar value per active voxel |
| `NanoVDB_0_2` | Same topology as `0_1`, separated into a `NanoGrid<ValueOnIndex>` plus a parallel `std::vector<float>` sidecar; topology-only `build::Grid<ValueMask>` build, leaf-level sparse indexing, sidecar fill in parallel |
| `NanoVDB_0_3` | Discrete 7-point Laplacian over the IndexGrid sphere shell (`u` → `Lu`, both sidecars); inactive neighbours read through index 0 → `u[0] = 0`, so the stencil needs no explicit boundary tests |
