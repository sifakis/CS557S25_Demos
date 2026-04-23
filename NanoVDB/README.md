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
