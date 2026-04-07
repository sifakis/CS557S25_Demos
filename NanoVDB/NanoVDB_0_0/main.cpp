// NanoVDB_0_0/main.cpp
//
// A minimal introduction to NanoVDB's build-time grid and the "bake" step.
//
// Concepts introduced:
//   - nanovdb::tools::build::Grid<T>  — a mutable staging grid
//   - The build accessor: setValue / getValue / isActive
//   - nanovdb::tools::createNanoGrid() — compacts the staging grid into an
//     immutable, contiguous NanoGrid ready for compute kernels
//   - Reading values back from a NanoGrid<float> via its own accessor

#include <iostream>
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/GridBuilder.h"
#include "nanovdb/tools/CreateNanoGrid.h"
#include "Utility.h"

int main()
{
    // ------------------------------------------------------------------ //
    // Phase 1 : Build                                                     //
    // ------------------------------------------------------------------ //
    // build::Grid<T> is a mutable, hash-map-backed container.  It lets you
    // insert voxels one at a time in any order without pre-allocating the
    // full domain.  The constructor argument is the background value
    // returned for any coordinate that has never been set (inactive voxels).

    nanovdb::tools::build::Grid<float> buildGrid(/*background=*/0.0f);

    // The accessor caches the path through the tree so that repeated writes
    // to nearby voxels are fast.  Every voxel touched by setValue becomes
    // "active" in the sparse representation.
    auto acc = buildGrid.getAccessor();

    // Three voxels close together — they will land in the same leaf node.
    acc.setValue(nanovdb::Coord(0, 0, 0), 1.0f);
    acc.setValue(nanovdb::Coord(1, 0, 0), 2.0f);
    acc.setValue(nanovdb::Coord(0, 1, 0), 3.0f);

    // A voxel far away — forces allocation of a separate branch of the tree.
    acc.setValue(nanovdb::Coord(100, 200, 300), 42.0f);

    // ------------------------------------------------------------------ //
    // Phase 2 : Bake                                                      //
    // ------------------------------------------------------------------ //
    // createNanoGrid() traverses the mutable build grid and serialises it
    // into a single contiguous memory buffer holding an immutable NanoGrid.
    // The result is a GridHandle that owns that buffer.

    auto handle = nanovdb::tools::createNanoGrid(buildGrid);

    // Retrieve a typed pointer to the NanoGrid inside the handle's buffer.
    const nanovdb::NanoGrid<float>* grid = handle.grid<float>();

    // ------------------------------------------------------------------ //
    // Phase 3 : Inspect                                                   //
    // ------------------------------------------------------------------ //
    printGridInfo(*grid, "NanoVDB_0_0");

    // ------------------------------------------------------------------ //
    // Phase 4 : Read                                                      //
    // ------------------------------------------------------------------ //
    // The NanoGrid has its own (read-only) accessor — separate from the
    // build accessor used above.  Like the build accessor it caches the
    // last-visited tree path, so sequential access to nearby voxels is fast.

    auto readAcc = grid->getAccessor();

    // Active voxels return their stored value.
    std::cout << "\nValue at (  0,  0,  0) : " << readAcc.getValue(nanovdb::Coord(  0,   0,   0)) << "\n";
    std::cout << "Value at (  1,  0,  0) : " << readAcc.getValue(nanovdb::Coord(  1,   0,   0)) << "\n";
    std::cout << "Value at (  0,  1,  0) : " << readAcc.getValue(nanovdb::Coord(  0,   1,   0)) << "\n";
    std::cout << "Value at (100,200,300) : " << readAcc.getValue(nanovdb::Coord(100, 200, 300)) << "\n";

    // An inactive voxel silently returns the background value (0.0f here).
    std::cout << "Value at (  5,  5,  5) : " << readAcc.getValue(nanovdb::Coord(  5,   5,   5))
              << "  <- background (inactive)\n";

    // isActive() lets you distinguish a stored value from the background.
    std::cout << "\nIs (0,0,0) active? " << (readAcc.isActive(nanovdb::Coord(0, 0, 0)) ? "yes" : "no") << "\n";
    std::cout << "Is (5,5,5) active? " << (readAcc.isActive(nanovdb::Coord(5, 5, 5)) ? "yes" : "no") << "\n";

    return 0;
}
