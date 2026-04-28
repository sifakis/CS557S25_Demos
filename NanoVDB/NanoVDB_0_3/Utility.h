#pragma once

#include "nanovdb/NanoVDB.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/HostBuffer.h"
#include <iostream>
#include <string>
#include <vector>

// Print topology diagnostics for a NanoGrid to stdout.
template<typename BuildT>
void printGridInfo(const nanovdb::NanoGrid<BuildT>& grid, const std::string& label = "")
{
    const auto& tree = grid.tree();
    const auto  bbox = grid.indexBBox();

    if (!label.empty())
        std::cout << "=== " << label << " ===\n";

    std::cout << "Grid name         : " << grid.gridName() << "\n";
    std::cout << "Bounding box      : ("
              << bbox.min()[0] << ", " << bbox.min()[1] << ", " << bbox.min()[2] << ") -- ("
              << bbox.max()[0] << ", " << bbox.max()[1] << ", " << bbox.max()[2] << ")\n";
    std::cout << "Node counts       : "
              << tree.mNodeCount[2] << " upper, "
              << tree.mNodeCount[1] << " lower, "
              << tree.mNodeCount[0] << " leaf\n";
    std::cout << "Active voxels     : " << tree.mVoxelCount << "\n";
}

// IndexGrid + sidecar bundle (same as NanoVDB_0_2).  The sidecar's
// slot 0 is reserved for the background (kept at 0); slots 1..N hold
// the signed-distance value (d - D) of each active voxel.
struct SphereShellIndexed {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;  // NanoGrid<ValueOnIndex>
    std::vector<float>                       values;  // size = 1 + activeVoxelCount
};

SphereShellIndexed
initializeSphereShellIndexed(float D = 200.0f, float R = 3.0f);

// Compute the discrete 7-point Laplacian of u into Lu, voxel-by-voxel
// over the active set of the IndexGrid.  Both u and Lu are sized to
// grid.valueCount() and indexed via the grid's accessor.  Boundary
// handling falls out of the IndexGrid+sidecar pattern: when the
// stencil reaches an inactive neighbour, acc.getValue() returns 0 and
// u[0] is the background (also 0) -- so the kernel needs no explicit
// active-flag tests.
void computeLaplacian(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& grid,
                      const std::vector<float>&                       u,
                      std::vector<float>&                             Lu);
