#pragma once

#include "nanovdb/NanoVDB.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/HostBuffer.h"
#include <iostream>
#include <string>
#include <vector>

// Print topology diagnostics for a NanoGrid to stdout.  All values are
// read directly from the non-typed TreeData members, so the function
// works for any value type BuildT (float, ValueOnIndex, ...).
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

// A NanoVDB index-grid topology plus its companion "sidecar" array of
// per-voxel data.  The grid is a NanoGrid<ValueOnIndex>: it holds only
// the topology (which voxels are active) and, for every active voxel,
// a unique index in [1 .. activeVoxelCount].  Index 0 is reserved for
// the background (inactive voxels).  The actual per-voxel data lives
// in `values`, whose i-th entry is the value associated with index i.
struct SphereShellIndexed {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;  // NanoGrid<ValueOnIndex>
    std::vector<float>                       values;  // sidecar: size == valueCount
};

// Build the same sphere-shell topology as initializeSphereShell() in
// NanoVDB_0_1, but represent it as an IndexGrid + sidecar rather than
// a float-valued grid.
//
// Step 1 : build a topology-only build::Grid<ValueMask>; only the
//          active-voxel pattern is recorded, no numeric values.
// Step 2 : bake directly to a NanoGrid<ValueOnIndex>.  With
//          includeStats=false and includeTiles=false the grid's
//          valueCount is exactly 1 + activeVoxelCount.
// Step 3 : allocate the sidecar array of size valueCount (slot 0 stays
//          at background 0, slots 1..N hold the per-voxel signed
//          distances) and fill it in parallel by walking the leaves
//          of the IndexGrid.
SphereShellIndexed
initializeSphereShellIndexed(float D = 200.0f, float R = 3.0f);
