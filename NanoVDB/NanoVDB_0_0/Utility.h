#pragma once

#include "nanovdb/NanoVDB.h"
#include <iostream>
#include <string>

// Print topology diagnostics for a NanoGrid to stdout.
//
// All values are read directly from the non-typed TreeData members
// (mNodeCount, mTileCount, mVoxelCount), so the function works for
// any value type BuildT without any additional overhead.

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

    // mNodeCount[0..2] = leaf, lower internal, upper internal node counts
    std::cout << "Node counts       : "
              << tree.mNodeCount[2] << " upper, "
              << tree.mNodeCount[1] << " lower, "
              << tree.mNodeCount[0] << " leaf\n";

    // mVoxelCount = active voxels at the leaf level
    std::cout << "Active voxels     : " << tree.mVoxelCount << "\n";
}
