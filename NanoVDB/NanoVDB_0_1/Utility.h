#pragma once

#include "nanovdb/NanoVDB.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/HostBuffer.h"
#include <iostream>
#include <string>

// Print topology diagnostics for a NanoGrid to stdout.
//
// All values are read directly from the non-typed TreeData members
// (mNodeCount, mVoxelCount), so the function works for any value
// type BuildT without any additional overhead.
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

// Build a NanoGrid<float> whose active voxels form a shell of thickness 2R
// around a sphere of radius D centered at the origin.  A voxel at integer
// coordinates (x,y,z) is activated iff its distance d = sqrt(x^2+y^2+z^2)
// from the origin satisfies  |d - D| <= R,  and its stored value is the
// signed distance  d - D  (positive outside the D-sphere, negative inside).
// Inactive voxels hold the background value 0.
//
// The ambient index-space domain is [-256, 255]^3, visited in 8x8x8 blocks
// aligned with NanoVDB leaf nodes.  Each block is first tested cheaply by
// looking at its 8 corner voxels; only blocks the shell can actually touch
// go on to be tested voxel-by-voxel.
nanovdb::GridHandle<nanovdb::HostBuffer>
initializeSphereShell(float D = 200.0f, float R = 3.0f);

// Write an ASCII PGM image of the x = 0 slice of the [-256, 255]^3 domain.
// Inactive voxels are coloured 0 (black); active voxels have their value
// linearly mapped from [-R, +R] to [64, 191] and clamped to [0, 255].
// The image is 512x512 with +y running left-to-right and +z running
// bottom-to-top (standard mathematical orientation).
void outputDomainImage(const nanovdb::NanoGrid<float>& grid,
                       const std::string&              filename = "domain.pgm",
                       float                           R        = 3.0f);
