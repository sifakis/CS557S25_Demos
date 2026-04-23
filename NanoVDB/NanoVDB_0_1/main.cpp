// NanoVDB_0_1/main.cpp
//
// Builds a sparse NanoGrid<float> whose active voxels form a spherical
// shell around a sphere of radius D=200 (shell thickness 2R=6) centered
// at the origin, embedded in an ambient [-256, 255]^3 domain.
//
// Concepts introduced:
//   - A non-trivial sparse topology (a thin shell inside a large ambient box)
//   - Leaf-aligned coarse rejection: checking the 8 corner voxels of each
//     8x8x8 block to skip blocks the shell cannot touch
//   - Storing a per-voxel scalar value (signed distance d - D) alongside
//     the activation pattern

#include "nanovdb/NanoVDB.h"
#include "Utility.h"

int main()
{
    auto        handle = initializeSphereShell(/*D=*/200.0f, /*R=*/3.0f);
    const auto* grid   = handle.grid<float>();

    printGridInfo(*grid, "NanoVDB_0_1  (sphere shell: D=200, R=3)");

    // A handful of sample queries to illustrate the stored values.
    // Expected:  d = sqrt(x^2+y^2+z^2),  stored value = d - 200  when
    // the voxel is active (|d - 200| <= 3), background 0 otherwise.
    auto acc = grid->getAccessor();
    std::cout << "\nSample queries:\n";
    auto showVoxel = [&](int x, int y, int z) {
        const nanovdb::Coord c(x, y, z);
        std::cout << "  (" << x << ", " << y << ", " << z << ") : "
                  << "value=" << acc.getValue(c)
                  << "  active=" << (acc.isActive(c) ? "yes" : "no") << "\n";
    };
    showVoxel(200,   0,   0);  // exactly on the shell         -> active, value 0
    showVoxel(203,   0,   0);  // at the outer shell boundary  -> active, value +3
    showVoxel(197,   0,   0);  // at the inner shell boundary  -> active, value -3
    showVoxel(204,   0,   0);  // just outside the shell       -> inactive
    showVoxel(  0,   0,   0);  // deep in the interior         -> inactive

    // Debug visualisation: the x=0 slice of the domain as an ASCII PGM.
    outputDomainImage(*grid, "domain.pgm", /*R=*/3.0f);
    std::cout << "\nWrote domain.pgm (x=0 slice of the sphere shell)\n";

    // ------------------------------------------------------------------ //
    // Parallel walk over leaf nodes                                       //
    // ------------------------------------------------------------------ //
    // Recommended iteration pattern for a NanoGrid on the CPU:
    //
    //   - Outer loop:  parallel-for over the contiguous array of leaves
    //                  (grid.tree().getFirstLeaf() + grid.tree().nodeCount(0)).
    //                  Leaf-level granularity gives enough work per task
    //                  and keeps each thread writing to disjoint voxels.
    //   - Inner loop:  beginValueOn() visits only the *active* voxels of
    //                  the leaf.
    //   - Accessor:    marked firstprivate so every thread gets its own
    //                  copy.  Accessors cache their traversal path through
    //                  the tree and are NOT thread-safe to share.  (This
    //                  simple count could use *iter directly, but using
    //                  acc here sets up the pattern we will need in later
    //                  examples that look up neighbour voxels.)
    auto*          leaves    = grid->tree().getFirstLeaf();
    const uint32_t numLeaves = grid->tree().nodeCount(0);

    uint64_t posCount = 0, negCount = 0;

#pragma omp parallel for firstprivate(acc) reduction(+:posCount,negCount)
    for (uint32_t i = 0; i < numLeaves; ++i) {
        const auto& leaf = leaves[i];
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const float v = acc.getValue(iter.getCoord());
            if (v >= 0.0f) ++posCount;
            else           ++negCount;
        }
    }

    std::cout << "\nParallel count over active voxels:\n";
    std::cout << "  values >= 0 : " << posCount << "\n";
    std::cout << "  values <  0 : " << negCount << "\n";
    std::cout << "  total       : " << (posCount + negCount) << "\n";

    return 0;
}
