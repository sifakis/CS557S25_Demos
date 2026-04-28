// NanoVDB_0_2/main.cpp
//
// Same sphere-shell topology as NanoVDB_0_1, but represented as a
// NanoGrid<ValueOnIndex> + companion sidecar array instead of a
// NanoGrid<float>.  The grid carries only the topology and a per-voxel
// index; the actual signed-distance values live in a parallel
// std::vector<float>.
//
// Concepts introduced:
//   - Topology-only build::Grid<ValueMask>
//   - Conversion to NanoGrid<ValueOnIndex>: leaf-level sparse indexing,
//     valueCount = 1 (background) + activeVoxelCount
//   - Sidecar pattern: caller-managed array of per-voxel data; lookup
//     is values[acc.getValue(coord)]

#include "nanovdb/NanoVDB.h"
#include "Utility.h"

int main()
{
    auto         shell  = initializeSphereShellIndexed(/*D=*/200.0f, /*R=*/3.0f);
    const auto*  grid   = shell.handle.grid<nanovdb::ValueOnIndex>();
    const auto&  values = shell.values;

    printGridInfo(*grid, "NanoVDB_0_2  (sphere shell, IndexGrid + sidecar)");
    std::cout << "Sidecar size      : " << values.size()
              << "  (= 1 + activeVoxelCount)\n";

    // Sample queries.  For an inactive voxel acc.getValue() returns 0
    // (the background slot); for an active voxel it returns its 1-based
    // index, which we use to look up the float value in the sidecar.
    auto acc = grid->getAccessor();
    std::cout << "\nSample queries (value via sidecar[acc.getValue(coord)]):\n";
    auto showVoxel = [&](int x, int y, int z) {
        const nanovdb::Coord c(x, y, z);
        const bool     active = acc.isActive(c);
        const uint64_t idx    = acc.getValue(c);
        const float    v      = values[idx];
        std::cout << "  (" << x << ", " << y << ", " << z << ") : "
                  << "index=" << idx
                  << "  value=" << v
                  << "  active=" << (active ? "yes" : "no") << "\n";
    };
    showVoxel(200,   0,   0);  // exactly on the shell -> active, value 0
    showVoxel(203,   0,   0);  // outer boundary       -> active, value +3
    showVoxel(197,   0,   0);  // inner boundary       -> active, value -3
    showVoxel(204,   0,   0);  // just outside         -> inactive, idx=0
    showVoxel(  0,   0,   0);  // deep interior        -> inactive, idx=0

    // ------------------------------------------------------------------ //
    // Parallel walk: same as NanoVDB_0_1, but reading through the sidecar //
    // ------------------------------------------------------------------ //
    auto*          leaves    = grid->tree().getFirstLeaf();
    const uint32_t numLeaves = grid->tree().nodeCount(0);

    uint64_t posCount = 0, negCount = 0;

#pragma omp parallel for firstprivate(acc) reduction(+:posCount,negCount)
    for (uint32_t i = 0; i < numLeaves; ++i) {
        const auto& leaf = leaves[i];
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const uint64_t idx = acc.getValue(iter.getCoord());
            const float    v   = values[idx];
            if (v >= 0.0f) ++posCount;
            else           ++negCount;
        }
    }

    std::cout << "\nParallel count over active voxels (reading through sidecar):\n";
    std::cout << "  values >= 0 : " << posCount << "\n";
    std::cout << "  values <  0 : " << negCount << "\n";
    std::cout << "  total       : " << (posCount + negCount) << "\n";

    return 0;
}
