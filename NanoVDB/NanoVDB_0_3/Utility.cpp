#include "Utility.h"

#include "nanovdb/tools/GridBuilder.h"
#include "nanovdb/tools/CreateNanoGrid.h"
#include <cmath>

SphereShellIndexed
initializeSphereShellIndexed(float D, float R)
{
    // Phase 1: topology-only build::Grid<ValueMask>.
    nanovdb::tools::build::Grid<nanovdb::ValueMask> buildGrid(false);
    auto buildAcc = buildGrid.getAccessor();

    const float innerSq = (D - R) * (D - R);
    const float outerSq = (D + R) * (D + R);

    for (int bi = -256; bi < 256; bi += 8)
    for (int bj = -256; bj < 256; bj += 8)
    for (int bk = -256; bk < 256; bk += 8) {

        bool allBeyondOuter = true;
        bool allBelowInner  = true;
        for (int ci = 0; ci <= 7; ci += 7)
        for (int cj = 0; cj <= 7; cj += 7)
        for (int ck = 0; ck <= 7; ck += 7) {
            const int x = bi + ci, y = bj + cj, z = bk + ck;
            const float d2 = float(x)*x + float(y)*y + float(z)*z;
            if (d2 <= outerSq) allBeyondOuter = false;
            if (d2 >= innerSq) allBelowInner  = false;
        }
        if (allBeyondOuter || allBelowInner) continue;

        for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 8; ++k) {
            const int x = bi + i, y = bj + j, z = bk + k;
            const float d = std::sqrt(float(x)*x + float(y)*y + float(z)*z);
            if (std::fabs(d - D) <= R)
                buildAcc.setValue(nanovdb::Coord(x, y, z), true);
        }
    }

    // Phase 2: bake to NanoGrid<ValueOnIndex> (slot 0 = background, slots
    // 1..N = active voxels; valueCount = 1 + activeVoxelCount).
    auto idxHandle = nanovdb::tools::createNanoGrid<
        nanovdb::tools::build::Grid<nanovdb::ValueMask>,
        nanovdb::ValueOnIndex,
        nanovdb::HostBuffer>(buildGrid, /*channels=*/0u,
                                        /*includeStats=*/false,
                                        /*includeTiles=*/false);
    auto* idxGrid = idxHandle.grid<nanovdb::ValueOnIndex>();

    // Phase 3: allocate & fill the sidecar in parallel.
    std::vector<float> values(idxGrid->valueCount(), 0.0f);

    auto*          leaves    = idxGrid->tree().getFirstLeaf();
    const uint32_t numLeaves = idxGrid->tree().nodeCount(0);
    auto           acc       = idxGrid->getAccessor();

#pragma omp parallel for firstprivate(acc)
    for (uint32_t i = 0; i < numLeaves; ++i) {
        const auto& leaf = leaves[i];
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const auto     coord = iter.getCoord();
            const uint64_t idx   = acc.getValue(coord);
            const float    d     = std::sqrt(float(coord.x())*coord.x() +
                                             float(coord.y())*coord.y() +
                                             float(coord.z())*coord.z());
            values[idx] = d - D;
        }
    }

    return { std::move(idxHandle), std::move(values) };
}

void computeLaplacian(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& grid,
                      const std::vector<float>&                       u,
                      std::vector<float>&                             Lu)
{
    auto*          leaves    = grid.tree().getFirstLeaf();
    const uint32_t numLeaves = grid.tree().nodeCount(0);
    auto           acc       = grid.getAccessor();

#pragma omp parallel for firstprivate(acc)
    for (uint32_t i = 0; i < numLeaves; ++i) {
        const auto& leaf = leaves[i];
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const auto     coord = iter.getCoord();
            const uint64_t idxC  = *iter;  // center index (== acc.getValue(coord))

            // Inactive neighbours return idx=0 from the accessor; u[0] is
            // the background (0) -- so the stencil naturally extends with
            // zeros at the shell boundary, no active-flag test required.
            const uint64_t idxXm = acc.getValue(nanovdb::Coord(coord.x()-1, coord.y(),   coord.z()  ));
            const uint64_t idxXp = acc.getValue(nanovdb::Coord(coord.x()+1, coord.y(),   coord.z()  ));
            const uint64_t idxYm = acc.getValue(nanovdb::Coord(coord.x(),   coord.y()-1, coord.z()  ));
            const uint64_t idxYp = acc.getValue(nanovdb::Coord(coord.x(),   coord.y()+1, coord.z()  ));
            const uint64_t idxZm = acc.getValue(nanovdb::Coord(coord.x(),   coord.y(),   coord.z()-1));
            const uint64_t idxZp = acc.getValue(nanovdb::Coord(coord.x(),   coord.y(),   coord.z()+1));

            Lu[idxC] = -6.0f * u[idxC]
                     + u[idxXm] + u[idxXp]
                     + u[idxYm] + u[idxYp]
                     + u[idxZm] + u[idxZp];
        }
    }
}
