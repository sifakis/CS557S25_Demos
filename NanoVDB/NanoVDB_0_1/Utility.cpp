#include "Utility.h"

#include "nanovdb/tools/GridBuilder.h"
#include "nanovdb/tools/CreateNanoGrid.h"
#include <cmath>

nanovdb::GridHandle<nanovdb::HostBuffer>
initializeSphereShell(float D, float R)
{
    nanovdb::tools::build::Grid<float> buildGrid(/*background=*/0.0f);
    auto acc = buildGrid.getAccessor();

    const float innerSq = (D - R) * (D - R);  // d^2 threshold for "inside the shell"
    const float outerSq = (D + R) * (D + R);  // d^2 threshold for "outside the shell"

    // Iterate over leaf-aligned 8^3 blocks covering [-256, 255]^3.
    for (int bi = -256; bi < 256; bi += 8)
    for (int bj = -256; bj < 256; bj += 8)
    for (int bk = -256; bk < 256; bk += 8) {

        // Cheap reject: evaluate only the 8 corner voxels of the block.
        // If every corner lies strictly beyond the outer radius D+R, no
        // voxel in the block can be on the shell; similarly if every
        // corner lies strictly inside the inner radius D-R.
        //
        // Correctness rests on the block being a convex region far from
        // the origin: the extrema of |position| over the block are
        // attained at corners, so the 8 corner tests bound all 512
        // interior voxels.
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

        // Fine pass: the block straddles the shell, so test all 512 voxels.
        for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 8; ++k) {
            const int x = bi + i, y = bj + j, z = bk + k;
            const float d = std::sqrt(float(x)*x + float(y)*y + float(z)*z);
            const float signedDist = d - D;
            if (std::fabs(signedDist) <= R)
                acc.setValue(nanovdb::Coord(x, y, z), signedDist);
        }
    }

    return nanovdb::tools::createNanoGrid(buildGrid);
}
