// NanoVDB_0_3/main.cpp
//
// Compute the discrete 7-point Laplacian over the active voxels of an
// IndexGrid sphere shell.  u  is the signed-distance sidecar from
// initialization, Lu  is a freshly-allocated companion sidecar.
//
// Concepts introduced:
//   - A real per-voxel kernel over the IndexGrid+sidecar topology
//   - Boundary handling "for free" via the background-index trick:
//     inactive neighbours look up u[0] = 0 in the sidecar

#include "nanovdb/NanoVDB.h"
#include "Utility.h"

#include <chrono>

int main()
{
    auto         shell  = initializeSphereShellIndexed(/*D=*/200.0f, /*R=*/3.0f);
    const auto*  grid   = shell.handle.grid<nanovdb::ValueOnIndex>();
    const auto&  u      = shell.values;

    printGridInfo(*grid, "NanoVDB_0_3  (Laplacian on the sphere shell)");
    std::cout << "Sidecar size      : " << u.size() << "\n";

    // Lu sidecar matches u in size; slot 0 stays at 0 (background).
    std::vector<float> Lu(u.size(), 0.0f);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    computeLaplacian(*grid, u, Lu);
    const auto t1 = clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "computeLaplacian elapsed : " << ms << " ms\n";

    // Spot checks.  For f(p) = |p| - D the analytical Laplacian is 2/|p|,
    // so on shell-interior voxels Lu ~ 2/200 = 0.01.  At voxels next to
    // the inner/outer boundary the stencil reaches inactive neighbours
    // (which read as 0 through the background slot), so |Lu| is much
    // larger there -- a useful sanity check that boundary handling is
    // doing what we expect.
    auto acc = grid->getAccessor();
    auto showLu = [&](int x, int y, int z) {
        const nanovdb::Coord c(x, y, z);
        const uint64_t       idx = acc.getValue(c);
        std::cout << "  (" << x << ", " << y << ", " << z << ") : "
                  << "u="  << u [idx]
                  << "  Lu=" << Lu[idx]
                  << "  active=" << (acc.isActive(c) ? "yes" : "no") << "\n";
    };
    std::cout << "\nSample (u, Lu) values (analytical Lu = 2/|p| ~ 0.01 in shell interior):\n";
    showLu(200,   0,   0);  // exactly on the sphere -> shell interior
    showLu(199,   0,   0);  // one voxel inwards     -> shell interior
    showLu(202,   0,   0);  // shell interior, outer half
    showLu(203,   0,   0);  // outer boundary        -> stencil clipped
    showLu(197,   0,   0);  // inner boundary        -> stencil clipped

    return 0;
}
