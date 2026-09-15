# Architecture

```text
future feeding model
        |
desired utensil task U(t) + desired redundancy psi(t)
        |
thin Exact-SEW adapter
        |
pinned/frozen Exact-SEW dependency
        |
Kinova Gen3 q(t)
```

`external/exact_sew` is the authoritative frozen dependency.  This project does
not copy its configuration, MuJoCo assets, C++ build files, solver policy, or
solver results.

The adapter creates `gen3_kinematics()`,
`Gen3StereoSewGeometry.from_robot(robot)`, and
`StereoSew(project_stereo_sew_reference())`, then creates one `ExactSewSolver`
for one call to `solve_trajectory`. It converts each input row directly into an
`ExactSewTarget` and returns the exact result object from each `solve` call.

There is intentionally no output fallback, failed-q substitution, smoothing,
clipping, or retiming layer. The frozen solver's own state and status are the
sole continuation and failure semantics.

Exact-SEW is the robot-realization dependency, not an implementation owned by
the future feeding-coordination method and not part of that method's novelty.
Human-data processing and learning are intentionally deferred beyond Phase 0.

Phase 1 now provides the upstream human-data boundary: raw Motive landmarks are
explicitly transformed into B-frame measurements and stored with masks,
diagnostics and frozen Stereo-SEW. Tool pose remains unavailable until a real
hand-to-utensil calibration is supplied; Phase 2 and learning remain absent.
