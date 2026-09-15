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

Phase 2 is the integration boundary between Phase 1.5 and the adapter. It
freezes one mounted base per take, transforms valid B-frame fork poses to that
base, and passes contiguous measured runs directly to the stateful adapter.
The `P -> U_R` convention is exactly identity for this virtual replay only.

Phase 3 is an independent human-data analysis boundary. It uses Phase 1.5
tool/psi measurements in a segment-local initial-tool coordinate system and
evaluates ridge models by leaving complete takes out. It never consumes Phase 2
outputs, robot statuses, or upstream derived mouth targets.

Phase 1 provides the upstream human-data boundary: raw Motive landmarks are
transformed into B-frame measurements and stored with masks and frozen
Stereo-SEW. Phase 1.5 associates the tracked fork rigid-body pose while keeping
the missing physical fork-tip calibration explicit; Phases 2 and 3 consume
these records downstream without modifying them.
