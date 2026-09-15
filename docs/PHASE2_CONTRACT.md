# Phase 2 contract

Phase 2 is a downstream replay artifact.  It reads immutable Phase 1.5 NPZs and
writes one result NPZ per source segment.  Each take is mounted exactly once at
its earliest observed finite shoulder using frozen `load_humanoid_mounted_gen3`
with `Rx(+90deg)` and `[0, 0.15, 0.2] m`; subsequent shoulder movement never
remounts the robot.

For B-frame fork pose `(p_B, R_B_U)`, the native base target is
`p_base=(p_B-p_B_of_base) @ R_B_from_base` and
`R_base_U=R_B_from_base.T @ R_B_U`.  The virtual `P -> U_R` transform is exactly
identity, so no fork-tip calibration or extra alignment rotation is applied.
`psi_wrapped` is retained as the target.  Transformed human S/E/W recompute
Stereo-SEW solely as an audit.

`input_valid` requires valid and finite fork pose and psi. Invalid rows remain
`INPUT_NOT_SOLVED`; no values are interpolated. Valid rows split at invalid or
non-consecutive Motive frames. Every run gets a fresh `ExactSewTrajectoryAdapter`
and calls it once. Frozen solver status remains authoritative. Successful exact
solutions are independently checked with frozen physical residual utilities.
