# Phase 3 contract

Phase 3 consumes only Phase 1.5 human trajectory records. A target row requires
`tool_pose_valid`, `psi_valid`, and finite measured values; no Phase 2 result,
robot status, or mouth-target field is read. Each segment is represented in its
own first-valid-tool frame: `p_rel = R0.T (p-p0)` and
`r_rel = log(R0.T R)`. Linear velocity is a causal finite difference in that
frame; angular velocity is `log(R_prev.T R)/dt` from an immediate measured
predecessor. Missing predecessor values are zero rather than bridged.

Psi runs are defined by contiguous `psi_valid` rows even when a tool row is
missing. `psi0` is the first unwrapped psi in that run and the target is delta
psi. The only full-trajectory features are M5/M6 U-only planned-path features.
Outer evaluation leaves one whole take out, with inner leave-one-training-take
alpha selection. Scalers and fits only see their corresponding training rows;
parent bites have equal total weight.
