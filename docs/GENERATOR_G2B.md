# G2B: oracle full-pose transfer endpoint feasibility

G2B asks whether the frozen G0 Contextual ProMP path can be smoothly adapted to a **known** terminal fork-rigid-body pose. It is an oracle algorithm test, not an out-of-fold endpoint prediction or a feeding deployment result. It compares the unchanged G0 nominal path, a position-only oracle correction, and a full-pose oracle correction. Withdrawal, robot realization, physical calibration, StrongLocal and Exact-SEW are out of scope.

## Final-frame pose semantics

The source labeled trajectory's `Hand_X`, `Hand_Y`, `Hand_Z` columns are Motive Global positions in millimetres. Phase 1.5 transforms them to `hand_xyz` in project B metres. The tracked fork rigid-body source has `x/y/z` in Motive Global metres and quaternion `qx/qy/qz/qw`; Phase 1.5 joins it by exact NatNet frame identifier, transforms the rigid-body origin to `tool_position` and its quaternion orientation to `tool_orientation` in B. The tool point is the tracked fork rigid-body **origin**, not a calibrated fork tip.

The frozen Phase 1.5 convention checks

```text
p_H = p_U - 0.100 m * R_U[:, 1]
```

so the fork-local offset `d_UH` is `[0, -0.100, 0]` metres. For each transfer, G2B reads `hand_xyz[-1]` and `tool_orientation[-1]` from the **same final frame** of its Phase 1.5 NPZ, then constructs

```text
oracle_hand_goal_position_B = hand_xyz[-1]
oracle_fork_goal_orientation_B = tool_orientation[-1]
oracle_fork_goal_position_B = hand_xyz[-1] + 0.100 m * tool_orientation[-1, :, 1]
```

Before adaptation, the constructed fork pose is checked against the measured terminal G0 transfer pose. All 127 transfer final frames were present and valid in the pre-implementation audit; the maximum position reconstruction residual was 2.01e-6 m and the maximum orientation residual 2.48e-16 rad. The run fails on a larger mismatch rather than silently adapting to an inconsistent target.

Neither the G2A virtual target nor its provisional 18 cm mouth offset is used to construct this oracle. The frozen G0 nominal model does, however, retain its original query context, which includes `mouth_proxy`; G2B does **not** claim to remove that historical context input. Its new terminal target comes only from final-frame Hand position and fork orientation. This use of held-out terminal data is intentional, so `deployment_generalization_test=false`.

## Adaptation and interpretation

G2B reuses each outer fold's frozen G0 ProMP alpha and G2A adaptation start `s_a`. It verifies nominal parity against saved G0 trajectories before modifying anything. Position-only and full-pose variants use the same quintic `h(u)=10u³−15u⁴+6u⁵` from `s_a` to the endpoint. Full-pose orientation applies the SO(3) logarithm of `R_n(1)^T R_g`, scaled by `h(u)` and composed on the right of each nominal orientation. The path before `s_a` remains exactly nominal. The quintic correction has zero first and second derivatives at both boundaries; sampled angular increments are also reported to check for an obvious discrete jump. These are normalized-phase geometric checks, not physical robot speed or acceleration claims.

The hard-contract errors are distance to the **given oracle** terminal pose. Whole-trajectory and phase-specific errors compare the resulting path against a held-out human demonstration; nonzero intermediate error is a trajectory-shape issue, not endpoint-constraint failure. The experiment does not evaluate acquisition of an independent mouth or fork goal, calibrated physical tool accuracy, collision, retiming, IK, or real feeding safety.
