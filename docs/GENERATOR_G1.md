# G1 end-to-end simulated feeding contract

G1 is a replay-style algorithm-feasibility experiment. It consumes the frozen
G0 out-of-fold trajectories, predicts redundancy with a held-out-take-specific
`strong-local-l3-v1` model, and realizes each generated virtual fork trajectory
through the frozen Exact-SEW and provisional Robot R0/R1 geometry.

The complete chain is evaluated for Retrieval and Contextual ProMP with B0,
StrongLocal, and RobotSmooth redundancy policies. A MeasuredUReference uses the
same fold model, measured initial human redundancy, 101-sample phase grid, and
provisional G0 duration. It does not use the measured human redundancy
evolution. Generated-vs-reference joint and elbow differences are full
end-to-end pipeline deviations, not pure redundancy errors.

## Scientific gates

Each outer StrongLocal model is trained on the other six takes with the exact
Phase 3.2 bite weighting, scaler, alpha, and ridge semantics. Before generated-U
inference, its predictions on original held-out measured features must reproduce
the saved Phase 3.2 OOF predictions within `1e-10`.

G1 uses the exact Phase 1.5 row supplying G0's initial valid fork pose. A record
is eligible only if human ψ is valid at that same row. It never substitutes a
later or interpolated ψ₀. Predicted Δψ is anchored to zero at the first sample.

The G0 provisional timing remains a diagnostic convention. Joint velocity,
acceleration, and jerk are not compared with real Gen3 limits and are not a
hardware-executability claim. No derivatives cross Exact-SEW failures or the
frozen 0.5-rad continuity boundary.

## Fixed limitations

```text
mouth_proxy_is_independently_measured = false
deployment_context_validated = false
initial_redundancy_source = measured_human_demo
deployment_initial_redundancy_selection = not yet solved
```

G1 adds no physical P-to-U/fork-tip/mouth calibration, collision model,
retiming, real-robot execution, new redundancy model, or new trajectory
generator. It does not claim physical feeding success or safety.

The visualization path reads stored G1 q only, runs forward kinematics for
display/parity, and overlays generated U, realized U, measured demonstration U,
and the plate context. Video playback uses stored provisional timing without
physical retiming.
