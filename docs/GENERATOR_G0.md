# G0 tool-trajectory generation contract

G0 is an offline algorithm-feasibility experiment. It maps phase, the initial
tracked fork rigid-body pose, robust plate position, and an upstream-derived
mouth proxy to a complete 101-sample fork SE(3) trajectory. It does not execute
the trajectory on a robot.

The only accepted source is the immutable `phase1-v1.5` manifest and its demo
NPZ files. The upstream field `mouth_target_position` is named `mouth_proxy` in
G0. It is not an independently measured mouth position:

```text
mouth_proxy_is_independently_measured = false
deployment_context_validated = false
```

No physical mouth-position generalization or feeding-success claim follows
from this experiment.

## Representation and split

Each path is represented in its initial fork frame by relative translation and
the SO(3) logarithm of relative orientation. Plate and mouth-proxy positions are
also expressed in that frame. Transfer and withdrawal are modeled separately.
Seven outer folds each hold out one complete take. Held-out records never enter
context scaling, retrieval, ProMP fitting, ridge selection, or duration
statistics.

Retrieval selects the nearest standardized six-dimensional context from the
same phase, with stable record-ID tie breaking. Its relative path is transferred
to the query initial pose. A cubic smoothstep translation adapts the transfer
endpoint context by mouth proxy and the withdrawal endpoint context by plate;
orientation evolution is reused without inventing an endpoint orientation.

The Contextual ProMP uses 12 normalized Gaussian bases for three relative
position and three relative rotation-vector channels. Multi-output ridge maps
standardized context to the conditional mean trajectory weights. Alpha is
selected by inner leave-one-training-take-out position RMS. The predicted
relative pose is anchored on SE(3), including multiplicative orientation
anchoring rather than rotation-vector subtraction. Residual weight covariance
is estimated but trajectories are not sampled.

Both methods use the outer-training phase median duration. Timing is
provisional and has not been checked against physical limits.

## Artifact contract

Each generated NPZ stores identity, held-out fold, training takes/record IDs,
source manifest hash, the 101-point normalized phase and provisional time,
generated position and proper rotation matrices in B, the requested initial
pose, robust plate and mouth-proxy context, generation status, and model
metadata. Pose, time, phase, and plate context are sufficient to reconstruct
the frozen `strong-local-l3-v1` 19-feature input later, without running it in
G0.

G0 does not start G1, robot IK, StrongLocal inference, Exact-SEW, calibration,
collision checking, retiming, base optimization, or a complex trajectory model.
