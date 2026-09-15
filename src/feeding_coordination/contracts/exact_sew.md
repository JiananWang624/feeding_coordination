# Exact-SEW trajectory contract

`solve_trajectory(positions, rotations, psi)` receives one canonical target per
row. `positions` has shape `(N, 3)` in metres, `rotations` has shape `(N, 3, 3)`
and each element is a finite proper SO(3) matrix, and `psi` has shape `(N,)` in
radians. All values are expressed in the native Gen3 `base_link` frame and use
the canonical aligned pinch orientation expected by the frozen dependency. A
rotation maps the canonical tool basis into `base_link`; it is not a quaternion
or Euler-angle input.

`psi` is the signed Stereo-SEW angle defined by the frozen reference
`e_t=[0,0,-1]`, `e_r=[1,0,0]`. The adapter does not clip or change it. Creating
the dependency's `ExactSewTarget` applies its established `wrap_to_pi` behavior,
so wrapped and unwrapped equivalent angles have the same target semantics.

The API is stateful but intentionally has no `q0` parameter. At the first frame
the frozen solver makes its canonical global selection; later frames continue
from its internally stored previous successful `q`, branch slot/id, and one or
two wrist-search angles. A failed frame does not overwrite that last successful
state. The dependency's `reset()` clears it; the adapter instead builds a fresh
solver for every `solve_trajectory` call, so state never crosses trajectory
boundaries. The dependency's separate one-frame `solve_exact_sew(...,
q_previous=...)` API is not the validated stateful trajectory path and is not
used here.

`R_robot_align` is already owned by the dependency. Its target conversion is
`rotation @ robot.R_robot_align.T @ geometry.R_7T.T`, while physical evaluation
uses `ee_rotation(q) @ robot.R_robot_align`. Therefore the adapter passes the
aligned target rotation unchanged and never applies `R_robot_align` itself.

Each returned value is the unmodified frozen dependency result: `status`, `q`,
`diagnostics`, and `message` are not translated or replaced. In particular, a
failed frame remains failed with `q=None`, and no previous joint vector is
emitted in its place. On success, `q` has shape `(7,)`, is ordered
`joint_1` through `joint_7`, and is in radians. Status is the dependency's
`SolverStatus`: `SUCCESS_EXACT`, `SUCCESS_APPROX`, `UNREACHABLE`, `JOINT_LIMIT`,
`SEW_SINGULAR`, `NO_VALID_BRANCH`, `INVALID_INPUT`, `NUMERICAL_FAILURE`, or
`LEGACY_FAILURE`. Exact-SEW production successes are `SUCCESS_EXACT`.
Diagnostics preserve position error (m), orientation and SEW errors (rad),
joint-limit margin (rad), solve time (ms), branch id, and method metadata;
`message` carries failure detail when available.
