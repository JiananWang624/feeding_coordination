# Feeding coordination

This repository is a deliberately thin, stateful adapter around the frozen
Gen3 Exact-SEW implementation in `external/exact_sew`.  Initialize recursively:

```powershell
git submodule update --init --recursive
```

Install the frozen dependency from its own directory so its `config.yaml`,
assets, and compiled C++ extension remain colocated:

```powershell
python -m pip install -e external/exact_sew
python -m pip install -e .[test]
```

`ExactSewTrajectoryAdapter.solve_trajectory(positions, rotations, psi)` accepts
native Gen3 `base_link` target poses: metres, proper SO(3) rotations, radians,
and the canonical aligned pinch orientation. It has no external `q0`: its first
frame uses the frozen solver's canonical global selection and later frames use
that solver instance's internal continuation. A new solver is constructed for
every trajectory call.

The complete software boundary, status semantics, state behavior, and alignment
formula are recorded in
[`src/feeding_coordination/contracts/exact_sew.md`](src/feeding_coordination/contracts/exact_sew.md).
The immutable dependency identity and required assets are recorded in
[`DEPENDENCY_LOCK.json`](DEPENDENCY_LOCK.json).
