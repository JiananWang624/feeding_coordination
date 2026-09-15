"""Direct stateful trajectory adapter for the frozen Exact-SEW solver."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..dependency.exact_sew import ExactSewDependency, load_exact_sew_dependency

_SO3_TOL = 1e-10


class ExactSewTrajectoryAdapter:
    """Solve one native-base trajectory with one fresh, internal-state solver."""

    def __init__(
        self,
        dependency_loader: Callable[[], ExactSewDependency] = load_exact_sew_dependency,
    ) -> None:
        self._dependency_loader = dependency_loader

    def solve_trajectory(
        self, positions: Any, rotations: Any, psi: Any
    ) -> list[Any]:
        """Return unmodified frozen results; no external q0 or output fallback exists."""
        p, r, angles = _validate_trajectory(positions, rotations, psi)
        dependency = self._dependency_loader()
        robot = dependency.gen3_kinematics()
        geometry = dependency.geometry_type.from_robot(robot)
        stereo = dependency.stereo_type(dependency.project_reference())
        solver = dependency.solver_type(robot, geometry, stereo)
        return [
            solver.solve(dependency.target_type(p[index], r[index], angles[index]))
            for index in range(len(p))
        ]


def _validate_trajectory(
    positions: Any, rotations: Any, psi: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(positions, dtype=float)
    r = np.asarray(rotations, dtype=float)
    angles = np.asarray(psi, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) == 0:
        raise ValueError("positions must be finite with shape (N, 3), with N >= 1")
    if r.shape != (len(p), 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3) matching positions")
    if angles.shape != (len(p),):
        raise ValueError("psi must have shape (N,) matching positions")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r)) and np.all(np.isfinite(angles))):
        raise ValueError("positions, rotations, and psi must be finite")
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(r, 1, 2) @ r, identity, atol=_SO3_TOL, rtol=0.0):
        raise ValueError("rotations must contain proper SO(3) matrices")
    if not np.allclose(np.linalg.det(r), 1.0, atol=_SO3_TOL, rtol=0.0):
        raise ValueError("rotations must contain proper SO(3) matrices")
    return p.copy(), r.copy(), angles.copy()
