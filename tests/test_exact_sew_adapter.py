from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from feeding_coordination.adapters.exact_sew import ExactSewTrajectoryAdapter
from feeding_coordination.dependency.exact_sew import ExactSewDependency


@dataclass
class _Result:
    status: str
    q: object
    diagnostics: object = field(default_factory=dict)
    message: str | None = None


def _fake_dependency(results):
    created = []

    class Robot:
        pass

    class Geometry:
        @classmethod
        def from_robot(cls, robot):
            return cls()

    class Stereo:
        def __init__(self, reference):
            self.reference = reference

    class Target:
        def __init__(self, position, rotation, psi):
            self.position, self.rotation, self.psi = position, rotation, psi

    class Solver:
        def __init__(self, robot, geometry, stereo):
            self.calls = []
            created.append(self)

        def solve(self, target):
            self.calls.append(target)
            return results[len(self.calls) - 1]

    return lambda: ExactSewDependency(lambda: Robot(), Geometry, Stereo, lambda: object(), Solver, Target), created


def test_one_solver_per_trajectory_and_no_hidden_failure_fallback():
    successful = _Result("SUCCESS_EXACT", np.arange(7.0), {"source": "frozen"}, "ok")
    failed = _Result("NUMERICAL_FAILURE", None, {"source": "frozen"}, "failed")
    loader, created = _fake_dependency([successful, failed])
    adapter = ExactSewTrajectoryAdapter(loader)
    positions = np.zeros((2, 3))
    rotations = np.repeat(np.eye(3)[None], 2, axis=0)
    returned = adapter.solve_trajectory(positions, rotations, np.array([0.0, 0.1]))
    assert returned == [successful, failed]
    assert returned[1].q is None
    assert returned[1].diagnostics is failed.diagnostics
    assert len(created) == 1 and len(created[0].calls) == 2
    adapter.solve_trajectory(positions[:1], rotations[:1], np.array([0.0]))
    assert len(created) == 2


@pytest.mark.parametrize(
    ("positions", "rotations", "psi"),
    [
        (np.zeros((0, 3)), np.empty((0, 3, 3)), np.empty((0,))),
        (np.zeros((1, 2)), np.eye(3)[None], np.zeros(1)),
        (np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 1))),
        (np.zeros((1, 3)), np.diag([1.0, 1.0, -1.0])[None], np.zeros(1)),
        (np.array([[np.nan, 0.0, 0.0]]), np.eye(3)[None], np.zeros(1)),
    ],
)
def test_input_validation(positions, rotations, psi):
    with pytest.raises(ValueError):
        ExactSewTrajectoryAdapter().solve_trajectory(positions, rotations, psi)


def test_adapter_matches_direct_frozen_solver_for_first_two_test_csv_frames():
    import sew_mimic
    from sew_mimic.exact import ExactSewSolver, human_arm_to_exact_sew_target
    from sew_mimic.pipeline import prepare_trajectory

    root = Path(__file__).resolve().parents[1] / "external" / "exact_sew"
    assert Path(sew_mimic.__file__).resolve().is_relative_to(root.resolve())
    prepared = prepare_trajectory(root / "data" / "test.csv", max_frames=2)
    targets = [
        human_arm_to_exact_sew_target(frame.target, prepared.stereo)
        for frame in prepared.frames
    ]
    direct_solver = ExactSewSolver(prepared.robot, prepared.geometry, prepared.stereo)
    direct = [direct_solver.solve(target) for target in targets]
    actual = ExactSewTrajectoryAdapter().solve_trajectory(
        np.asarray([target.position for target in targets]),
        np.asarray([target.rotation for target in targets]),
        np.asarray([target.psi for target in targets]),
    )
    assert [result.status for result in actual] == [result.status for result in direct]
    for expected, result in zip(direct, actual, strict=True):
        assert result.message == expected.message
        if expected.q is None:
            assert result.q is None
        else:
            # Separate fresh geometry extraction can perturb residual reporting
            # at roundoff level; the selected joint branch remains identical.
            np.testing.assert_allclose(result.q, expected.q, atol=1e-10, rtol=0.0)
