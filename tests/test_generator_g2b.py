import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from feeding_coordination.generator_g0 import GRID, _proper_so3
from feeding_coordination.generator_g2a import constrain_orientation_endpoint, constrain_position_endpoint, quintic_smoothstep_first, quintic_smoothstep_second
from feeding_coordination.generator_g2b import oracle_terminal_pose
from test_generator_g0 import synthetic_record


def _oracle_file(tmp_path: Path, record, terminal_valid=True):
    d = tmp_path / "demos"; d.mkdir()
    R = record.gt_orientation_B[-1]
    p = record.gt_position_B[-1]
    hand = p - .1 * R[:, 1]
    n = 4
    pos = np.repeat(p[None], n, axis=0); ori = np.repeat(R[None], n, axis=0); hands = np.repeat(hand[None], n, axis=0)
    valid = np.ones(n, bool); valid[-1] = terminal_valid
    np.savez_compressed(d / "r.npz", tool_pose_valid=valid, tool_position=pos, tool_orientation=ori, hand_xyz=hands)
    return {"records": [{"record_id": record.record_id, "file": "demos/r.npz"}]}


def test_oracle_uses_terminal_hand_and_same_terminal_fork_orientation(tmp_path):
    record = synthetic_record("r", "held")
    manifest = _oracle_file(tmp_path, record)
    p, R, audit = oracle_terminal_pose(record, tmp_path, manifest, .1)
    assert np.allclose(p, record.gt_position_B[-1], atol=1e-12)
    assert np.array_equal(R, record.gt_orientation_B[-1])
    assert audit["source_final_frame_index"] == 3


def test_oracle_rejects_missing_recorded_terminal_pose(tmp_path):
    record = synthetic_record("r", "held")
    with pytest.raises(ValueError, match="terminal frame"):
        oracle_terminal_pose(record, tmp_path, _oracle_file(tmp_path, record, False), .1)


def test_oracle_rejects_inconsistent_upstream_hand_offset(tmp_path):
    record = synthetic_record("r", "held")
    manifest = _oracle_file(tmp_path, record)
    with pytest.raises(ValueError, match="semantic mismatch"):
        oracle_terminal_pose(record, tmp_path, manifest, .2)


def test_position_constraint_keeps_pre_start_and_hits_oracle_exactly():
    nominal = np.c_[GRID, GRID**2, np.zeros(101)]
    p, delta = constrain_position_endpoint(nominal, np.array([.1, -.2, .4]), .65)
    assert np.array_equal(p[GRID < .65], nominal[GRID < .65])
    assert np.array_equal(delta[GRID < .65], np.zeros((sum(GRID < .65), 3)))
    assert np.array_equal(p[-1], np.array([.1, -.2, .4]))


def test_full_pose_constraint_is_so3_prestart_nominal_and_exact_at_terminal():
    nominal = Rotation.from_rotvec(np.c_[np.zeros(101), GRID * .1, np.zeros(101)]).as_matrix()
    goal = Rotation.from_rotvec([.3, -.2, .1]).as_matrix()
    constrained = constrain_orientation_endpoint(nominal, .65, goal)
    assert np.array_equal(constrained[GRID < .65], nominal[GRID < .65])
    assert _proper_so3(constrained).all()
    assert Rotation.from_matrix(constrained[-1].T @ goal).magnitude() <= 1e-12


def test_position_only_orientation_is_nominal():
    R = Rotation.from_rotvec(np.c_[GRID, np.zeros(101), np.zeros(101)]).as_matrix()
    assert np.array_equal(constrain_orientation_endpoint(R, .5, None), R)


def test_quintic_boundary_first_and_second_derivatives_vanish():
    assert np.allclose(quintic_smoothstep_first([0., 1.]), 0.)
    assert np.allclose(quintic_smoothstep_second([0., 1.]), 0.)


def test_g2b_source_has_no_mouth_target_or_robot_pipeline():
    source = Path("src/feeding_coordination/generator_g2b.py").read_text().lower()
    assert "mouth_proxy_b" not in source
    assert "0.18" not in source
    assert "from .robot_" not in source and "from .exact_sew" not in source and "inverse_kinematics(" not in source
