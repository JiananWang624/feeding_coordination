"""Nine focused, solver-free contracts for Robot Visualization V0A."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import feeding_coordination.robot_visualization as visualization


ROOT = Path(__file__).resolve().parents[1]
CLEAN_RECORD = "trial_0017_bite_015_transfer"


def test_01_visualization_reads_stored_q_without_solver_calls():
    record = visualization.load_record(CLEAN_RECORD, ROOT)
    expected = record.arrays["B0_q"][0].copy()

    displayed, held = visualization.display_q(record, "B0", 0)

    np.testing.assert_array_equal(displayed, expected)
    assert not held
    source = inspect.getsource(visualization)
    assert "ExactSewTrajectoryAdapter" not in source
    assert ".solve(" not in source
    corrupt = dict(record.arrays)
    corrupt["record_id"] = np.full(record.frames, "wrong_record")
    with pytest.raises(ValueError, match="NPZ record_id identity mismatch"):
        visualization._validate_record_arrays(CLEAN_RECORD, corrupt)


def test_02_correct_per_take_mounting_is_reconstructed():
    record = visualization.load_record(CLEAN_RECORD, ROOT)
    _, _, observed = visualization.mount_record(record)

    np.testing.assert_allclose(observed["R_B_from_base"], record.mounting["R_B_from_base"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(observed["p_B_of_base"], record.mounting["p_B_of_base"], atol=1e-12, rtol=0)
    expected_joint1 = np.asarray(record.mounting["anchor_B"]) + np.asarray(record.mounting["robot_world_offset_m"])
    np.testing.assert_allclose(observed["joint1_B"], expected_joint1, atol=1e-12, rtol=0)


def test_03_replay_fk_matches_stored_robot_r1_actual_pose():
    record = visualization.load_record(CLEAN_RECORD, ROOT)
    assert visualization.parity_check(record) == {"frames_checked": 12}


def test_04_desired_tool_path_is_strategy_independent():
    record = visualization.load_record(CLEAN_RECORD, ROOT)
    robot, _, _ = visualization.mount_record(record)

    first = visualization.replay_frame(record, "B0", 0, ROOT, robot=robot)[2]
    second = visualization.replay_frame(record, "RobotSmooth", 0, ROOT, robot=robot)[2]
    first_path = [(line.start, line.end) for line in first.lines if line.name == "desired_path"]
    second_path = [(line.start, line.end) for line in second.lines if line.name == "desired_path"]

    assert 0 < len(first_path) <= 99
    for (start_a, end_a), (start_b, end_b) in zip(first_path, second_path, strict=True):
        np.testing.assert_array_equal(start_a, start_b)
        np.testing.assert_array_equal(end_a, end_b)


def test_05_strategy_switching_only_selects_stored_q_and_psi():
    record = visualization.load_record(CLEAN_RECORD, ROOT)
    common_before = record.arrays["desired_U_position_base"].copy()

    b0 = record.strategy("B0")
    smooth = record.strategy("RobotSmooth")

    assert b0["q"] is record.arrays["B0_q"]
    assert b0["strategy_psi"] is record.arrays["B0_strategy_psi"]
    assert smooth["q"] is record.arrays["RobotSmooth_q"]
    assert smooth["strategy_psi"] is record.arrays["RobotSmooth_strategy_psi"]
    np.testing.assert_array_equal(record.arrays["desired_U_position_base"], common_before)


def test_06_failure_display_does_not_create_fake_success():
    record = visualization.load_record("trial_0012_bite_004_withdrawal", ROOT)
    payload = record.strategy("B0")
    failure = int(np.flatnonzero(payload["solver_status"] != visualization.SUCCESS)[0])
    status_before = payload["solver_status"][failure]
    run = record.arrays["run_id"][failure]
    previous = np.flatnonzero(
        (record.arrays["run_id"][:failure] == run)
        & (payload["solver_status"][:failure] == visualization.SUCCESS)
    )

    displayed, held = visualization.display_q(record, "B0", failure)

    assert held
    np.testing.assert_array_equal(displayed, payload["q"][previous[-1]])
    assert payload["solver_status"][failure] == status_before != visualization.SUCCESS


def test_07_continuity_event_lookup_resolves_saved_identity():
    expected = pd.read_csv(ROOT / "outputs/robot_r1/continuity_events.csv").iloc[0]
    event = visualization.event_at(ROOT, 0)

    assert event["record_id"] == expected["record_id"]
    assert event["strategy"] == expected["strategy"]
    assert event["current_index"] == expected["current_index"]
    assert event["current_motive_frame"] == expected["current_motive_frame"]


def test_08_dry_run_performs_no_gui_or_solver_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GUI/export path called")

    monkeypatch.setattr(visualization, "interactive_replay", forbidden)
    monkeypatch.setattr(visualization, "export_video", forbidden)

    result = visualization.dry_run(CLEAN_RECORD, "StrongLocal", ROOT)

    assert result["frame_identity_valid"]
    assert result["frames_checked"] == 12
    source = inspect.getsource(visualization.dry_run)
    assert "interactive_replay" not in source and "export_video" not in source and "solve" not in source


def test_09_video_and_interactive_use_the_same_replay_trajectory():
    interactive_source = inspect.getsource(visualization.interactive_replay)
    video_source = inspect.getsource(visualization.export_video)

    for shared in ("replay_frame", "replay_plan", "set_stored_q"):
        assert shared in interactive_source
        assert shared in video_source
