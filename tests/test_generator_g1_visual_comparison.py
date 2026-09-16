from pathlib import Path
import inspect
import json

import numpy as np

from feeding_coordination.generator_g1_visual_comparison import (
    FRAME_COUNT, export_split_video, fk_parity, hud_state, interactive_overlay,
    load_comparison_record, replay_frame,
)
from feeding_coordination.generator_g1_visualization import mount_g1_record
from feeding_coordination.robot_r1 import wrap


ROOT = Path(__file__).resolve().parents[1]
RECORD = "trial_0014_bite_004_transfer"


def test_loads_frozen_g1_and_phase15_human_psi_without_mutating_arrays():
    record = load_comparison_record(RECORD, root=ROOT)
    assert record.frames == FRAME_COUNT == 101
    assert record.human_psi.shape == (101,) and not record.human_psi.flags.writeable
    assert not record.generated.arrays["StrongLocal_q"].flags.writeable
    assert record.generated.record_id == RECORD


def test_displayed_q_is_bitwise_the_frozen_npz_q():
    record = load_comparison_record(RECORD, root=ROOT)
    path = ROOT / "outputs/generator_g1/results/contextual_promp" / f"{RECORD}.npz"
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(record.generated.arrays["StrongLocal_q"], z["StrongLocal_q"])


def test_human_psi_resampling_uses_phase15_segment_endpoints():
    record = load_comparison_record(RECORD, root=ROOT)
    manifest = json.loads((ROOT / "outputs/phase1/manifest.json").read_text())
    entry = next(x for x in manifest["records"] if x["record_id"] == RECORD)
    with np.load(ROOT / "outputs/phase1" / entry["file"], allow_pickle=False) as z:
        time, psi = np.asarray(z["time"], float), np.asarray(z["psi_unwrapped"], float)
        valid = np.asarray(z["psi_valid"], bool) & np.isfinite(psi) & np.isfinite(time)
    expected = np.interp(np.linspace(0, 1, 101), (time[valid] - time[0]) / (time[-1] - time[0]), psi[valid])
    assert np.array_equal(record.human_psi, expected)


def test_hud_uses_saved_psi_and_wrapped_differences():
    record = load_comparison_record(RECORD, root=ROOT); i = 37
    state = hud_state(record, "StrongLocal", i)
    assert state["generated_stronglocal_psi_rad"] == record.generated.arrays["stronglocal_psi"][i]
    assert state["reference_stronglocal_psi_rad"] == record.reference.arrays["stronglocal_psi"][i]
    assert state["human_minus_generated_wrapped_rad"] == wrap(record.human_psi[i] - record.generated.arrays["stronglocal_psi"][i])


def test_hud_tool_and_joint_errors_are_saved_trajectory_comparisons():
    record = load_comparison_record(RECORD, root=ROOT); i = 20; state = hud_state(record, "StrongLocal", i)
    expected = np.linalg.norm(record.generated.arrays["generated_U_position_base"][i] - record.reference.arrays["measured_U_position_base"][i])
    assert state["desired_tool_position_error_m"] == expected
    if state["generated_success"] and state["reference_success"]:
        qg = record.generated.arrays["StrongLocal_q"][i]; qr = record.reference.arrays["StrongLocal_q"][i]
        assert state["robot_wrapped_q_l2_rad"] == np.linalg.norm(wrap(qg - qr))


def test_hud_robot_geometry_comparison_is_available_when_mounted():
    record = load_comparison_record(RECORD, root=ROOT)
    robot, _ = mount_g1_record(record.generated)
    from sew_mimic.sew import Gen3StereoSewGeometry
    state = hud_state(record, "StrongLocal", 20, Gen3StereoSewGeometry.from_robot(robot))
    assert np.isfinite(state["robot_elbow_distance_m"])
    assert np.isfinite(state["robot_arm_plane_angle_rad"])


def test_shared_replay_frame_uses_stored_q_for_interactive_and_export():
    source = inspect.getsource(replay_frame)
    assert "display_stored_q" in source and "actual_virtual_u" in source
    assert "solve" not in source.lower() and "inverse" not in source.lower()
    assert "replay_frame(record" in inspect.getsource(interactive_overlay)
    assert "replay_frame(record" in inspect.getsource(export_split_video)
    failure = load_comparison_record("trial_0015_bite_019_transfer", root=ROOT)
    failed = np.flatnonzero(failure.generated.arrays["StrongLocal_solver_status"].astype(str) != "SUCCESS_EXACT")
    status = hud_state(failure, "StrongLocal", int(failed[0]))["status_label"]
    assert "PIPELINE FAILURE - VIEWER HOLD ONLY" in status and "NO_VALID_BRANCH" in status


def test_fk_parity_for_reference_and_generated_stored_q():
    report = fk_parity(load_comparison_record(RECORD, root=ROOT))
    assert report["frames_checked"] > 0
    assert report["max_position_error_m"] <= 1e-8 and report["max_orientation_error_rad"] <= 1e-8


def test_preset_file_resolves_required_presentation_examples():
    presets = json.loads((ROOT / "outputs/generator_g1_visualization/presets.json").read_text())
    names = set(presets if isinstance(presets, list) else presets.get("presets", presets))
    assert {"promp_clean_transfer", "promp_clean_withdrawal", "retrieval_clean_example",
            "generator_error_propagation", "robot_failure_example"} <= names
