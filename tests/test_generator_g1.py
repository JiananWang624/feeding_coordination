from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from feeding_coordination.generator_g1 import (
    GENERATORS,
    _paired_bootstrap,
    _solve_stateful,
    anchor_delta,
    audit_eligibility,
    measured_reference,
    reconstruct_fold_models,
    stronglocal_features,
    verify_common_first_target,
)
from feeding_coordination.generator_g1_visualization import display_stored_q
from feeding_coordination.robot_r1 import LOCAL_OFFSETS_RAD


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fold_reconstruction(tmp_path_factory):
    trace = {}
    models, parity = reconstruct_fold_models(ROOT / "outputs/phase1", ROOT / "outputs/phase32",
                                              tmp_path_factory.mktemp("g1_folds"), trace=trace)
    return models, parity, trace


@pytest.fixture(scope="module")
def eligibility():
    import json, hashlib
    phase1 = ROOT / "outputs/phase1"; manifest_path = phase1 / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return audit_eligibility(phase1, ROOT / "outputs/generator_g0", manifest, digest)


def test_g0_artifacts_are_read_only_and_never_retrained():
    source = (ROOT / "src/feeding_coordination/generator_g1.py").read_text()
    assert "process_generator_g0" not in source
    assert "generator_g0/generated" not in source.replace(" / ", "/")
    assert "validate_g0_artifact" in source


def test_invalid_psi0_at_exact_g0_initial_pose_is_explicitly_excluded(eligibility):
    frame, records = eligibility
    excluded = frame.loc[frame.eligibility_status == "NO_VALID_INITIAL_PSI"]
    assert len(frame) == 254 and len(records) == 246 and len(excluded) == 8
    assert not excluded.psi_valid_at_g0_initial_pose.any()


def test_fold_specific_stronglocal_excludes_outer_held_take(fold_reconstruction):
    models, _, trace = fold_reconstruction
    assert len(models) == 7
    assert all(model.held_out_take not in model.training_takes and len(model.training_takes) == 6 for model in models.values())
    assert all(event["held_out_take"] not in event["training_takes"] for event in trace["fold_fit_events"])


def test_fold_model_parity_reproduces_phase32_oof(fold_reconstruction):
    _, parity, _ = fold_reconstruction
    assert parity.parity_passed.all()
    assert parity.max_abs_delta_prediction_error.max() <= 1e-10


def test_generated_features_match_frozen_19d_order():
    s = np.linspace(0, 1, 101); time = 2 * s
    position = np.column_stack([.1 * s, np.zeros(101), np.zeros(101)])
    rotation = Rotation.from_rotvec(np.column_stack([np.zeros(101), np.zeros(101), .2 * s])).as_matrix()
    features = stronglocal_features(position, rotation, time, s, "transfer", np.array([.2, .1, 0.]))
    assert features.shape == (101, 19)
    assert np.array_equal(features[0, 6:12], np.zeros(6))
    assert np.array_equal(features[:, 13], np.ones(101)) and np.array_equal(features[:, 14], np.zeros(101))
    assert np.array_equal(features[:, 18], np.ones(101))


def test_mouth_proxy_never_enters_stronglocal_features():
    signature = inspect.signature(stronglocal_features)
    assert "mouth" not in str(signature).lower()
    assert all("mouth" not in name.lower() for name in stronglocal_features.__code__.co_varnames)


def test_first_delta_is_exactly_zero_after_anchoring():
    raw = np.linspace(.37, .91, 101)
    anchored = anchor_delta(raw)
    assert anchored[0] == 0.0 and np.allclose(anchored, raw - raw[0])


def test_all_strategies_share_identical_first_target_and_q():
    q = np.arange(7, dtype=float)
    payloads = {name: {"solver_status": np.array(["SUCCESS_EXACT"]), "q": q[None].copy()} for name in ("B0", "StrongLocal", "RobotSmooth")}
    result = verify_common_first_target(np.zeros((1, 3)), np.eye(3)[None], .2, payloads)
    assert result["first_position_equal"] and result["first_rotation_equal"]
    assert result["first_psi_equal"] and result["max_first_q_abs_difference_rad"] == 0.0


def test_generated_u_is_single_shared_array_not_strategy_specific():
    source = inspect.getsource(__import__("feeding_coordination.generator_g1", fromlist=["_save_result"])._save_result)
    assert '"generated_U_position_B"' in source and '"generated_U_rotation_B"' in source
    assert "B0_generated_U" not in source and "StrongLocal_generated_U" not in source


def test_exact_sew_failure_remains_explicit():
    class Diagnostics:
        branch_id = None; metadata = {}; solve_time_ms = 1.0
        def to_dict(self): return {"metadata": {}}
    class Result:
        status = "NO_VALID_BRANCH"; q = None; diagnostics = Diagnostics(); message = "failed"
    class Adapter:
        def solve_trajectory(self, positions, rotations, psi): return [Result() for _ in psi]
    payload = _solve_stateful(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), np.zeros(2),
                              object(), adapter_factory=Adapter)
    assert payload["solver_status"].tolist() == ["NO_VALID_BRANCH", "NO_VALID_BRANCH"]
    assert np.isnan(payload["q"]).all()


def test_robot_smooth_frozen_parameters_are_unchanged():
    assert np.array_equal(LOCAL_OFFSETS_RAD, np.array([0., .05, -.05, .10, -.10, .20, -.20, .40, -.40]))
    source = (ROOT / "src/feeding_coordination/generator_g1.py").read_text()
    assert "robot_smooth_run(" in source and "_default_solver()" in source


def test_measured_reference_uses_same_provisional_timing(eligibility):
    _, records = eligibility; source = records[sorted(records)[0]]
    time_s = np.linspace(0, 1.23, 101)
    position, rotation, psi = measured_reference(source, time_s)
    assert position.shape == (101, 3) and rotation.shape == (101, 3, 3) and psi.shape == (101,)
    save_source = inspect.getsource(__import__("feeding_coordination.generator_g1", fromlist=["_save_reference"])._save_reference)
    assert '"time_s": time_s' in save_source


def test_g1_bootstrap_resamples_parent_bites_not_frames():
    rows = []
    for bite in ("b1", "b2"):
        for generator, value in (("retrieval", 2.), ("contextual_promp", 1.)):
            row = {"parent_bite_id": bite, "phase": "transfer", "generator": generator}
            for metric in ("tool_rms_position_error_m", "tool_rms_orientation_error_rad",
                           "generated_vs_measured_reference_delta_psi_mae_rad", "stronglocal_frame_success_rate",
                           "stronglocal_complete_trajectory_success", "stronglocal_continuity_violations",
                           "end_to_end_q_rms_deviation_rad", "end_to_end_elbow_mean_deviation_m",
                           "generated_joint_travel_common_support_rad"):
                row[metric] = value
            rows.append(row)
    result = _paired_bootstrap(pd.DataFrame(rows), 20260915, 20)
    assert set(result.bootstrap_unit) == {"parent_bite"} and set(result.n_parent_bites) == {2}


def test_viewer_uses_stored_q_and_imports_no_ik_solver():
    source = (ROOT / "src/feeding_coordination/generator_g1_visualization.py").read_text()
    assert "display_stored_q" in source and "solver_status" in inspect.getsource(display_stored_q)
    assert "ExactSewTrajectoryAdapter" not in source and "solve_exact_sew" not in source and "_default_solver" not in source


def test_no_physical_calibration_collision_or_retiming_implementation_added():
    source = (ROOT / "src/feeding_coordination/generator_g1.py").read_text()
    assert "collision" not in "\n".join(line for line in source.splitlines() if line.startswith("from ") or line.startswith("import "))
    assert "calibration" not in "\n".join(line for line in source.splitlines() if line.startswith("def "))
    assert "retime" not in "\n".join(line for line in source.splitlines() if line.startswith("def "))
    assert '"physical_calibration_added": False' in source
