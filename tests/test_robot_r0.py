import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from feeding_coordination.phase2 import INPUT_NOT_SOLVED, SUCCESS_EXACT
from feeding_coordination.robot_r0 import (
    LEARNED_STRATEGIES,
    R_INPUT_ALIGN,
    STRATEGIES,
    RobotR0Config,
    StrongLocalModel,
    _empty_solver_payload,
    _prediction_tables,
    anchor_predictions,
    build_strategy_psi,
    common_source_mask,
    compute_motion,
    process_robot_r0,
    realized_virtual_tool_rotation,
    source_runs,
    verify_common_initial_state,
    virtual_pinch_rotation,
)


def _aligned_table(n=4, psi_pred=None):
    if psi_pred is None:
        psi_pred = np.linspace(0.2, 0.5, n)
    return pd.DataFrame(
        {
            "_oof_present": np.ones(n, bool),
            "psi_pred": psi_pred,
            "psi0": np.full(n, 0.1),
            "time": np.arange(n, dtype=float) * 0.01,
            "segment_index": np.arange(n),
        }
    )


def test_virtual_tool_transform_and_no_external_robot_alignment():
    rotation_u = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    rotation_p = virtual_pinch_rotation(rotation_u)
    assert np.array_equal(rotation_p, rotation_u @ R_INPUT_ALIGN)
    assert np.array_equal(realized_virtual_tool_rotation(rotation_p), rotation_u)


def test_source_runs_respect_mask_frame_and_segment_continuity():
    valid = np.array([True, True, False, True, True, True])
    frames = np.array([1, 2, 3, 9, 10, 11])
    segment = np.array([0, 1, 2, 3, 5, 6])
    assert [run.tolist() for run in source_runs(valid, frames, segment)] == [[0, 1], [3], [4, 5]]


def test_learned_anchor_is_exactly_zero_at_run_start():
    raw = np.array([0.35, 0.42, -0.1])
    anchored = anchor_predictions(raw)
    assert anchored[0] == 0.
    assert np.array_equal(anchored, raw - raw[0])


def test_common_source_mask_requires_every_oof_but_no_solver_input():
    n = 4
    tables = {strategy: _aligned_table(n) for strategy in LEARNED_STRATEGIES}
    tables["B2"].loc[2, "_oof_present"] = False
    mask = common_source_mask(
        np.ones(n, bool), np.ones(n, bool), np.zeros((n, 3)),
        np.repeat(np.eye(3)[None], n, axis=0), np.ones(n), np.arange(n, dtype=float), tables,
    )
    assert mask.tolist() == [True, True, False, True]


def test_all_strategies_share_measured_run_psi0_and_human_gt_delta():
    human = np.array([1.0, 1.2, 2.0, 2.5])
    runs = [np.array([0, 1]), np.array([2, 3])]
    tables = {strategy: _aligned_table(4, np.array([0.3, 0.5, -0.1, 0.4]))
              for strategy in LEARNED_STRATEGIES}
    result = build_strategy_psi(human, runs, tables)
    for first in (0, 2):
        assert {result[strategy]["strategy_psi"][first] for strategy in STRATEGIES} == {human[first]}
    assert np.array_equal(result["B0"]["anchored_delta_psi"], np.zeros(4))
    assert np.allclose(result["HumanGT"]["anchored_delta_psi"], [0., .2, 0., .5])


def test_hstar_uses_oof_absolute_prediction_joined_to_phase32_psi0(tmp_path):
    p32 = tmp_path / "p32" / "predictions"; p31 = tmp_path / "p31" / "predictions"
    p32.mkdir(parents=True); p31.mkdir(parents=True)
    base = {"record_id": "r", "parent_bite_id": "b", "take": "t", "phase": "transfer",
            "frame": 10, "motive_frame": 10, "segment_index": 0, "time": 0., "psi0": 0.4,
            "psi_pred": 0.7}
    pd.DataFrame([{**base, "model": model} for model in
                  ("R1_phase_only", "R2_pose_only", "R3_pose_velocity", "StrongLocal")]).to_csv(
        p32 / "oof_predictions.csv", index=False)
    pd.DataFrame([{**base, "model": "H_star", "psi0": np.nan,
                   "delta_pred": np.nan, "psi_pred": 0.9}]).to_csv(
        p31 / "oof_predictions.csv", index=False)
    tables = _prediction_tables(p32.parent, p31.parent)
    assert tables["H_star"].iloc[0].psi0 == 0.4
    assert tables["H_star"].iloc[0].psi_pred == 0.9


def test_motion_never_differences_across_failure_or_source_gap():
    q = np.arange(35., dtype=float).reshape(5, 7) / 100.
    motion = compute_motion(
        q, np.arange(5, dtype=float), np.array([1, 2, 3, 8, 9]), np.array([0, 0, 0, 1, 1]),
        np.array([True, True, False, True, True]), np.array(["a", "b", "b", "a", "a"]),
    )
    assert np.isfinite(motion["wrapped_delta_q_rad"][1]).all()
    assert np.isnan(motion["wrapped_delta_q_rad"][2]).all()
    assert np.isnan(motion["wrapped_delta_q_rad"][3]).all()
    assert np.isfinite(motion["wrapped_delta_q_rad"][4]).all()
    assert motion["branch_changed"].tolist() == [False, True, False, False, False]


def test_initial_validation_accepts_explicit_common_failure_and_rejects_drift():
    payloads = {}
    for strategy in STRATEGIES:
        payload = _empty_solver_payload(1)
        payload["strategy_psi"] = np.array([0.25])
        payloads[strategy] = payload
    check = verify_common_initial_state([np.array([0])], payloads)
    assert check["runs_checked"] == 1
    assert {payloads[strategy]["solver_status"][0] for strategy in STRATEGIES} == {INPUT_NOT_SOLVED}
    payloads["B2"]["solver_status"][0] = SUCCESS_EXACT
    with pytest.raises(RuntimeError, match="status"):
        verify_common_initial_state([np.array([0])], payloads)


def test_saved_stronglocal_model_loader_exact_parity():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "outputs" / "phase32" / "final_model" / "model.npz"
    contract_path = root / "outputs" / "phase32" / "final_model" / "model_contract.json"
    loader = StrongLocalModel(model_path, contract_path)
    features = loader.mean + np.linspace(-0.2, 0.2, 19) * loader.scale
    expected = ((features - loader.mean) / loader.scale) @ loader.coef + loader.intercept
    assert loader.predict_delta(features) == expected


def test_fake_end_to_end_uses_shared_targets_oof_and_preserves_failures(tmp_path):
    phase1 = tmp_path / "phase1"; demos = phase1 / "demos"; demos.mkdir(parents=True)
    p32 = tmp_path / "p32" / "predictions"; p31 = tmp_path / "p31" / "predictions"
    p32.mkdir(parents=True); p31.mkdir(parents=True)
    record = {"record_id": "take_bite_transfer", "parent_bite_id": "take_bite", "source_take": "take",
              "phase": "transfer", "file": "demos/take_bite_transfer.npz"}
    (phase1 / "manifest.json").write_text(json.dumps({"schema_version": "test", "records": [record]}))
    n = 3; frames = np.arange(100, 103); time = np.arange(n) * .01
    np.savez(demos / "take_bite_transfer.npz", motive_frame=frames, time=time,
             phase=np.full(n, "transfer"), tool_position=np.zeros((n, 3)),
             tool_orientation=np.repeat(np.eye(3)[None], n, axis=0), tool_pose_valid=np.ones(n, bool),
             psi_valid=np.ones(n, bool), psi_unwrapped=np.array([.4, .5, .6]),
             shoulder_valid_observation=np.ones(n, bool), shoulder_xyz=np.zeros((n, 3)))
    rows = []
    for model_index, model in enumerate(("R1_phase_only", "R2_pose_only", "R3_pose_velocity", "StrongLocal")):
        for index in range(n):
            rows.append({"record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
                         "take": "take", "phase": "transfer", "motive_frame": frames[index],
                         "segment_index": index, "time": time[index], "psi0": .4,
                         "psi_pred": .4 + .1 * model_index + .02 * index, "model": model})
    pd.DataFrame(rows).to_csv(p32 / "oof_predictions.csv", index=False)
    pd.DataFrame([{**row, "model": "H_star", "psi_pred": .8 + .03 * row["segment_index"],
                   "psi0": np.nan, "delta_pred": np.nan} for row in rows[:n]]).to_csv(
        p31 / "oof_predictions.csv", index=False)

    class Diagnostics:
        branch_id = "raw"
        solve_time_ms = 1.
        metadata = {"search_branch": "branch"}
        def to_dict(self):
            return {"branch_id": self.branch_id, "solve_time_ms": self.solve_time_ms}

    calls = []
    class Adapter:
        def solve_trajectory(self, position, rotation, psi):
            calls.append((position.copy(), rotation.copy(), psi.copy()))
            return [SimpleNamespace(status="NO_VALID_BRANCH", q=None, diagnostics=Diagnostics(), message="kept")
                    for _ in psi]

    robot = SimpleNamespace(frame_body_ids=np.array([0]))
    mounted = SimpleNamespace(xmat=np.eye(3)[None], xpos=np.zeros((1, 3)))
    cfg = RobotR0Config(phase1_output_path=str(phase1), phase32_output_path=str(p32.parent),
                        phase31_output_path=str(p31.parent), output_path=str(tmp_path / "out"),
                        mounting={"name": "Rx(+90deg)", "robot_world_offset_m": [0., .15, .2]},
                        virtual_P_to_U={"translation_m": [0., 0., 0.], "rotation": R_INPUT_ALIGN.T.tolist()})
    output = tmp_path / "out"
    process_robot_r0(phase1, output, cfg, mounting_factory=lambda _: (robot, mounted),
                     adapter_factory=Adapter)
    assert len(calls) == len(STRATEGIES)
    assert all(np.array_equal(call[0], calls[0][0]) and np.array_equal(call[1], calls[0][1]) for call in calls)
    assert {call[2][0] for call in calls} == {.4}
    with np.load(output / "results" / "take_bite_transfer.npz") as result:
        assert np.allclose(result["StrongLocal_anchored_delta_psi"], np.array([0., .02, .04]))
        assert set(result["HumanGT_solver_status"]) == {"NO_VALID_BRANCH"}
        assert np.isnan(result["HumanGT_q"]).all()
        assert "HumanGT_solver_diagnostics_json" in result.files
        assert "HumanGT_actual_virtual_U_rotation_base" in result.files
    assert (output / "strategy_metrics.csv").exists()
    assert (output / "pairwise_metrics.csv").exists()
    assert (output / "run_metrics.csv").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evaluation_predictions"].startswith("OOF only")
    assert summary["virtual_simulation_tool_transform"]["external_R_robot_align_applied"] is False
