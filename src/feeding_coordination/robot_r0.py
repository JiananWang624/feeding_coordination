"""Robot R0: task-equivalent fixed-tool replay through frozen Exact-SEW."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adapters.exact_sew import ExactSewTrajectoryAdapter
from .phase2 import INPUT_NOT_SOLVED, SUCCESS_EXACT, _finite_so3, base_rotations, base_transform

R_INPUT_ALIGN = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
STRATEGIES = ("B0", "B1", "B2", "B3", "StrongLocal", "H_star", "HumanGT")
LEARNED_STRATEGIES = ("B1", "B2", "B3", "StrongLocal", "H_star")
PHASE32_MODELS = {"B1": "R1_phase_only", "B2": "R2_pose_only", "B3": "R3_pose_velocity", "StrongLocal": "StrongLocal"}
STRONG_LOCAL_FEATURE_ORDER = (
    "relative_tool_position_x", "relative_tool_position_y", "relative_tool_position_z",
    "relative_tool_rotation_x", "relative_tool_rotation_y", "relative_tool_rotation_z",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "normalized_phase_s", "phase_transfer", "phase_withdrawal",
    "plate_relative_x", "plate_relative_y", "plate_relative_z", "plate_valid_mask",
)


@dataclass(frozen=True)
class RobotR0Config:
    phase1_output_path: str = "outputs/phase1"
    phase32_output_path: str = "outputs/phase32"
    phase31_output_path: str = "outputs/phase31"
    output_path: str = "outputs/robot_r0"
    mounting: dict[str, Any] | None = None
    virtual_P_to_U: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> "RobotR0Config":
        raw = json.loads(Path(path).read_text())
        raw.setdefault("mounting", {"name": "Rx(+90deg)", "robot_world_offset_m": [0., .15, .2]})
        raw.setdefault("virtual_P_to_U", {"translation_m": [0., 0., 0.], "rotation": R_INPUT_ALIGN.T.tolist()})
        value = cls(**raw)
        mounting_ok = value.mounting is not None and value.mounting.get("name") == "Rx(+90deg)" and np.array_equal(
            np.asarray(value.mounting.get("robot_world_offset_m"), float), np.array([0., .15, .2]))
        virtual_ok = value.virtual_P_to_U is not None and np.array_equal(
            np.asarray(value.virtual_P_to_U.get("translation_m"), float), np.zeros(3)) and np.array_equal(
            np.asarray(value.virtual_P_to_U.get("rotation"), float), R_INPUT_ALIGN.T)
        if not mounting_ok:
            raise ValueError("Robot R0 mounting is frozen to Rx(+90deg), [0,.15,.2] m")
        if not virtual_ok:
            raise ValueError("Robot R0 virtual P-to-U transform is frozen")
        return value


class StrongLocalModel:
    """Minimal loader for the frozen Phase-3.2 19-D alpha=1 ridge model."""

    def __init__(self, model_path: Path, contract_path: Path):
        with np.load(model_path) as model:
            if not {"mean", "scale", "coef", "intercept", "alpha"}.issubset(model.files):
                raise ValueError("StrongLocal model archive is incomplete")
            self.mean = np.asarray(model["mean"], float).copy()
            self.scale = np.asarray(model["scale"], float).copy()
            self.coef = np.asarray(model["coef"], float).copy()
            self.intercept = float(np.asarray(model["intercept"]).item())
            alpha = float(np.asarray(model["alpha"]).item())
        self.contract = json.loads(Path(contract_path).read_text())
        valid = (
            self.contract.get("schema_version") == "phase32-final-local-ridge-v1"
            and self.contract.get("output") == "predicted_delta_psi"
            and self.contract.get("feature_count") == 19
            and tuple(self.contract.get("feature_order", ())) == STRONG_LOCAL_FEATURE_ORDER
            and self.mean.shape == (19,) and self.scale.shape == (19,) and self.coef.shape == (19,)
            and np.isfinite(self.mean).all() and np.isfinite(self.scale).all()
            and np.all(self.scale != 0.) and np.isfinite(self.coef).all()
            and np.isfinite(self.intercept) and alpha == 1.
            and float(self.contract.get("alpha", np.nan)) == 1.
        )
        if not valid:
            raise ValueError("invalid frozen StrongLocal 19-D alpha=1 contract")

    def predict_delta(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, float)
        if features.shape[-1:] != (19,):
            raise ValueError("StrongLocal requires features in the frozen 19-D order")
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept


def virtual_pinch_rotation(rotation_u: np.ndarray) -> np.ndarray:
    """Desired virtual utensil U -> canonical aligned pinch P."""
    return np.asarray(rotation_u, float) @ R_INPUT_ALIGN


def realized_virtual_tool_rotation(rotation_p: np.ndarray) -> np.ndarray:
    """Realized aligned pinch P -> virtual utensil U."""
    return np.asarray(rotation_p, float) @ R_INPUT_ALIGN.T


def source_runs(valid: np.ndarray, motive_frame: np.ndarray, segment_index: np.ndarray | None = None) -> list[np.ndarray]:
    """Split without crossing invalid rows or original frame/segment gaps."""
    valid = np.asarray(valid, bool)
    frames = np.asarray(motive_frame, int)
    segment = np.arange(len(valid)) if segment_index is None else np.asarray(segment_index, int)
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, ok in enumerate(valid):
        adjacent = index > 0 and frames[index] == frames[index - 1] + 1 and segment[index] == segment[index - 1] + 1
        if ok and (start is None or not adjacent):
            if start is not None:
                runs.append(np.arange(start, index))
            start = index
        elif not ok and start is not None:
            runs.append(np.arange(start, index))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(valid)))
    return runs


def anchor_predictions(raw_predicted_delta_psi: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_predicted_delta_psi, float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("a learned run must contain finite raw delta predictions")
    return raw - raw[0]


def _prediction_tables(phase32_dir: Path, phase31_dir: Path) -> dict[str, pd.DataFrame]:
    phase32 = pd.read_csv(phase32_dir / "predictions" / "oof_predictions.csv", low_memory=False)
    tables = {strategy: phase32.loc[phase32.model == model].copy() for strategy, model in PHASE32_MODELS.items()}
    identity = ["record_id", "motive_frame"]
    for strategy, table in tables.items():
        if table.empty or table.duplicated(identity).any():
            raise ValueError(f"{strategy}: Phase-3.2 OOF identities are missing or duplicated")
    phase31 = pd.read_csv(phase31_dir / "predictions" / "oof_predictions.csv", low_memory=False)
    h_star = phase31.loc[phase31.model == "H_star"].copy()
    if h_star.empty or h_star.duplicated(identity).any():
        raise ValueError("H_star: Phase-3.1 OOF identities are missing or duplicated")
    # Phase 3.1 has no usable psi0; join the exact Phase-3.2 OOF identity.
    base = tables["StrongLocal"][identity + ["psi0"]].copy()
    h_star = h_star.drop(columns=["psi0", "delta_pred"], errors="ignore").merge(
        base, on=identity, how="left", validate="one_to_one")
    if h_star.psi0.isna().any() or len(h_star) != len(base):
        raise ValueError("H_star and Phase-3.2 OOF identities differ")
    tables["H_star"] = h_star
    return tables


def _lookup_predictions(table: pd.DataFrame, record: dict[str, Any], motive_frame: np.ndarray,
                        time_s: np.ndarray, phase: np.ndarray) -> pd.DataFrame:
    selected = table.loc[table.record_id.astype(str) == str(record["record_id"])].copy()
    if selected.duplicated("motive_frame").any():
        raise ValueError(f"{record['record_id']}: duplicate OOF frame identity")
    selected["_oof_present"] = True
    aligned = selected.set_index("motive_frame").reindex(motive_frame)
    present = aligned["_oof_present"].eq(True).to_numpy(bool)
    if present.any():
        indices = np.flatnonzero(present)
        checks = {
            "take": aligned["take"].to_numpy()[indices].astype(str) == str(record["source_take"]),
            "phase": aligned["phase"].to_numpy()[indices].astype(str) == np.asarray(phase).astype(str)[indices],
            "segment_index": aligned["segment_index"].to_numpy(float)[indices] == indices.astype(float),
            "time": np.isfinite(aligned["time"].to_numpy(float)[indices]) & np.isclose(
                aligned["time"].to_numpy(float)[indices], time_s[indices], atol=1e-12, rtol=0.),
        }
        for name, good in checks.items():
            if not np.all(good):
                raise ValueError(f"{record['record_id']}: OOF {name} identity mismatch")
    return aligned


def common_source_mask(tool_pose_valid: np.ndarray, psi_valid: np.ndarray, tool_position: np.ndarray,
                       tool_rotation: np.ndarray, human_psi: np.ndarray, time_s: np.ndarray,
                       predictions: dict[str, pd.DataFrame]) -> np.ndarray:
    """Strategy-independent source mask, with no robot feasibility input."""
    mask = (np.asarray(tool_pose_valid, bool) & np.asarray(psi_valid, bool)
            & np.isfinite(tool_position).all(axis=1) & _finite_so3(tool_rotation)
            & np.isfinite(human_psi) & np.isfinite(time_s))
    for table in predictions.values():
        mask &= (table["_oof_present"].eq(True).to_numpy(bool)
                 & np.isfinite(table.psi_pred.to_numpy(float))
                 & np.isfinite(table.psi0.to_numpy(float))
                 & np.isfinite(table.time.to_numpy(float))
                 & np.isfinite(table.segment_index.to_numpy(float)))
    return mask


def build_strategy_psi(human_psi: np.ndarray, runs: list[np.ndarray],
                       predictions: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """Create raw, anchored, and absolute psi arrays on shared run boundaries."""
    n = len(human_psi)
    result: dict[str, dict[str, np.ndarray]] = {}
    for strategy in STRATEGIES:
        psi0_array = np.full(n, np.nan)
        raw = np.full(n, np.nan)
        anchored = np.full(n, np.nan)
        psi = np.full(n, np.nan)
        for indices in runs:
            psi0 = float(human_psi[indices[0]])
            if strategy == "B0":
                run_raw = np.zeros(len(indices)); run_anchored = run_raw.copy()
            elif strategy == "HumanGT":
                run_raw = np.asarray(human_psi[indices], float) - psi0; run_anchored = run_raw.copy()
            else:
                table = predictions[strategy]
                run_raw = table.psi_pred.to_numpy(float)[indices] - table.psi0.to_numpy(float)[indices]
                run_anchored = anchor_predictions(run_raw)
            psi0_array[indices] = psi0
            raw[indices] = run_raw
            anchored[indices] = run_anchored
            psi[indices] = psi0 + run_anchored
        result[strategy] = {"psi0": psi0_array, "raw_predicted_delta_psi": raw,
                            "anchored_delta_psi": anchored, "strategy_psi": psi}
    return result


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True,
                      default=lambda item: item.item() if isinstance(item, np.generic) else str(item))


def _empty_solver_payload(n: int) -> dict[str, Any]:
    return {
        "solver_status": np.full(n, INPUT_NOT_SOLVED, dtype="<U32"),
        "q": np.full((n, 7), np.nan),
        # Lists are converted by np.savez only after solving so frozen diagnostic
        # strings are never truncated to an arbitrary fixed-width dtype.
        "solver_message": [""] * n,
        "solver_diagnostics_json": ["{}"] * n,
        "branch_id": [""] * n,
        "search_branch": [""] * n,
        "actual_pinch_position_base": np.full((n, 3), np.nan),
        "actual_pinch_rotation_base": np.full((n, 3, 3), np.nan),
        "actual_psi": np.full(n, np.nan),
        "aligned_pinch_position_error_m": np.full(n, np.nan),
        "aligned_pinch_orientation_error_rad": np.full(n, np.nan),
        "psi_error_rad": np.full(n, np.nan),
        "joint_limit_margin_rad": np.full(n, np.nan),
        "solve_time_ms": np.full(n, np.nan),
        "actual_virtual_U_position_base": np.full((n, 3), np.nan),
        "actual_virtual_U_rotation_base": np.full((n, 3, 3), np.nan),
        "virtual_utensil_position_error_m": np.full(n, np.nan),
        "virtual_utensil_orientation_error_rad": np.full(n, np.nan),
    }


def _rotation_error(actual: np.ndarray, desired: np.ndarray) -> float:
    relative = actual.T @ desired
    cosine = np.clip((np.trace(relative) - 1.) / 2., -1., 1.)
    sine = np.linalg.norm(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
         relative[1, 0] - relative[0, 1]]
    ) / 2.
    return float(np.arctan2(sine, cosine))


def compute_motion(q: np.ndarray, time_s: np.ndarray, motive_frame: np.ndarray, run_id: np.ndarray,
                   success: np.ndarray, search_branch: np.ndarray) -> dict[str, np.ndarray]:
    """Finite-difference only uninterrupted, successful source frames."""
    n = len(q)
    step = np.full((n, 7), np.nan); velocity = np.full((n, 7), np.nan)
    acceleration = np.full((n, 7), np.nan); jerk = np.full((n, 7), np.nan)
    branch_changed = np.zeros(n, bool)
    for indices in source_runs(success, motive_frame):
        for previous, current in zip(indices[:-1], indices[1:]):
            if run_id[previous] != run_id[current]:
                continue
            dt = time_s[current] - time_s[previous]
            if not np.isfinite(dt) or dt <= 0.:
                continue
            step[current] = np.arctan2(np.sin(q[current] - q[previous]), np.cos(q[current] - q[previous]))
            velocity[current] = step[current] / dt
            branch_changed[current] = bool(search_branch[previous] and search_branch[current]
                                                   and search_branch[previous] != search_branch[current])
        for previous, current in zip(indices[:-1], indices[1:]):
            dt = time_s[current] - time_s[previous]
            if run_id[previous] == run_id[current] and dt > 0. and np.isfinite(velocity[[previous, current]]).all():
                acceleration[current] = (velocity[current] - velocity[previous]) / dt
        for previous, current in zip(indices[:-1], indices[1:]):
            dt = time_s[current] - time_s[previous]
            if run_id[previous] == run_id[current] and dt > 0. and np.isfinite(acceleration[[previous, current]]).all():
                jerk[current] = (acceleration[current] - acceleration[previous]) / dt
    return {"wrapped_delta_q_rad": step, "joint_velocity_rad_s": velocity,
            "joint_acceleration_rad_s2": acceleration, "joint_jerk_rad_s3": jerk,
            "branch_changed": branch_changed}


def verify_common_initial_state(runs: list[np.ndarray], payloads: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Fail on any run-initial target/status/q discrepancy."""
    maximum_q_difference = 0.; successful = 0
    for indices in runs:
        first = int(indices[0])
        initial_psi = np.array([payloads[strategy]["strategy_psi"][first] for strategy in STRATEGIES])
        if not np.array_equal(initial_psi, np.full(len(STRATEGIES), initial_psi[0])):
            raise RuntimeError("run-initial psi target differs across strategies")
        statuses = [payloads[strategy]["solver_status"][first] for strategy in STRATEGIES]
        if len(set(statuses)) != 1:
            raise RuntimeError("run-initial Exact-SEW status differs across strategies")
        if statuses[0] == SUCCESS_EXACT:
            successful += 1; q0 = payloads[STRATEGIES[0]]["q"][first]
            for strategy in STRATEGIES[1:]:
                difference = float(np.max(np.abs(payloads[strategy]["q"][first] - q0)))
                maximum_q_difference = max(maximum_q_difference, difference)
                if not np.allclose(payloads[strategy]["q"][first], q0, atol=1e-10, rtol=0.):
                    raise RuntimeError("run-initial successful q differs across strategies")
    return {"runs_checked": len(runs), "successful_first_frames_checked": successful,
            "first_target_position_equal": True, "first_target_rotation_equal": True,
            "first_target_psi_equal": True, "first_status_equal": True,
            "max_first_q_abs_difference_rad": maximum_q_difference}


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values):
        return {key: (0 if key == "count" else None) for key in ("count", "mean", "median", "p05", "p95", "min", "max")}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def _flatten_finite(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], float)
    value = np.concatenate([np.asarray(array, float).reshape(-1) for array in arrays])
    return value[np.isfinite(value)]


def _motion_summary(items: list[dict[str, Any]], strategy: str, masks: list[np.ndarray]) -> dict[str, Any]:
    motions: list[dict[str, np.ndarray]] = []; margins: list[np.ndarray] = []
    support_runs = 0
    for item, mask in zip(items, masks):
        payload = item["strategies"][strategy]
        support_runs += len(source_runs(mask, item["motive_frame"]))
        motions.append(compute_motion(payload["q"], item["time_s"], item["motive_frame"], item["run_id"],
                                      mask, payload["search_branch"]))
        margins.append(payload["joint_limit_margin_rad"][mask])
    step = _flatten_finite([motion["wrapped_delta_q_rad"] for motion in motions])
    velocity = _flatten_finite([motion["joint_velocity_rad_s"] for motion in motions])
    acceleration = _flatten_finite([motion["joint_acceleration_rad_s2"] for motion in motions])
    jerk = _flatten_finite([motion["joint_jerk_rad_s3"] for motion in motions])
    margin = _flatten_finite(margins)
    return {
        "support_frames": int(sum(np.asarray(mask, bool).sum() for mask in masks)),
        "support_runs": support_runs,
        "joint_step_support_values": int(len(step)), "total_wrapped_joint_travel_rad": float(np.abs(step).sum()),
        "max_wrapped_joint_step_rad": float(np.abs(step).max()) if len(step) else None,
        "velocity_support_values": int(len(velocity)),
        "rms_joint_velocity_rad_s": float(np.sqrt(np.mean(velocity ** 2))) if len(velocity) else None,
        "max_joint_velocity_rad_s": float(np.abs(velocity).max()) if len(velocity) else None,
        "acceleration_support_values": int(len(acceleration)),
        "rms_joint_acceleration_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))) if len(acceleration) else None,
        "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()) if len(acceleration) else None,
        "jerk_support_values": int(len(jerk)),
        "rms_joint_jerk_rad_s3": float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else None,
        "max_joint_jerk_rad_s3": float(np.abs(jerk).max()) if len(jerk) else None,
        "branch_changes": int(sum(motion["branch_changed"].sum() for motion in motions)),
        "joint_limit_margin_support_frames": int(len(margin)), "joint_limit_margin_rad": _distribution(margin),
    }


def _status_summary(items: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    statuses = np.concatenate([item["strategies"][strategy]["solver_status"][item["source_valid"]] for item in items])
    counts = {str(status): int((statuses == status).sum()) for status in np.unique(statuses)}
    success = counts.get(SUCCESS_EXACT, 0)
    known = success + counts.get("JOINT_LIMIT", 0) + counts.get("NO_VALID_BRANCH", 0)
    source_frames = int(sum(len(item["source_valid"]) for item in items)); attempted = int(len(statuses))
    return {"total_source_frames": source_frames, "source_frames": attempted, "attempted_frames": attempted,
            "valid_runs": int(sum(len(item["runs"]) for item in items)), "SUCCESS_EXACT": success,
            "JOINT_LIMIT": counts.get("JOINT_LIMIT", 0), "NO_VALID_BRANCH": counts.get("NO_VALID_BRANCH", 0),
            "other_failures": attempted - known, "success_percent": float(100. * success / attempted) if attempted else None,
            "status_counts_json": json.dumps(counts, sort_keys=True)}


def _descriptive_summary(items: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    payloads = [item["strategies"][strategy] for item in items]
    successful = [payload["solver_status"] == SUCCESS_EXACT for payload in payloads]
    fields = {
        "abs_psi_difference_from_HumanGT_rad": [np.abs(payload["strategy_psi"][item["source_valid"]]
                                                        - item["human_psi"][item["source_valid"]])
                                                  for item, payload in zip(items, payloads)],
        "aligned_pinch_position_error_m": [payload["aligned_pinch_position_error_m"][mask]
                                             for payload, mask in zip(payloads, successful)],
        "aligned_pinch_orientation_error_rad": [payload["aligned_pinch_orientation_error_rad"][mask]
                                                  for payload, mask in zip(payloads, successful)],
        "psi_error_rad": [payload["psi_error_rad"][mask] for payload, mask in zip(payloads, successful)],
        "virtual_utensil_position_error_m": [payload["virtual_utensil_position_error_m"][mask]
                                               for payload, mask in zip(payloads, successful)],
        "virtual_utensil_orientation_error_rad": [payload["virtual_utensil_orientation_error_rad"][mask]
                                                    for payload, mask in zip(payloads, successful)],
        "solve_time_ms": [payload["solve_time_ms"][item["source_valid"]]
                            for item, payload in zip(items, payloads)],
    }
    return {name: _distribution(_flatten_finite(values)) for name, values in fields.items()}


def _groups(items: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups = [("overall", "overall", items)]
    for phase in sorted({item["record"]["phase"] for item in items}):
        groups.append(("phase", phase, [item for item in items if item["record"]["phase"] == phase]))
    for take in sorted({item["record"]["source_take"] for item in items}):
        groups.append(("take", take, [item for item in items if item["record"]["source_take"] == take]))
    return groups


def _flatten_columns(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update({f"{key}_{nested_key}": nested_value for nested_key, nested_value in value.items()})
        else:
            result[key] = value
    return result


def _summarize_outputs(items: list[dict[str, Any]], output_dir: Path, phase2_summary_path: Path) -> dict[str, Any]:
    strategy_rows: list[dict[str, Any]] = []; pairwise_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {"overall": {}, "per_phase": {}, "per_take": {}}
    for grouping, group, group_items in _groups(items):
        all_masks = [np.logical_and.reduce([item["strategies"][strategy]["solver_status"] == SUCCESS_EXACT
                                            for strategy in STRATEGIES]) for item in group_items]
        feasibility: dict[str, Any] = {}; common: dict[str, Any] = {}
        for strategy in STRATEGIES:
            status = _status_summary(group_items, strategy); descriptive = _descriptive_summary(group_items, strategy)
            strategy_rows.append({"scope": "feasibility", "grouping": grouping, "group": group,
                                  "strategy": strategy, **status, **_flatten_columns(descriptive)})
            feasibility[strategy] = {**status, **descriptive}
            motion = _motion_summary(group_items, strategy, all_masks)
            strategy_rows.append({"scope": "all_method_common_success", "grouping": grouping,
                                  "group": group, "strategy": strategy, **_flatten_columns(motion)})
            common[strategy] = motion
        for left_index, left in enumerate(STRATEGIES):
            for right in STRATEGIES[left_index + 1:]:
                pair_masks = [(item["strategies"][left]["solver_status"] == SUCCESS_EXACT)
                              & (item["strategies"][right]["solver_status"] == SUCCESS_EXACT)
                              for item in group_items]
                for reported in (left, right):
                    motion = _motion_summary(group_items, reported, pair_masks)
                    pairwise_rows.append({"scope": "pairwise_common_success", "grouping": grouping,
                                          "group": group, "strategy_a": left, "strategy_b": right,
                                          "reported_strategy": reported, **_flatten_columns(motion)})
        value = {"feasibility": feasibility, "all_method_common_success": common}
        if grouping == "overall":
            grouped["overall"].update(value)
        else:
            grouped[f"per_{grouping}"][group] = value
    strategy_metrics = pd.DataFrame(strategy_rows); pairwise_metrics = pd.DataFrame(pairwise_rows)
    strategy_metrics.to_csv(output_dir / "strategy_metrics.csv", index=False)
    pairwise_metrics.to_csv(output_dir / "pairwise_metrics.csv", index=False)

    run_rows: list[dict[str, Any]] = []; correction_values = {strategy: [] for strategy in LEARNED_STRATEGIES}
    initial = {"runs_checked": 0, "successful_first_frames_checked": 0, "first_target_position_equal": True,
               "first_target_rotation_equal": True, "first_target_psi_equal": True, "first_status_equal": True,
               "max_first_q_abs_difference_rad": 0.}
    for item in items:
        initial["runs_checked"] += item["initial_validation"]["runs_checked"]
        initial["successful_first_frames_checked"] += item["initial_validation"]["successful_first_frames_checked"]
        initial["max_first_q_abs_difference_rad"] = max(initial["max_first_q_abs_difference_rad"],
                                                         item["initial_validation"]["max_first_q_abs_difference_rad"])
        for current_run, indices in enumerate(item["runs"]):
            for strategy in STRATEGIES:
                payload = item["strategies"][strategy]; success = payload["solver_status"] == SUCCESS_EXACT
                correction = float(abs(payload["raw_predicted_delta_psi"][indices[0]])) if strategy in LEARNED_STRATEGIES else 0.
                if strategy in LEARNED_STRATEGIES:
                    correction_values[strategy].append(correction)
                run_rows.append({"record_id": item["record"]["record_id"],
                                 "parent_bite_id": item["record"]["parent_bite_id"],
                                 "take": item["record"]["source_take"], "phase": item["record"]["phase"],
                                 "run_id": current_run, "strategy": strategy, "source_frames": int(len(indices)),
                                 "SUCCESS_EXACT": int(success[indices].sum()),
                                 "success_percent": float(100. * success[indices].mean()),
                                 "first_psi0_rad": float(payload["psi0"][indices[0]]),
                                 "first_raw_prediction_correction_abs_rad": correction,
                                 "first_target_position_equal": True, "first_target_rotation_equal": True,
                                 "first_target_psi_equal": True, "first_status_equal": True,
                                 **_flatten_columns(_motion_summary([item], strategy,
                                                                    [success & (item["run_id"] == current_run)]))})
    pd.DataFrame(run_rows).to_csv(output_dir / "run_metrics.csv", index=False)

    correction = {strategy: _distribution(np.asarray(values, float)) for strategy, values in correction_values.items()}
    overall_pairs = pairwise_metrics.loc[pairwise_metrics.grouping == "overall"]
    focus_pairs: dict[str, Any] = {}
    for control in ("B0", "B2", "B3", "H_star", "HumanGT"):
        rows = overall_pairs.loc[((overall_pairs.strategy_a == "StrongLocal") & (overall_pairs.strategy_b == control))
                                 | ((overall_pairs.strategy_a == control) & (overall_pairs.strategy_b == "StrongLocal"))]
        focus_pairs[f"StrongLocal_vs_{control}"] = {
            "support_frames": int(rows.support_frames.iloc[0]), "metrics": rows.to_dict("records")}

    current_human = grouped["overall"]["feasibility"]["HumanGT"]
    comparison: dict[str, Any] = {
        "robot_r0_virtual_tool_success_percent": current_human["success_percent"],
        "phase2_identity_tool_success_percent": None, "percentage_point_difference": None,
        "interpretation": "Descriptive only. Remaining failures may reflect provisional zero P-to-U translation, fixed base placement, and uncalibrated physical fork geometry."}
    if phase2_summary_path.exists():
        old_rate = float(json.loads(phase2_summary_path.read_text())["overall"]["success_percent_among_attempts"])
        comparison["phase2_identity_tool_success_percent"] = old_rate
        comparison["percentage_point_difference"] = float(current_human["success_percent"] - old_rate)
    return {
        "schema_version": "robot-r0-v1",
        "virtual_simulation_tool_transform": {"physical_calibration": False, "translation_calibrated": False,
            "translation_P_to_U_m": [0., 0., 0.], "rotation_P_to_U": R_INPUT_ALIGN.T.tolist(),
            "R_input_align": R_INPUT_ALIGN.tolist(), "pinch_target_rule": "R_P = R_U @ R_input_align; p_P = p_U",
            "external_R_robot_align_applied": False},
        "human_model_input_frame": "raw tracked human fork U; unchanged Phase-3.2 semantics",
        "evaluation_predictions": "OOF only; final fitted StrongLocal is deployment-only",
        "strategies": list(STRATEGIES),
        "mounting": "Rx(+90deg), world offset [0,0.15,0.2] m, one fixed anchor per take",
        **grouped, "first_delta_correction_abs_rad": correction,
        "common_initial_state_validation": initial,
        "focused_pairwise_common_success": focus_pairs,
        "HumanGT_virtual_tool_feasibility_comparison": comparison,
        "scope_exclusions": ["physical fork calibration", "fork-tip transform", "P-to-U translation optimization",
                             "base/mounting optimization", "robot-centric redundancy heuristic", "collision checking",
                             "retiming", "real Gen3 execution", "tool trajectory generation", "PCRC/full-plan learning"]}


def process_robot_r0(phase1_dir: Path, output_dir: Path, cfg: RobotR0Config,
                     mounting_factory: Callable[[np.ndarray], tuple[Any, Any]] | None = None,
                     adapter_factory: Callable[[], Any] = ExactSewTrajectoryAdapter) -> dict[str, Any]:
    """Run Robot R0 without consulting Phase-2 feasibility."""
    phase1_dir = Path(phase1_dir).resolve(); output_dir = Path(output_dir).resolve()
    results_dir = output_dir / "results"; plots_dir = output_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    phase32_dir = Path(cfg.phase32_output_path); phase31_dir = Path(cfg.phase31_output_path)
    if not phase32_dir.is_absolute(): phase32_dir = root / phase32_dir
    if not phase31_dir.is_absolute(): phase31_dir = root / phase31_dir
    predictions = _prediction_tables(phase32_dir, phase31_dir)
    manifest1 = json.loads((phase1_dir / "manifest.json").read_text()); records = manifest1["records"]
    if mounting_factory is None:
        from sew_mimic.mounting import load_humanoid_mounted_gen3
        mounting_factory = load_humanoid_mounted_gen3
    from sew_mimic.common import ExactSewTarget, joint_limit_margin
    from sew_mimic.exact.residuals import robot_exact_sew_residuals
    from sew_mimic.kinematics import gen3_kinematics
    from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
    evaluation_robot = gen3_kinematics()
    evaluation_geometry = Gen3StereoSewGeometry.from_robot(evaluation_robot)
    evaluation_stereo = StereoSew(project_stereo_sew_reference())

    mountings: dict[str, dict[str, Any]] = {}
    for take in sorted({record["source_take"] for record in records}):
        candidates: list[tuple[int, np.ndarray]] = []
        for record in records:
            if record["source_take"] != take: continue
            with np.load(phase1_dir / record["file"]) as data:
                valid = data["shoulder_valid_observation"].astype(bool) & np.isfinite(data["shoulder_xyz"]).all(axis=1)
                if valid.any():
                    index = int(np.flatnonzero(valid)[0])
                    candidates.append((int(data["motive_frame"][index]), data["shoulder_xyz"][index].copy()))
        if not candidates: raise ValueError(f"{take}: no valid observed shoulder anchor")
        anchor_frame, anchor = min(candidates, key=lambda candidate: candidate[0])
        mounted_robot, mounted_data = mounting_factory(anchor); base_id = int(mounted_robot.frame_body_ids[0])
        mountings[take] = {"anchor_B": anchor, "anchor_motive_frame": anchor_frame,
            "R_B_from_base": np.asarray(mounted_data.xmat[base_id], float).reshape(3, 3).copy(),
            "p_B_of_base": np.asarray(mounted_data.xpos[base_id], float).copy()}

    items: list[dict[str, Any]] = []
    for record in records:
        with np.load(phase1_dir / record["file"]) as source_data:
            z = {key: source_data[key].copy() for key in source_data.files}
        n = len(z["motive_frame"]); frames = np.asarray(z["motive_frame"], int)
        segment_index = np.arange(n); time_s = np.asarray(z["time"], float); phase = np.asarray(z["phase"])
        tool_position = np.asarray(z["tool_position"], float); tool_rotation = np.asarray(z["tool_orientation"], float)
        human_psi = np.asarray(z["psi_unwrapped"], float)
        aligned_predictions = {strategy: _lookup_predictions(table, record, frames, time_s, phase)
                               for strategy, table in predictions.items()}
        source_valid = common_source_mask(z["tool_pose_valid"], z["psi_valid"], tool_position,
                                          tool_rotation, human_psi, time_s, aligned_predictions)
        runs = source_runs(source_valid, frames, segment_index); run_id = np.full(n, -1, int)
        for current_run, indices in enumerate(runs): run_id[indices] = current_run
        mounting = mountings[record["source_take"]]
        desired_u_position = np.full((n, 3), np.nan); desired_u_rotation = np.full((n, 3, 3), np.nan)
        desired_u_position[source_valid] = base_transform(tool_position[source_valid],
                                                          mounting["R_B_from_base"], mounting["p_B_of_base"])
        desired_u_rotation[source_valid] = base_rotations(tool_rotation[source_valid], mounting["R_B_from_base"])
        desired_pinch_position = desired_u_position.copy(); desired_pinch_rotation = np.full((n, 3, 3), np.nan)
        desired_pinch_rotation[source_valid] = virtual_pinch_rotation(desired_u_rotation[source_valid])
        strategy_psi = build_strategy_psi(human_psi, runs, aligned_predictions)

        payloads: dict[str, dict[str, np.ndarray]] = {}
        for strategy in STRATEGIES:
            payload = {**strategy_psi[strategy], **_empty_solver_payload(n)}
            for indices in runs:
                started = perf_counter()
                solved = adapter_factory().solve_trajectory(desired_pinch_position[indices], desired_pinch_rotation[indices],
                                                             payload["strategy_psi"][indices])
                elapsed_ms = (perf_counter() - started) * 1000.
                if len(solved) != len(indices): raise RuntimeError("trajectory adapter result length mismatch")
                for index, result in zip(indices, solved):
                    diagnostics = result.diagnostics; status = getattr(result.status, "value", str(result.status))
                    payload["solver_status"][index] = status; payload["solver_message"][index] = result.message or ""
                    payload["solver_diagnostics_json"][index] = _json(diagnostics.to_dict())
                    payload["branch_id"][index] = diagnostics.branch_id or ""
                    payload["search_branch"][index] = str(diagnostics.metadata.get("search_branch", ""))
                    payload["solve_time_ms"][index] = diagnostics.solve_time_ms if diagnostics.solve_time_ms is not None else elapsed_ms / len(indices)
                    if status != SUCCESS_EXACT or result.q is None: continue
                    payload["q"][index] = result.q
                    target = ExactSewTarget(desired_pinch_position[index], desired_pinch_rotation[index],
                                            payload["strategy_psi"][index])
                    residual = robot_exact_sew_residuals(result.q, target, evaluation_robot,
                                                         evaluation_geometry, evaluation_stereo)
                    payload["actual_pinch_position_base"][index] = residual.actual_position
                    payload["actual_pinch_rotation_base"][index] = residual.actual_rotation
                    payload["actual_psi"][index] = np.nan if residual.actual_psi is None else residual.actual_psi
                    payload["aligned_pinch_position_error_m"][index] = residual.position_error_m
                    payload["aligned_pinch_orientation_error_rad"][index] = residual.orientation_error_rad
                    payload["psi_error_rad"][index] = np.nan if residual.sew_error_rad is None else residual.sew_error_rad
                    payload["joint_limit_margin_rad"][index] = joint_limit_margin(result.q, evaluation_robot)
                    actual_u_position = residual.actual_position
                    actual_u_rotation = realized_virtual_tool_rotation(residual.actual_rotation)
                    payload["actual_virtual_U_position_base"][index] = actual_u_position
                    payload["actual_virtual_U_rotation_base"][index] = actual_u_rotation
                    payload["virtual_utensil_position_error_m"][index] = np.linalg.norm(actual_u_position - desired_u_position[index])
                    payload["virtual_utensil_orientation_error_rad"][index] = _rotation_error(actual_u_rotation,
                                                                                              desired_u_rotation[index])
            payload.update(compute_motion(payload["q"], time_s, frames, run_id,
                                          payload["solver_status"] == SUCCESS_EXACT, payload["search_branch"]))
            payloads[strategy] = payload

        initial_validation = verify_common_initial_state(runs, payloads)
        result = {"take": np.full(n, record["source_take"]), "parent_bite_id": np.full(n, record["parent_bite_id"]),
                  "record_id": np.full(n, record["record_id"]), "phase": phase, "original_frame": segment_index,
                  "motive_frame": frames, "time_s": time_s, "run_id": run_id, "common_source_valid": source_valid,
                  "desired_U_position_base": desired_u_position, "desired_U_rotation_base": desired_u_rotation,
                  "desired_pinch_position_base": desired_pinch_position, "desired_pinch_rotation_base": desired_pinch_rotation,
                  "human_psi_unwrapped": human_psi}
        for strategy, payload in payloads.items():
            result.update({f"{strategy}_{key}": value for key, value in payload.items()})
        np.savez_compressed(results_dir / f"{record['record_id']}.npz", **result)
        items.append({"record": record, "time_s": time_s, "motive_frame": frames, "human_psi": human_psi,
                      "source_valid": source_valid, "runs": runs, "run_id": run_id,
                      "strategies": payloads, "initial_validation": initial_validation})

    summary = _summarize_outputs(items, output_dir, root / "outputs" / "phase2" / "summary.json")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    phase1_manifest_path = phase1_dir / "manifest.json"
    manifest = {"schema_version": "robot-r0-v1", "config": asdict(cfg), "phase1_manifest": str(phase1_manifest_path),
        "phase1_manifest_sha256": hashlib.sha256(phase1_manifest_path.read_bytes()).hexdigest(),
        "prediction_sources": {"phase32_oof": str(phase32_dir / "predictions" / "oof_predictions.csv"),
            "phase31_H_star_oof": str(phase31_dir / "predictions" / "oof_predictions.csv"),
            "full_fit_used_for_scientific_evaluation": False},
        "virtual_simulation_tool_transform": summary["virtual_simulation_tool_transform"],
        "mountings": {take: {**{key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in mounting.items()}, "name": cfg.mounting["name"],
                              "robot_world_offset_m": cfg.mounting["robot_world_offset_m"]}
                      for take, mounting in mountings.items()},
        "records": [{"record_id": record["record_id"], "source_record": record["file"],
                     "file": f"results/{record['record_id']}.npz"} for record in records], "summary_file": "summary.json"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    overall = pd.DataFrame([{"strategy": strategy, **summary["overall"]["feasibility"][strategy]}
                            for strategy in STRATEGIES])
    figure, axis = plt.subplots(figsize=(9, 4)); axis.bar(overall.strategy, overall.success_percent)
    axis.set_ylabel("SUCCESS_EXACT (%)"); axis.tick_params(axis="x", rotation=25); figure.tight_layout()
    figure.savefig(plots_dir / "feasibility_by_strategy.png", dpi=160); plt.close(figure)
    strict = summary["overall"]["all_method_common_success"]
    figure, axis = plt.subplots(figsize=(9, 4)); axis.bar(
        STRATEGIES, [strict[strategy]["total_wrapped_joint_travel_rad"] for strategy in STRATEGIES])
    axis.set_ylabel("joint travel on all-method common success (rad)")
    axis.tick_params(axis="x", rotation=25); figure.tight_layout()
    figure.savefig(plots_dir / "joint_motion_by_strategy.png", dpi=160); plt.close(figure)
    example = max(items, key=lambda item: max((len(run) for run in item["runs"]), default=0))
    example_run = max(example["runs"], key=len)
    figure, axis = plt.subplots(figsize=(9, 4))
    for strategy in STRATEGIES:
        axis.plot(example["time_s"][example_run], example["strategies"][strategy]["strategy_psi"][example_run],
                  label=strategy, linewidth=2 if strategy in ("StrongLocal", "HumanGT") else 1)
    axis.set(xlabel="time (s)", ylabel="psi (rad)"); axis.legend(ncol=2, fontsize=8); figure.tight_layout()
    figure.savefig(plots_dir / "psi_examples.png", dpi=160); plt.close(figure)
    return summary
