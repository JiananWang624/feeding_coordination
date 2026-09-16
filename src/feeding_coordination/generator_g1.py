"""G1: end-to-end simulated feeding pipeline over frozen G0 trajectories.

G1 is an algorithm-feasibility replay.  It consumes, but never retrains, G0;
reconstructs the seven scientific StrongLocal folds; and uses the frozen
Robot-R0/R1 geometry and Exact-SEW semantics.  Nothing here is a physical
calibration, collision check, retiming policy, or real-robot command path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .adapters.exact_sew import ExactSewTrajectoryAdapter
from .generator_g0 import GRID, _proper_so3, _resample_position, slerp_rotations
from .phase1 import _revision
from .phase2 import INPUT_NOT_SOLVED, SUCCESS_EXACT, base_rotations, base_transform
from .phase3 import Phase3Config, build_dataset
from .phase32 import STRONG_LOCAL_ORDER, _fit_components, add_baseline_features
from .robot_r0 import (
    R_INPUT_ALIGN,
    _empty_solver_payload,
    _rotation_error,
    realized_virtual_tool_rotation,
    virtual_pinch_rotation,
)
from .robot_r1 import (
    CONTINUITY_THRESHOLD_RAD,
    LOCAL_OFFSETS_RAD,
    _default_solver,
    clean_motion,
    continuity_labels,
    robot_smooth_run,
    wrap,
)
from .robot_r2 import arm_plane_angle


GENERATORS = ("retrieval", "contextual_promp")
STRATEGIES = ("B0", "StrongLocal", "RobotSmooth")
REFERENCE = "measured_reference"
RESULT_CACHE_VERSION = "generator-g1-frozen-solver-v1"


@dataclass(frozen=True)
class GeneratorG1Config:
    phase1_output_path: str = "outputs/phase1"
    phase32_output_path: str = "outputs/phase32"
    generator_g0_output_path: str = "outputs/generator_g0"
    robot_r0_output_path: str = "outputs/robot_r0"
    output_path: str = "outputs/generator_g1"
    parity_tolerance: float = 1e-10
    task_position_tolerance_m: float = 1e-8
    task_orientation_tolerance_rad: float = 1e-8
    task_psi_tolerance_rad: float = 1e-8
    continuity_threshold_rad: float = 0.5
    bootstrap_seed: int = 20260915
    bootstrap_resamples: int = 2000
    generated_samples: int = 101

    @classmethod
    def load(cls, path: Path) -> "GeneratorG1Config":
        cfg = cls(**json.loads(Path(path).read_text()))
        if cfg.parity_tolerance != 1e-10:
            raise ValueError("G1 fold parity tolerance is frozen to 1e-10")
        if cfg.continuity_threshold_rad != CONTINUITY_THRESHOLD_RAD or cfg.continuity_threshold_rad != .5:
            raise ValueError("G1 continuity threshold is frozen to 0.5 rad")
        if cfg.bootstrap_seed != 20260915 or cfg.bootstrap_resamples != 2000:
            raise ValueError("G1 bootstrap is frozen to seed 20260915 and 2000 resamples")
        if cfg.generated_samples != 101:
            raise ValueError("G1 requires the frozen 101-sample G0 grid")
        return cfg


@dataclass(frozen=True)
class FoldModel:
    held_out_take: str
    training_takes: tuple[str, ...]
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    parity_max_abs_error: float

    def predict_delta(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, float)
        if features.shape[-1:] != (19,) or not np.isfinite(features).all():
            raise ValueError("StrongLocal requires finite features in frozen 19-D order")
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _storable_array(value: Any) -> np.ndarray:
    """NPZ artifacts stay loadable with allow_pickle=False."""
    array = np.asarray(value)
    return array.astype(str) if array.dtype == object else array


def reconstruct_fold_models(phase1_dir: Path, phase32_dir: Path, fold_dir: Path,
                            tolerance: float = 1e-10, trace: dict | None = None) -> tuple[dict[str, FoldModel], pd.DataFrame]:
    """Rebuild exact Phase-3.2 outer folds and enforce OOF parity before G1."""
    phase1_dir, phase32_dir, fold_dir = Path(phase1_dir), Path(phase32_dir), Path(fold_dir)
    source_manifest = phase1_dir / "manifest.json"
    source_hash = _sha256(source_manifest)
    summary = json.loads((phase32_dir / "summary.json").read_text())
    data, _ = build_dataset(phase1_dir, Phase3Config())
    data = add_baseline_features(data)
    saved = pd.read_csv(phase32_dir / "predictions" / "oof_predictions.csv", low_memory=False)
    saved = saved.loc[saved.model.astype(str) == "StrongLocal"].copy()
    models, rows = {}, []
    for held in sorted(data["take"].unique()):
        train, test = data.loc[data["take"] != held], data.loc[data["take"] == held]
        alpha = float(summary["selected_alphas"][held]["StrongLocal"])
        mean, scale, fit = _fit_components(train, "StrongLocal", alpha)
        delta = fit.predict((np.vstack(test["StrongLocal"]) - mean) / scale)
        calculated = test[["record_id", "frame"]].copy()
        calculated["reconstructed_delta_pred"] = delta
        expected = saved.loc[saved["take"].astype(str) == str(held),
                             ["record_id", "motive_frame", "delta_pred"]]
        joined = calculated.merge(expected, left_on=["record_id", "frame"],
                                  right_on=["record_id", "motive_frame"], validate="one_to_one")
        if len(joined) != len(test):
            raise RuntimeError(f"{held}: Phase-3.2 parity identity coverage mismatch")
        errors = np.abs(joined.reconstructed_delta_pred.to_numpy() - joined.delta_pred.to_numpy())
        maximum = float(errors.max())
        passed = bool(maximum <= tolerance)
        training_takes = tuple(sorted(train["take"].unique()))
        if held in training_takes:
            raise RuntimeError(f"{held}: held-out take leaked into StrongLocal training")
        if trace is not None:
            trace.setdefault("fold_fit_events", []).append({"held_out_take": held, "training_takes": training_takes})
        model = FoldModel(str(held), training_takes, alpha, mean.copy(), scale.copy(),
                          np.asarray(fit.coef_, float).copy(), float(fit.intercept_), maximum)
        models[str(held)] = model
        target = fold_dir / f"heldout_{held}"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "model.npz", mean=model.mean, scale=model.scale,
                            coef=model.coef, intercept=np.array(model.intercept), alpha=np.array(alpha))
        contract = {
            "schema_version": "generator-g1-fold-stronglocal-v1",
            "held_out_take": held, "training_takes": list(training_takes), "alpha": alpha,
            "feature_contract_version": "strong-local-l3-v1", "feature_count": 19,
            "feature_order": list(STRONG_LOCAL_ORDER), "target": "delta_psi=psi_unwrapped-psi0",
            "bite_weighting": "each row weight = 1 / rows in parent_bite",
            "scaler": "bite-weighted mean and population standard deviation; scale<1e-12 -> 1",
            "phase1_manifest_sha256": source_hash, "phase32_oof_parity_max_abs_delta_error": maximum,
            "phase32_oof_parity_tolerance": tolerance, "phase32_oof_parity_passed": passed,
        }
        (target / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True))
        rows.append({"held_out_take": held, "training_takes": "|".join(training_takes), "alpha": alpha,
                     "test_frames": len(test), "max_abs_delta_prediction_error": maximum,
                     "tolerance": tolerance, "parity_passed": passed})
    parity = pd.DataFrame(rows)
    if not parity.parity_passed.all():
        failed = parity.loc[~parity.parity_passed].to_dict("records")
        raise RuntimeError(f"mandatory fold-model parity gate failed: {failed}")
    return models, parity


def stronglocal_features(position_B: np.ndarray, orientation_B: np.ndarray, time_s: np.ndarray,
                         normalized_phase: np.ndarray, phase: str, plate_B: np.ndarray,
                         plate_valid: bool = True) -> np.ndarray:
    """Frozen strong-local-l3-v1 order, with no mouth-proxy feature."""
    position_B, orientation_B = np.asarray(position_B, float), np.asarray(orientation_B, float)
    time_s, normalized_phase = np.asarray(time_s, float), np.asarray(normalized_phase, float)
    n = len(position_B)
    if (position_B.shape != (n, 3) or orientation_B.shape != (n, 3, 3)
            or time_s.shape != (n,) or normalized_phase.shape != (n,)
            or phase not in ("transfer", "withdrawal") or not _proper_so3(orientation_B).all()):
        raise ValueError("invalid generated U for StrongLocal feature construction")
    if n != 101 or not np.isfinite(position_B).all() or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError("G1 features require 101 finite samples and strictly increasing provisional time")
    p0, R0 = position_B[0], orientation_B[0]
    position_rel = (R0.T @ (position_B - p0).T).T
    rotation_rel = np.einsum("ij,njk->nik", R0.T, orientation_B)
    rotvec = Rotation.from_matrix(rotation_rel).as_rotvec()
    velocity = np.zeros((n, 6))
    dt = np.diff(time_s)
    velocity[1:, :3] = np.diff(position_rel, axis=0) / dt[:, None]
    increments = np.einsum("nij,njk->nik", orientation_B[:-1].transpose(0, 2, 1), orientation_B[1:])
    velocity[1:, 3:] = Rotation.from_matrix(increments).as_rotvec() / dt[:, None]
    plate_rel = R0.T @ (np.asarray(plate_B, float) - p0) if plate_valid else np.zeros(3)
    phase_onehot = np.array([phase == "transfer", phase == "withdrawal"], float)
    features = np.column_stack([
        position_rel, rotvec, velocity, normalized_phase,
        np.tile(phase_onehot, (n, 1)), np.tile(plate_rel, (n, 1)),
        np.full(n, float(plate_valid)),
    ])
    if features.shape != (101, 19) or not np.isfinite(features).all():
        raise RuntimeError("generated features violate frozen StrongLocal 19-D contract")
    return features


def anchor_delta(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, float)
    if raw.shape != (101,) or not np.isfinite(raw).all():
        raise ValueError("StrongLocal raw delta must contain 101 finite samples")
    anchored = raw - raw[0]
    anchored[0] = 0.0
    return anchored


def validate_g0_artifact(path: Path, record_id: str, take: str, phase: str,
                         source_hash: str, generator: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    checks = {
        "record_id": str(arrays.get("record_id", "")) == record_id,
        "take": str(arrays.get("take", "")) == take,
        "phase": str(arrays.get("phase", "")) == phase,
        "generator": str(arrays.get("generator", "")) == generator,
        "outer_fold": str(arrays.get("outer_fold", "")) == take,
        "held_out_excluded": bool(arrays.get("held_out_take_excluded", False)),
        "source_hash": str(arrays.get("source_phase1_manifest_sha256", "")) == source_hash,
        "generation_status": str(arrays.get("generation_status", "")) == "SUCCESS",
        "timing_model": str(arrays.get("timing_model", "")) == "training_phase_median",
    }
    training_takes = set(np.asarray(arrays.get("training_take_ids", []), str).tolist())
    checks["training_take_exclusion"] = take not in training_takes and len(training_takes) == 6
    position = np.asarray(arrays.get("generated_tool_position_B"), float)
    rotation = np.asarray(arrays.get("generated_tool_orientation_B"), float)
    normalized = np.asarray(arrays.get("normalized_phase"), float)
    time_s = np.asarray(arrays.get("time_s"), float)
    checks.update({
        "shapes": position.shape == (101, 3) and rotation.shape == (101, 3, 3)
                  and normalized.shape == (101,) and time_s.shape == (101,),
        "finite": np.isfinite(position).all() and np.isfinite(rotation).all() and np.isfinite(time_s).all(),
        "so3": rotation.shape == (101, 3, 3) and _proper_so3(rotation).all(),
        "initial_position": position.shape == (101, 3) and np.allclose(position[0], arrays["initial_tool_position_B"], atol=1e-12, rtol=0),
        "initial_rotation": rotation.shape == (101, 3, 3) and np.allclose(rotation[0], arrays["initial_tool_orientation_B"], atol=1e-12, rtol=0),
        "provisional_time": time_s.shape == (101,) and np.allclose(time_s, normalized * time_s[-1], atol=1e-12, rtol=0)
                            and np.all(np.diff(time_s) > 0),
    })
    if not all(checks.values()):
        raise ValueError(f"{record_id}/{generator}: frozen G0 artifact mismatch {checks}")
    return arrays


def audit_eligibility(phase1_dir: Path, g0_dir: Path, source_manifest: dict,
                      source_hash: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows, eligible = [], {}
    for entry in source_manifest["records"]:
        record_id, take, phase = entry["record_id"], entry["source_take"], entry["phase"]
        with np.load(phase1_dir / entry["file"], allow_pickle=False) as z:
            position, rotation = np.asarray(z["tool_position"], float), np.asarray(z["tool_orientation"], float)
            valid = np.asarray(z["tool_pose_valid"], bool) & np.isfinite(position).all(1) & np.isfinite(rotation).all((1, 2))
            if not valid.any():
                raise RuntimeError(f"{record_id}: G0 record unexpectedly lacks an initial valid tool pose")
            index = int(np.flatnonzero(valid)[0])
            psi_valid = bool(z["psi_valid"][index]) and np.isfinite(z["psi_unwrapped"][index])
            psi0 = float(z["psi_unwrapped"][index]) if psi_valid else np.nan
            source_p0, source_R0 = position[index].copy(), rotation[index].copy()
            source_time = np.asarray(z["time"], float).copy()
            measured_psi = np.asarray(z["psi_unwrapped"], float).copy()
            measured_psi_valid = np.asarray(z["psi_valid"], bool).copy() & np.isfinite(measured_psi)
            source_arrays = {key: z[key].copy() for key in ("motive_frame", "tool_position", "tool_orientation", "tool_pose_valid", "plate_position", "plate_position_valid")}
        initial_errors = []
        for generator in GENERATORS:
            arrays = validate_g0_artifact(g0_dir / "generated" / generator / f"{record_id}.npz",
                                          record_id, take, phase, source_hash, generator)
            initial_errors.append((float(np.linalg.norm(arrays["initial_tool_position_B"] - source_p0)),
                                   float(Rotation.from_matrix(source_R0.T @ arrays["initial_tool_orientation_B"]).magnitude())))
        position_error = max(x[0] for x in initial_errors)
        rotation_error = max(x[1] for x in initial_errors)
        if position_error > 1e-12 or rotation_error > 1e-12:
            raise RuntimeError(f"{record_id}: G0 initial pose does not match authoritative Phase-1.5 row")
        status = "G1_ELIGIBLE" if psi_valid else "NO_VALID_INITIAL_PSI"
        rows.append({"record_id": record_id, "parent_bite_id": entry["parent_bite_id"], "take": take,
                     "phase": phase, "g0_initial_source_index": index,
                     "g0_initial_motive_frame": int(source_arrays["motive_frame"][index]),
                     "initial_position_error_m": position_error, "initial_orientation_error_rad": rotation_error,
                     "psi_valid_at_g0_initial_pose": psi_valid, "psi0": psi0, "eligibility_status": status})
        if psi_valid:
            eligible[record_id] = {"entry": entry, "psi0": psi0, "initial_index": index,
                                   "source_time": source_time, "measured_psi": measured_psi,
                                   "measured_psi_valid": measured_psi_valid, "source_arrays": source_arrays}
    return pd.DataFrame(rows), eligible


def measured_reference(source: dict[str, Any], generated_time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = source["source_arrays"]
    position, rotation = np.asarray(arrays["tool_position"], float), np.asarray(arrays["tool_orientation"], float)
    valid = np.asarray(arrays["tool_pose_valid"], bool) & np.isfinite(position).all(1) & np.isfinite(rotation).all((1, 2))
    time = np.asarray(source["source_time"], float)
    duration = time[-1] - time[0]
    source_s = (time[valid] - time[0]) / duration
    measured_position = _resample_position(source_s, position[valid], GRID)
    measured_rotation = slerp_rotations(source_s, rotation[valid], GRID)
    measured_position[0] = position[source["initial_index"]]
    measured_rotation[0] = rotation[source["initial_index"]]
    psi_valid = np.asarray(source["measured_psi_valid"], bool)
    psi_s = (time[psi_valid] - time[0]) / duration
    human_psi = np.interp(GRID, psi_s, np.asarray(source["measured_psi"])[psi_valid])
    if np.asarray(generated_time_s).shape != (101,) or not np.all(np.diff(generated_time_s) > 0):
        raise ValueError("MeasuredUReference requires frozen provisional G0 timing")
    return measured_position, measured_rotation, human_psi


class RobotEvaluator:
    def __init__(self) -> None:
        from sew_mimic.common import ExactSewTarget, joint_limit_margin
        from sew_mimic.exact.residuals import robot_exact_sew_residuals
        from sew_mimic.kinematics import gen3_kinematics
        from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
        self.ExactSewTarget = ExactSewTarget
        self.joint_limit_margin = joint_limit_margin
        self.residuals = robot_exact_sew_residuals
        self.robot = gen3_kinematics()
        self.geometry = Gen3StereoSewGeometry.from_robot(self.robot)
        self.stereo = StereoSew(project_stereo_sew_reference())

    def fill_success(self, payload: dict[str, Any], index: int, desired_u_p: np.ndarray,
                     desired_u_R: np.ndarray, desired_p: np.ndarray, desired_R: np.ndarray,
                     desired_psi: float) -> None:
        q = payload["q"][index]
        target = self.ExactSewTarget(desired_p, desired_R, desired_psi)
        residual = self.residuals(q, target, self.robot, self.geometry, self.stereo)
        payload["actual_pinch_position_base"][index] = residual.actual_position
        payload["actual_pinch_rotation_base"][index] = residual.actual_rotation
        payload["actual_psi"][index] = np.nan if residual.actual_psi is None else residual.actual_psi
        payload["aligned_pinch_position_error_m"][index] = residual.position_error_m
        payload["aligned_pinch_orientation_error_rad"][index] = residual.orientation_error_rad
        payload["psi_error_rad"][index] = np.nan if residual.sew_error_rad is None else residual.sew_error_rad
        payload["joint_limit_margin_rad"][index] = self.joint_limit_margin(q, self.robot)
        actual_u_p = residual.actual_position
        actual_u_R = realized_virtual_tool_rotation(residual.actual_rotation)
        payload["actual_virtual_U_position_base"][index] = actual_u_p
        payload["actual_virtual_U_rotation_base"][index] = actual_u_R
        payload["virtual_utensil_position_error_m"][index] = np.linalg.norm(actual_u_p - desired_u_p)
        payload["virtual_utensil_orientation_error_rad"][index] = _rotation_error(actual_u_R, desired_u_R)


def _solve_stateful(desired_u_p: np.ndarray, desired_u_R: np.ndarray, psi: np.ndarray,
                    evaluator: RobotEvaluator, adapter_factory=ExactSewTrajectoryAdapter) -> dict[str, Any]:
    n = len(psi); payload = _empty_solver_payload(n)
    desired_p, desired_R = desired_u_p, virtual_pinch_rotation(desired_u_R)
    started = perf_counter(); results = adapter_factory().solve_trajectory(desired_p, desired_R, psi)
    elapsed = (perf_counter() - started) * 1000.
    if len(results) != n:
        raise RuntimeError("Exact-SEW trajectory result length mismatch")
    for index, result in enumerate(results):
        diagnostics = result.diagnostics; status = getattr(result.status, "value", str(result.status))
        payload["solver_status"][index] = status
        payload["solver_message"][index] = result.message or ""
        payload["solver_diagnostics_json"][index] = json.dumps(diagnostics.to_dict(), sort_keys=True, default=str)
        payload["branch_id"][index] = diagnostics.branch_id or ""
        payload["search_branch"][index] = str(diagnostics.metadata.get("search_branch", ""))
        payload["solve_time_ms"][index] = diagnostics.solve_time_ms if diagnostics.solve_time_ms is not None else elapsed / n
        if status != SUCCESS_EXACT or result.q is None:
            continue
        payload["q"][index] = np.asarray(result.q, float)
        evaluator.fill_success(payload, index, desired_u_p[index], desired_u_R[index],
                               desired_p[index], desired_R[index], psi[index])
    payload["evaluation_status"] = np.asarray(payload["solver_status"], object).copy()
    payload["strategy_psi"] = np.asarray(psi, float).copy()
    return payload


def _solve_robot_smooth(desired_u_p: np.ndarray, desired_u_R: np.ndarray, psi0: float,
                        initial: dict[str, Any], evaluator: RobotEvaluator,
                        candidate_solver=None, margin_fn=None) -> dict[str, Any]:
    if candidate_solver is None:
        candidate_solver, default_margin = _default_solver(); margin_fn = margin_fn or default_margin
    if margin_fn is None:
        raise ValueError("RobotSmooth margin function is required")
    first_status = str(initial["solver_status"][0])
    first_q = initial["q"][0].copy() if first_status == SUCCESS_EXACT else None
    desired_p, desired_R = desired_u_p, virtual_pinch_rotation(desired_u_R)
    smooth = robot_smooth_run(desired_p, desired_R, psi0, first_q, first_status,
                              candidate_solver, margin_fn)
    payload = _empty_solver_payload(len(desired_u_p))
    payload.update(smooth)
    if first_status == SUCCESS_EXACT:
        payload["branch_id"][0] = initial["branch_id"][0]
        payload["search_branch"][0] = initial["search_branch"][0]
        payload["solver_diagnostics_json"][0] = initial["solver_diagnostics_json"][0]
        payload["solver_message"][0] = initial["solver_message"][0]
        payload["solve_time_ms"][0] = initial["solve_time_ms"][0]
    for index in np.flatnonzero(np.asarray(payload["solver_status"]) == SUCCESS_EXACT):
        evaluator.fill_success(payload, int(index), desired_u_p[index], desired_u_R[index],
                               desired_p[index], desired_R[index], float(payload["strategy_psi"][index]))
    return payload


def _add_continuity(payload: dict[str, Any], time_s: np.ndarray) -> dict[str, Any]:
    n = len(time_s); frame = np.arange(n); run_id = np.zeros(n, int)
    success = np.asarray(payload["solver_status"]) == SUCCESS_EXACT
    labels = continuity_labels(payload["q"], success, frame, run_id)
    payload.update(labels)
    motion = clean_motion(payload["q"], time_s, frame, run_id, success, labels["continuity_violation"])
    payload.update({key: value for key, value in motion.items() if key != "continuous_segments"})
    payload["continuous_segment_starts"] = np.array([x[0] for x in motion["continuous_segments"]], int)
    payload["continuous_segment_ends"] = np.array([x[-1] for x in motion["continuous_segments"]], int)
    return payload


def verify_common_first_target(position: np.ndarray, rotation: np.ndarray, psi0: float,
                               payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not (np.isfinite(position[0]).all() and _proper_so3(rotation[:1])[0] and np.isfinite(psi0)):
        raise RuntimeError("invalid common first target")
    statuses = [str(payloads[name]["solver_status"][0]) for name in STRATEGIES]
    if len(set(statuses)) != 1:
        raise RuntimeError(f"first solver status disagreement: {statuses}")
    q_difference = 0.0
    if statuses[0] == SUCCESS_EXACT:
        q0 = payloads["B0"]["q"][0]
        for name in STRATEGIES[1:]:
            q_difference = max(q_difference, float(np.max(np.abs(payloads[name]["q"][0] - q0))))
            if not np.allclose(payloads[name]["q"][0], q0, atol=1e-10, rtol=0):
                raise RuntimeError(f"first q disagreement for {name}")
    return {"first_position_equal": True, "first_rotation_equal": True, "first_psi_equal": True,
            "first_solver_status_equal": True, "first_status": statuses[0],
            "max_first_q_abs_difference_rad": q_difference}


def _motion_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    success = np.asarray(payload["solver_status"]) == SUCCESS_EXACT
    violation = np.asarray(payload["continuity_violation"], bool)
    segment_lengths = np.asarray(payload["continuous_segment_ends"]) - np.asarray(payload["continuous_segment_starts"]) + 1
    def finite(key: str) -> np.ndarray:
        values = np.asarray(payload[key], float)
        return values[np.isfinite(values)]
    dq, velocity = finite("clean_wrapped_delta_q_rad"), finite("clean_joint_velocity_rad_s")
    acceleration, jerk = finite("clean_joint_acceleration_rad_s2"), finite("clean_joint_jerk_rad_s3")
    branch = np.asarray(payload["search_branch"]).astype(str)
    branch_changes = sum(bool(branch[i - 1] and branch[i] and branch[i - 1] != branch[i])
                         for i in range(1, len(branch)) if success[i - 1] and success[i] and not violation[i])
    return {
        "first_frame_success": bool(success[0]), "successful_frames": int(success.sum()),
        "frame_success_rate": float(success.mean()),
        "complete_trajectory_success": bool(success.all()), "continuity_violations": int(violation.sum()),
        "continuous_frame_percent": float((success & ~violation).sum() / max(1, success.sum())),
        "longest_continuous_section_frames": int(segment_lengths.max()) if len(segment_lengths) else 0,
        "complete_pipeline_success": bool(success.all() and not violation.any()),
        "branch_changes": int(branch_changes), "joint_travel_rad": float(np.abs(dq).sum()),
        "rms_joint_velocity_rad_s": float(np.sqrt(np.mean(velocity ** 2))) if len(velocity) else np.nan,
        "rms_joint_acceleration_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))) if len(acceleration) else np.nan,
        "rms_joint_jerk_rad_s3": float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else np.nan,
    }


def _robot_metric_row(record: dict[str, Any], generator: str, strategy: str,
                      payload: dict[str, Any]) -> dict[str, Any]:
    status = np.asarray(payload["solver_status"]).astype(str)
    evaluation = np.asarray(payload.get("evaluation_status", status)).astype(str)
    success = status == SUCCESS_EXACT
    row = {"record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
           "take": record["source_take"], "phase": record["phase"], "generator": generator,
           "strategy": strategy, **_motion_metrics(payload),
           "JOINT_LIMIT": int((status == "JOINT_LIMIT").sum()),
           "NO_VALID_BRANCH": int((status == "NO_VALID_BRANCH").sum()),
           "NO_FEASIBLE_PSI": int((evaluation == "NO_FEASIBLE_PSI").sum()),
           "INPUT_NOT_SOLVED": int((status == INPUT_NOT_SOLVED).sum()),
           "provisional_timing_diagnostic": True}
    for key in ("aligned_pinch_position_error_m", "aligned_pinch_orientation_error_rad", "psi_error_rad",
                "virtual_utensil_position_error_m", "virtual_utensil_orientation_error_rad"):
        values = np.asarray(payload[key], float)[success]
        values = values[np.isfinite(values)]
        row[f"{key}_max"] = float(values.max()) if len(values) else np.nan
        row[f"{key}_rms"] = float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan
    margin = np.asarray(payload["joint_limit_margin_rad"], float)[success]
    row["joint_limit_margin_min_rad"] = float(np.nanmin(margin)) if np.isfinite(margin).any() else np.nan
    return row


def _coordination_metrics(record: dict[str, Any], generator: str, generated_delta: np.ndarray,
                          generated_psi: np.ndarray, reference_delta: np.ndarray,
                          human_psi: np.ndarray, first_correction: float) -> dict[str, Any]:
    propagation = np.abs(wrap(generated_delta - reference_delta))
    human = np.abs(wrap(generated_psi - human_psi))
    return {
        "record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
        "take": record["source_take"], "phase": record["phase"], "generator": generator,
        "first_delta_correction_rad": first_correction,
        "generated_vs_measured_reference_delta_psi_rmse_rad": float(np.sqrt(np.mean(propagation ** 2))),
        "generated_vs_measured_reference_delta_psi_mae_rad": float(np.mean(propagation)),
        "generated_vs_measured_reference_delta_psi_p95_rad": float(np.percentile(propagation, 95)),
        "generated_vs_measured_reference_delta_psi_max_rad": float(propagation.max()),
        "end_to_end_coordination_deviation_rmse_rad": float(np.sqrt(np.mean(human ** 2))),
        "end_to_end_coordination_deviation_mae_rad": float(np.mean(human)),
        "coordination_success": bool(np.isfinite(generated_psi).all()),
    }


def _robot_deviation(record: dict[str, Any], generator: str, generated: dict[str, Any],
                     reference: dict[str, Any], evaluator: RobotEvaluator) -> dict[str, Any]:
    generated_success = np.asarray(generated["solver_status"]) == SUCCESS_EXACT
    reference_success = np.asarray(reference["solver_status"]) == SUCCESS_EXACT
    generated_violation = np.asarray(generated["continuity_violation"], bool)
    reference_violation = np.asarray(reference["continuity_violation"], bool)
    common = generated_success & reference_success
    q_difference = wrap(np.asarray(generated["q"])[common] - np.asarray(reference["q"])[common])
    q_rms = float(np.sqrt(np.mean(q_difference ** 2))) if len(q_difference) else np.nan
    elbow, plane = [], []
    for q_generated, q_reference in zip(np.asarray(generated["q"])[common], np.asarray(reference["q"])[common]):
        points_g = evaluator.geometry.sew_points(q_generated); points_r = evaluator.geometry.sew_points(q_reference)
        elbow.append(np.linalg.norm(points_g.elbow - points_r.elbow))
        normal_g = np.cross(points_g.wrist - points_g.shoulder, points_g.elbow - points_g.shoulder)
        normal_r = np.cross(points_r.wrist - points_r.shoulder, points_r.elbow - points_r.shoulder)
        plane.append(arm_plane_angle(normal_g, normal_r))
    union_violation = generated_violation | reference_violation
    frame, run = np.arange(101), np.zeros(101, int)
    motion_g = clean_motion(generated["q"], np.arange(101, dtype=float), frame, run, common, union_violation)
    motion_r = clean_motion(reference["q"], np.arange(101, dtype=float), frame, run, common, union_violation)
    travel_g = float(np.nansum(np.abs(motion_g["clean_wrapped_delta_q_rad"])))
    travel_r = float(np.nansum(np.abs(motion_r["clean_wrapped_delta_q_rad"])))
    return {
        "record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
        "take": record["source_take"], "phase": record["phase"], "generator": generator,
        "common_success_frames": int(common.sum()), "end_to_end_q_rms_deviation_rad": q_rms,
        "end_to_end_elbow_mean_deviation_m": float(np.mean(elbow)) if elbow else np.nan,
        "end_to_end_elbow_rms_deviation_m": float(np.sqrt(np.mean(np.square(elbow)))) if elbow else np.nan,
        "end_to_end_arm_plane_mean_deviation_rad": float(np.nanmean(plane)) if plane else np.nan,
        "generated_joint_travel_common_support_rad": travel_g,
        "measured_reference_joint_travel_common_support_rad": travel_r,
        "joint_travel_difference_rad": travel_g - travel_r,
        "interpretation": "end-to-end generated-vs-measured reference",
    }


def _save_result(path: Path, record: dict[str, Any], generator: str, g0: dict[str, np.ndarray],
                 measured_position_B: np.ndarray, measured_rotation_B: np.ndarray, psi0: float,
                 features: np.ndarray, raw_delta: np.ndarray, anchored_delta: np.ndarray,
                 payloads: dict[str, dict[str, Any]], desired_u_p: np.ndarray, desired_u_R: np.ndarray,
                 first: dict[str, Any], fold: FoldModel, fingerprint: str) -> None:
    arrays: dict[str, Any] = {
        "schema_version": np.array("generator-g1-result-v1"), "run_fingerprint": np.array(fingerprint),
        "record_id": np.array(record["record_id"]), "parent_bite_id": np.array(record["parent_bite_id"]),
        "take": np.array(record["source_take"]), "phase": np.array(record["phase"]),
        "generator": np.array(generator), "outer_fold": np.array(record["source_take"]),
        "normalized_phase": GRID, "time_s": g0["time_s"],
        "generated_U_position_B": g0["generated_tool_position_B"],
        "generated_U_rotation_B": g0["generated_tool_orientation_B"],
        "generated_U_position_base": desired_u_p, "generated_U_rotation_base": desired_u_R,
        "measured_U_position_B": measured_position_B, "measured_U_rotation_B": measured_rotation_B,
        "plate_position_B": g0["plate_position_B"], "mouth_proxy_position_B": g0["mouth_proxy_position_B"],
        "mouth_proxy_is_independently_measured": np.array(False), "deployment_context_validated": np.array(False),
        "psi0": np.array(psi0), "initial_redundancy_source": np.array("measured_human_demo"),
        "deployment_initial_redundancy_selection": np.array("not yet solved"),
        "stronglocal_features": features, "raw_delta_psi": raw_delta,
        "anchored_delta_psi": anchored_delta, "stronglocal_psi": psi0 + anchored_delta,
        "first_delta_correction": np.array(raw_delta[0]), "fold_alpha": np.array(fold.alpha),
        "fold_training_takes": np.array(fold.training_takes), "fold_parity_max_abs_error": np.array(fold.parity_max_abs_error),
        **{key: np.array(value) for key, value in first.items()},
    }
    for strategy, payload in payloads.items():
        for key, value in payload.items():
            if key == "continuous_segments": continue
            arrays[f"{strategy}_{key}"] = _storable_array(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_reference(path: Path, record: dict[str, Any], time_s: np.ndarray,
                    measured_position_B: np.ndarray, measured_rotation_B: np.ndarray,
                    desired_u_p: np.ndarray, desired_u_R: np.ndarray, plate_B: np.ndarray,
                    psi0: float, features: np.ndarray, raw_delta: np.ndarray, anchored_delta: np.ndarray,
                    payload: dict[str, Any], fold: FoldModel, fingerprint: str) -> None:
    arrays: dict[str, Any] = {
        "schema_version": np.array("generator-g1-measured-reference-v1"), "run_fingerprint": np.array(fingerprint),
        "record_id": np.array(record["record_id"]), "parent_bite_id": np.array(record["parent_bite_id"]),
        "take": np.array(record["source_take"]), "phase": np.array(record["phase"]),
        "generator": np.array("MeasuredUReference"), "outer_fold": np.array(record["source_take"]),
        "normalized_phase": GRID, "time_s": time_s, "measured_U_position_B": measured_position_B,
        "measured_U_rotation_B": measured_rotation_B, "measured_U_position_base": desired_u_p,
        "measured_U_rotation_base": desired_u_R, "plate_position_B": plate_B, "psi0": np.array(psi0),
        "stronglocal_features": features, "raw_delta_psi": raw_delta,
        "anchored_delta_psi": anchored_delta, "stronglocal_psi": psi0 + anchored_delta,
        "first_delta_correction": np.array(raw_delta[0]), "fold_alpha": np.array(fold.alpha),
        "fold_training_takes": np.array(fold.training_takes), "fold_parity_max_abs_error": np.array(fold.parity_max_abs_error),
    }
    arrays.update({f"StrongLocal_{key}": _storable_array(value) for key, value in payload.items()
                   if key != "continuous_segments"})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_cached(path: Path, fingerprint: str) -> dict[str, np.ndarray] | None:
    if not path.is_file(): return None
    with np.load(path, allow_pickle=False) as z:
        if str(z.get("run_fingerprint", "")) != fingerprint: return None
        return {key: z[key].copy() for key in z.files}


def _payload_from_arrays(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    start = f"{prefix}_"
    return {key[len(start):]: value for key, value in arrays.items() if key.startswith(start)}


def _pipeline_rows_from_cache(record: dict[str, Any], generator: str, arrays: dict[str, np.ndarray],
                              reference: dict[str, np.ndarray], human_psi: np.ndarray,
                              g0_metrics: pd.DataFrame, evaluator: RobotEvaluator) -> tuple[list[dict], dict, dict]:
    payloads = {strategy: _payload_from_arrays(arrays, strategy) for strategy in STRATEGIES}
    ref_payload = _payload_from_arrays(reference, "StrongLocal")
    robot_rows = [_robot_metric_row(record, generator, strategy, payloads[strategy]) for strategy in STRATEGIES]
    coordination = _coordination_metrics(record, generator, arrays["anchored_delta_psi"], arrays["stronglocal_psi"],
                                         reference["anchored_delta_psi"], human_psi,
                                         float(arrays["first_delta_correction"]))
    deviation = _robot_deviation(record, generator, payloads["StrongLocal"], ref_payload, evaluator)
    g0 = g0_metrics.loc[(g0_metrics.record_id.astype(str) == record["record_id"])
                        & (g0_metrics.generator.astype(str) == generator)].iloc[0]
    main = next(row for row in robot_rows if row["strategy"] == "StrongLocal")
    pipeline = {"record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
                "take": record["source_take"], "phase": record["phase"], "generator": generator,
                "tool_rms_position_error_m": float(g0.rms_position_error_m),
                "tool_rms_orientation_error_rad": float(g0.rms_orientation_error_rad),
                "tool_final_position_error_m": float(g0.final_position_error_m),
                "tool_path_length_absolute_error_m": float(g0.path_length_absolute_error_m),
                **{key: coordination[key] for key in coordination if key not in ("record_id", "parent_bite_id", "take", "phase", "generator")},
                **{key: deviation[key] for key in deviation if key not in ("record_id", "parent_bite_id", "take", "phase", "generator", "interpretation")},
                "stronglocal_frame_success_rate": main["frame_success_rate"],
                "stronglocal_complete_trajectory_success": main["complete_trajectory_success"],
                "stronglocal_continuity_violations": main["continuity_violations"],
                "stronglocal_complete_pipeline_success": main["complete_pipeline_success"]}
    return robot_rows, coordination, pipeline


def _paired_bootstrap(pipeline: pd.DataFrame, seed: int, resamples: int) -> pd.DataFrame:
    metrics = (
        "tool_rms_position_error_m", "tool_rms_orientation_error_rad",
        "generated_vs_measured_reference_delta_psi_mae_rad", "stronglocal_frame_success_rate",
        "stronglocal_complete_trajectory_success", "stronglocal_continuity_violations",
        "end_to_end_q_rms_deviation_rad", "end_to_end_elbow_mean_deviation_m",
        "generated_joint_travel_common_support_rad",
    )
    rng = np.random.default_rng(seed); rows = []
    for group, frame in [("overall", pipeline), *list(pipeline.groupby("phase", sort=True))]:
        for metric in metrics:
            bite = frame.groupby(["generator", "parent_bite_id"])[metric].mean().unstack("generator")
            bite = bite.dropna(subset=list(GENERATORS))
            difference = bite["contextual_promp"].to_numpy(float) - bite["retrieval"].to_numpy(float)
            draws = rng.integers(0, len(difference), (resamples, len(difference)))
            samples = difference[draws].mean(axis=1)
            rows.append({"group": str(group), "metric": metric,
                         "comparison": "contextual_promp_minus_retrieval",
                         "mean_difference": float(difference.mean()),
                         "ci95_low": float(np.percentile(samples, 2.5)),
                         "ci95_high": float(np.percentile(samples, 97.5)),
                         "n_parent_bites": len(difference), "seed": seed, "resamples": resamples,
                         "bootstrap_unit": "parent_bite"})
    return pd.DataFrame(rows)


def _aggregate(robot: pd.DataFrame, pipeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (take, generator, strategy), group in robot.groupby(["take", "generator", "strategy"], sort=True):
        p = pipeline.loc[(pipeline.take == take) & (pipeline.generator == generator)]
        rows.append({"take": take, "generator": generator, "strategy": strategy,
                     "records": len(group), "parent_bites": group.parent_bite_id.nunique(),
                     "frame_success_rate": float(group.frame_success_rate.mean()),
                     "complete_trajectory_success_rate": float(group.complete_trajectory_success.mean()),
                     "complete_pipeline_success_rate": float(group.complete_pipeline_success.mean()),
                     "continuity_violations": int(group.continuity_violations.sum()),
                     "tool_rms_position_error_m": float(p.tool_rms_position_error_m.mean()) if len(p) else np.nan,
                     "delta_psi_propagation_mae_rad": float(p.generated_vs_measured_reference_delta_psi_mae_rad.mean()) if len(p) else np.nan,
                     "end_to_end_q_rms_deviation_rad": float(p.end_to_end_q_rms_deviation_rad.mean()) if len(p) else np.nan})
    return pd.DataFrame(rows)


def _representatives(pipeline: pd.DataFrame, robot: pd.DataFrame) -> dict[str, Any]:
    values = pipeline.copy()
    values["robot_deviation"] = values.end_to_end_q_rms_deviation_rad.fillna(np.inf)
    def pack(row: pd.Series) -> dict[str, Any]:
        return {key: row[key].item() if isinstance(row[key], np.generic) else row[key]
                for key in ("record_id", "parent_bite_id", "take", "phase", "generator",
                            "tool_rms_position_error_m", "generated_vs_measured_reference_delta_psi_mae_rad",
                            "end_to_end_q_rms_deviation_rad", "stronglocal_complete_pipeline_success")}
    clean_promp = values.loc[(values.generator == "contextual_promp") & values.stronglocal_complete_pipeline_success]
    clean_retrieval = values.loc[(values.generator == "retrieval") & values.stronglocal_complete_pipeline_success]
    result: dict[str, Any] = {}
    if len(clean_promp): result["clean_promp"] = pack(clean_promp.sort_values(["tool_rms_position_error_m", "record_id"]).iloc[0])
    if len(clean_retrieval): result["clean_retrieval"] = pack(clean_retrieval.sort_values(["tool_rms_position_error_m", "record_id"]).iloc[0])
    pivot = values.pivot(index=["record_id", "parent_bite_id", "take", "phase"], columns="generator",
                         values=["tool_rms_position_error_m", "end_to_end_q_rms_deviation_rad"]).reset_index()
    pivot.columns = ["_".join(str(x) for x in column if str(x)) for column in pivot.columns]
    pivot["advantage"] = ((pivot["tool_rms_position_error_m_retrieval"] - pivot["tool_rms_position_error_m_contextual_promp"])
                          + (pivot["end_to_end_q_rms_deviation_rad_retrieval"] - pivot["end_to_end_q_rms_deviation_rad_contextual_promp"]))
    if len(pivot):
        chosen = pivot.sort_values(["advantage", "record_id"], ascending=[False, True]).iloc[0]
        row = values.loc[(values.record_id == chosen["record_id"]) & (values.generator == "contextual_promp")].iloc[0]
        result["promp_advantage"] = pack(row)
    propagation = values.assign(score=values.tool_rms_position_error_m + values.generated_vs_measured_reference_delta_psi_mae_rad + values.robot_deviation)
    if len(propagation): result["generator_error_propagation"] = pack(propagation.sort_values(["score", "record_id"], ascending=[False, True]).iloc[0])
    failures = values.loc[~values.stronglocal_complete_pipeline_success]
    if len(failures): result["robot_failure_despite_valid_generated_u"] = pack(failures.sort_values(["tool_rms_position_error_m", "record_id"]).iloc[0])
    result["selection"] = "deterministic from OOF tool, coordination, and stored robot metrics; never visual preference"
    return result


def _plots(output: Path, robot: pd.DataFrame, pipeline: pd.DataFrame) -> None:
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    main = robot.loc[robot.strategy == "StrongLocal"]
    for filename, metric, ylabel in (
        ("pipeline_success.png", "complete_pipeline_success", "complete pipeline success rate"),
        ("coordination_propagation.png", "generated_vs_measured_reference_delta_psi_mae_rad", "delta-psi propagation MAE (rad)"),
        ("robot_deviation.png", "end_to_end_q_rms_deviation_rad", "end-to-end q RMS deviation (rad)"),
    ):
        frame = main if metric in main else pipeline
        values = frame.groupby("generator")[metric].mean().reindex(GENERATORS)
        fig, ax = plt.subplots(figsize=(5.5, 4)); ax.bar(values.index, values.values); ax.set(ylabel=ylabel)
        fig.tight_layout(); fig.savefig(plots / filename, dpi=160); plt.close(fig)
    values = pipeline.groupby("generator")[["tool_rms_position_error_m", "end_to_end_q_rms_deviation_rad"]].mean().reindex(GENERATORS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4)); values.tool_rms_position_error_m.plot.bar(ax=axes[0], title="generated vs measured U")
    values.end_to_end_q_rms_deviation_rad.plot.bar(ax=axes[1], title="generated vs measured robot reference")
    fig.tight_layout(); fig.savefig(plots / "generated_vs_measured_u.png", dpi=160); plt.close(fig)
    phase = pipeline.groupby(["phase", "generator"]).agg(tool_error=("tool_rms_position_error_m", "mean"), psi_error=("generated_vs_measured_reference_delta_psi_mae_rad", "mean"), q_error=("end_to_end_q_rms_deviation_rad", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("tool_error", "psi_error", "q_error"), ("tool", "coordination", "robot")):
        phase.pivot(index="phase", columns="generator", values=metric).plot.bar(ax=ax, title=title)
    fig.tight_layout(); fig.savefig(plots / "transfer_withdrawal.png", dpi=160); plt.close(fig)


def process_generator_g1(phase1_dir: Path, phase32_dir: Path, g0_dir: Path, robot_r0_dir: Path,
                         output_dir: Path, cfg: GeneratorG1Config, trace: dict | None = None) -> dict[str, Any]:
    phase1_dir, phase32_dir, g0_dir, robot_r0_dir, output_dir = map(Path, (phase1_dir, phase32_dir, g0_dir, robot_r0_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (output_dir / "fold_models", output_dir / "plots"):
        directory.mkdir(exist_ok=True)
    source_manifest_path = phase1_dir / "manifest.json"; source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema_version") != "phase1-v1.5": raise ValueError("G1 requires Phase 1.5")
    source_hash = _sha256(source_manifest_path)
    g0_manifest_path = g0_dir / "manifest.json"; g0_manifest = json.loads(g0_manifest_path.read_text())
    if g0_manifest.get("schema_version") != "generator-g0-v1" or g0_manifest.get("source_phase1_manifest_sha256") != source_hash:
        raise ValueError("G1 requires frozen G0 artifacts from the same Phase 1.5 manifest")
    r0_manifest_path = robot_r0_dir / "manifest.json"; r0_manifest = json.loads(r0_manifest_path.read_text())
    if r0_manifest.get("schema_version") != "robot-r0-v1" or r0_manifest.get("phase1_manifest_sha256") != source_hash:
        raise ValueError("G1 requires frozen Robot R0 geometry from the same Phase 1.5 manifest")

    models, parity = reconstruct_fold_models(phase1_dir, phase32_dir, output_dir / "fold_models", cfg.parity_tolerance, trace)
    parity.to_csv(output_dir / "fold_model_parity.csv", index=False)
    eligibility, eligible = audit_eligibility(phase1_dir, g0_dir, source_manifest, source_hash)
    eligibility.to_csv(output_dir / "eligibility.csv", index=False)
    g0_metrics = pd.read_csv(g0_dir / "record_metrics.csv")
    records = {entry["record_id"]: entry for entry in source_manifest["records"]}

    fingerprint = hashlib.sha256(json.dumps({"result_cache_version": RESULT_CACHE_VERSION,
        "config": asdict(cfg), "phase1": source_hash,
        "g0": _sha256(g0_manifest_path), "phase32": _sha256(phase32_dir / "summary.json"),
        "robot_r0": _sha256(r0_manifest_path)}, sort_keys=True).encode()).hexdigest()
    cache = output_dir.with_name(output_dir.name + ".cache")
    for name in (*GENERATORS, REFERENCE): (cache / "results" / name).mkdir(parents=True, exist_ok=True)

    evaluator = RobotEvaluator(); candidate_solver, margin_fn = _default_solver()
    completed = 0; total = len(eligible)
    for record_id in sorted(eligible):
        source, record = eligible[record_id], records[record_id]
        fold = models[record["source_take"]]
        paths = {generator: cache / "results" / generator / f"{record_id}.npz" for generator in GENERATORS}
        reference_path = cache / "results" / REFERENCE / f"{record_id}.npz"
        cached = {generator: _load_cached(path, fingerprint) for generator, path in paths.items()}
        cached_reference = _load_cached(reference_path, fingerprint)
        if cached_reference is None or any(value is None for value in cached.values()):
            g0_by_generator = {generator: validate_g0_artifact(g0_dir / "generated" / generator / f"{record_id}.npz",
                record_id, record["source_take"], record["phase"], source_hash, generator) for generator in GENERATORS}
            if not np.array_equal(g0_by_generator["retrieval"]["time_s"], g0_by_generator["contextual_promp"]["time_s"]):
                raise RuntimeError(f"{record_id}: G0 generators do not share provisional timing")
            time_s = g0_by_generator["retrieval"]["time_s"]
            measured_p, measured_R, human_psi = measured_reference(source, time_s)
            mounting = r0_manifest["mountings"][record["source_take"]]
            R_B_base, p_B_base = np.asarray(mounting["R_B_from_base"], float), np.asarray(mounting["p_B_of_base"], float)
            plate_B = g0_by_generator["retrieval"]["plate_position_B"]
            reference_features = stronglocal_features(measured_p, measured_R, time_s, GRID, record["phase"], plate_B)
            reference_raw = fold.predict_delta(reference_features); reference_anchored = anchor_delta(reference_raw)
            reference_u_p = base_transform(measured_p, R_B_base, p_B_base)
            reference_u_R = base_rotations(measured_R, R_B_base)
            reference_payload = _add_continuity(_solve_stateful(reference_u_p, reference_u_R,
                source["psi0"] + reference_anchored, evaluator), time_s)
            _save_reference(reference_path, record, time_s, measured_p, measured_R, reference_u_p, reference_u_R,
                            plate_B, source["psi0"], reference_features, reference_raw, reference_anchored,
                            reference_payload, fold, fingerprint)
            for generator, g0 in g0_by_generator.items():
                features = stronglocal_features(g0["generated_tool_position_B"], g0["generated_tool_orientation_B"],
                                                g0["time_s"], g0["normalized_phase"], record["phase"],
                                                g0["plate_position_B"], bool(g0["plate_position_valid"]))
                raw = fold.predict_delta(features); anchored = anchor_delta(raw)
                desired_u_p = base_transform(g0["generated_tool_position_B"], R_B_base, p_B_base)
                desired_u_R = base_rotations(g0["generated_tool_orientation_B"], R_B_base)
                b0 = _add_continuity(_solve_stateful(desired_u_p, desired_u_R, np.full(101, source["psi0"]), evaluator), g0["time_s"])
                strong = _add_continuity(_solve_stateful(desired_u_p, desired_u_R, source["psi0"] + anchored, evaluator), g0["time_s"])
                smooth = _add_continuity(_solve_robot_smooth(desired_u_p, desired_u_R, source["psi0"], b0,
                                                evaluator, candidate_solver, margin_fn), g0["time_s"])
                b0.update({"raw_predicted_delta_psi": np.zeros(101), "anchored_delta_psi": np.zeros(101)})
                strong.update({"raw_predicted_delta_psi": raw, "anchored_delta_psi": anchored})
                smooth.update({"raw_predicted_delta_psi": np.full(101, np.nan), "anchored_delta_psi": np.full(101, np.nan)})
                payloads = {"B0": b0, "StrongLocal": strong, "RobotSmooth": smooth}
                first = verify_common_first_target(desired_u_p, desired_u_R, source["psi0"], payloads)
                _save_result(paths[generator], record, generator, g0, measured_p, measured_R, source["psi0"],
                             features, raw, anchored, payloads, desired_u_p, desired_u_R, first, fold, fingerprint)
        completed += 1
        if completed == 1 or completed % 5 == 0 or completed == total:
            print(f"G1 progress {completed}/{total} eligible records", flush=True)

    results_dir = output_dir / "results"
    if results_dir.exists(): shutil.rmtree(results_dir)
    shutil.copytree(cache / "results", results_dir)

    robot_rows, coordination_rows, pipeline_rows = [], [], []
    for record_id in sorted(eligible):
        source, record = eligible[record_id], records[record_id]
        reference = _load_cached(results_dir / REFERENCE / f"{record_id}.npz", fingerprint)
        assert reference is not None
        _, _, human_psi = measured_reference(source, reference["time_s"])
        for generator in GENERATORS:
            arrays = _load_cached(results_dir / generator / f"{record_id}.npz", fingerprint)
            assert arrays is not None
            rr, cr, pr = _pipeline_rows_from_cache(record, generator, arrays, reference, human_psi, g0_metrics, evaluator)
            robot_rows.extend(rr); coordination_rows.append(cr); pipeline_rows.append(pr)
    robot = pd.DataFrame(robot_rows); coordination = pd.DataFrame(coordination_rows); pipeline = pd.DataFrame(pipeline_rows)
    residual_gate = {
        "aligned_pinch_position_error_m_max": cfg.task_position_tolerance_m,
        "virtual_utensil_position_error_m_max": cfg.task_position_tolerance_m,
        "aligned_pinch_orientation_error_rad_max": cfg.task_orientation_tolerance_rad,
        "virtual_utensil_orientation_error_rad_max": cfg.task_orientation_tolerance_rad,
        "psi_error_rad_max": cfg.task_psi_tolerance_rad,
    }
    for column, tolerance in residual_gate.items():
        maximum = pd.to_numeric(robot[column], errors="coerce").max()
        if np.isfinite(maximum) and maximum > tolerance:
            raise RuntimeError(f"authoritative Exact-SEW residual gate failed: {column}={maximum} > {tolerance}")
    robot.to_csv(output_dir / "robot_metrics.csv", index=False)
    coordination.to_csv(output_dir / "coordination_metrics.csv", index=False)
    pipeline.to_csv(output_dir / "pipeline_metrics.csv", index=False)
    bootstrap = _paired_bootstrap(pipeline, cfg.bootstrap_seed, cfg.bootstrap_resamples)
    bootstrap.to_csv(output_dir / "pairwise_bootstrap.csv", index=False)
    per_take = _aggregate(robot, pipeline); per_take.to_csv(output_dir / "per_take_metrics.csv", index=False)
    examples = _representatives(pipeline, robot)
    (output_dir / "representative_examples.json").write_text(json.dumps(_json_safe(examples), indent=2, sort_keys=True))
    _plots(output_dir, robot, pipeline)

    first_corrections = coordination.groupby("generator").first_delta_correction_rad.agg(["mean", "median", "min", "max"]).to_dict("index")
    robot_aggregate = robot.groupby(["generator", "strategy", "phase"], as_index=False).agg(
        records=("record_id", "size"), first_frame_success_rate=("first_frame_success", "mean"),
        frame_success_rate=("frame_success_rate", "mean"), complete_trajectory_success_rate=("complete_trajectory_success", "mean"),
        complete_pipeline_success_rate=("complete_pipeline_success", "mean"), continuity_violations=("continuity_violations", "sum"),
        JOINT_LIMIT=("JOINT_LIMIT", "sum"), NO_VALID_BRANCH=("NO_VALID_BRANCH", "sum"), NO_FEASIBLE_PSI=("NO_FEASIBLE_PSI", "sum"),
        joint_travel_rad=("joint_travel_rad", "mean"), rms_joint_velocity_rad_s=("rms_joint_velocity_rad_s", "mean"),
        rms_joint_acceleration_rad_s2=("rms_joint_acceleration_rad_s2", "mean"), rms_joint_jerk_rad_s3=("rms_joint_jerk_rad_s3", "mean"))
    residual_columns = [column for column in robot.columns if column.endswith("_max") and any(token in column for token in ("error_m", "error_rad"))]
    residuals = {column: float(pd.to_numeric(robot[column], errors="coerce").max()) for column in residual_columns}
    summary = {
        "schema_version": "generator-g1-v1",
        "eligibility": {"source_records": len(eligibility), "eligible_records": len(eligible),
                        "excluded_records": int((eligibility.eligibility_status != "G1_ELIGIBLE").sum()),
                        "status_counts": eligibility.eligibility_status.value_counts().to_dict()},
        "fold_model_parity": {"all_passed": bool(parity.parity_passed.all()),
                              "maximum_abs_delta_prediction_error": float(parity.max_abs_delta_prediction_error.max()),
                              "tolerance": cfg.parity_tolerance, "rows": parity.to_dict("records")},
        "stronglocal": {"feature_contract_version": "strong-local-l3-v1", "feature_order": list(STRONG_LOCAL_ORDER),
                        "fold_specific_primary_evaluation": True, "full_data_deployment_model_used": False,
                        "first_delta_correction_by_generator": first_corrections},
        "robot_feasibility": robot_aggregate.to_dict("records"),
        "task_residual_maxima": residuals,
        "coordination": coordination.groupby(["generator", "phase"]).mean(numeric_only=True).reset_index().to_dict("records"),
        "end_to_end_robot_deviation": pipeline.groupby(["generator", "phase"])[["end_to_end_q_rms_deviation_rad", "end_to_end_elbow_mean_deviation_m", "end_to_end_arm_plane_mean_deviation_rad", "joint_travel_difference_rad"]].mean().reset_index().to_dict("records"),
        "paired_parent_bite_bootstrap": bootstrap.to_dict("records"),
        "per_take": per_take.to_dict("records"), "representative_examples": examples,
        "initial_redundancy": {"initial_redundancy_source": "measured_human_demo",
                               "deployment_initial_redundancy_selection": "not yet solved", "optimization_performed": False},
        "timing": {"source": "G0 training-phase median duration", "provisional_timing_diagnostic": True,
                   "physical_limits_validated": False, "retimed": False},
        "semantic_boundary": {"mouth_proxy_is_independently_measured": False,
                              "deployment_context_validated": False, "physical_feeding_success_claimed": False,
                              "physical_safety_claimed": False},
        "scope": {"physical_calibration_added": False, "collision_added": False, "retiming_added": False,
                  "real_robot_execution_added": False, "new_learning_model_added": False,
                  "new_trajectory_generator_added": False},
        "provenance": {"phase1_manifest_sha256": source_hash, "generator_g0_manifest_sha256": _sha256(g0_manifest_path),
                       "phase32_summary_sha256": _sha256(phase32_dir / "summary.json"),
                       "robot_r0_manifest_sha256": _sha256(r0_manifest_path), "run_fingerprint": fingerprint,
                       "code_revision": _revision(Path(__file__).resolve().parents[2]),
                       "code_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True).stdout.strip()),
                       "config": asdict(cfg)},
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    manifest = {"schema_version": "generator-g1-v1", "summary": "summary.json", "eligibility": "eligibility.csv",
                "fold_model_parity": "fold_model_parity.csv", "pipeline_metrics": "pipeline_metrics.csv",
                "coordination_metrics": "coordination_metrics.csv", "robot_metrics": "robot_metrics.csv",
                "pairwise_bootstrap": "pairwise_bootstrap.csv", "per_take_metrics": "per_take_metrics.csv",
                "fold_models": [f"fold_models/heldout_{take}" for take in sorted(models)],
                "result_counts": {**{generator: len(eligible) for generator in GENERATORS}, REFERENCE: len(eligible)},
                "representative_examples": "representative_examples.json",
                "plots": [f"plots/{name}" for name in ("pipeline_success.png", "generated_vs_measured_u.png", "coordination_propagation.png", "robot_deviation.png", "transfer_withdrawal.png")],
                "mouth_proxy_is_independently_measured": False, "deployment_context_validated": False,
                "run_fingerprint": fingerprint}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(cache)
    return summary
