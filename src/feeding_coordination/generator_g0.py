"""G0: held-out-session complete fork SE(3) trajectory generation.

The only data source accepted here is the frozen Phase 1.5 artifact.  In
particular, the upstream ``mouth_target_position`` field is deliberately
renamed to ``mouth_proxy`` and is never represented as an independent
measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from sklearn.linear_model import Ridge

from .phase1 import _revision


GENERATORS = ("retrieval", "contextual_promp")
PHASES = ("transfer", "withdrawal")
GRID = np.linspace(0.0, 1.0, 101)
STRONG_LOCAL_FEATURE_ORDER = (
    "relative_tool_position_x", "relative_tool_position_y", "relative_tool_position_z",
    "relative_tool_rotation_x", "relative_tool_rotation_y", "relative_tool_rotation_z",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "normalized_phase_s", "phase_transfer", "phase_withdrawal",
    "plate_relative_x", "plate_relative_y", "plate_relative_z", "plate_valid_mask",
)


@dataclass(frozen=True)
class GeneratorG0Config:
    phase1_output_path: str = "outputs/phase1"
    output_path: str = "outputs/generator_g0"
    basis_functions: int = 12
    alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    bootstrap_seed: int = 20260915
    bootstrap_resamples: int = 2000
    generated_samples: int = 101

    @classmethod
    def load(cls, path: Path) -> "GeneratorG0Config":
        raw = json.loads(Path(path).read_text())
        raw["alphas"] = tuple(raw.get("alphas", cls.alphas))
        cfg = cls(**raw)
        if cfg.basis_functions != 12:
            raise ValueError("G0 freezes the Contextual ProMP to 12 RBF basis functions")
        if cfg.generated_samples != 101:
            raise ValueError("G0 freezes generated trajectories to 101 samples")
        if cfg.bootstrap_seed != 20260915 or cfg.bootstrap_resamples != 2000:
            raise ValueError("G0 bootstrap is frozen to seed 20260915 and 2000 resamples")
        if not cfg.alphas or any(a <= 0 for a in cfg.alphas):
            raise ValueError("all ridge alphas must be positive")
        return cfg


@dataclass
class G0Record:
    record_id: str
    parent_bite_id: str
    take: str
    phase: str
    p0: np.ndarray
    R0: np.ndarray
    plate_B: np.ndarray
    mouth_proxy_B: np.ndarray
    context: np.ndarray
    duration_s: float
    s_observed: np.ndarray
    y_observed: np.ndarray
    relative_position_grid: np.ndarray
    relative_orientation_grid: np.ndarray
    gt_position_B: np.ndarray
    gt_orientation_B: np.ndarray
    audit: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_pose(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    return np.isfinite(position).all(axis=1) & np.isfinite(orientation).all(axis=(1, 2))


def _proper_so3(matrices: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    matrices = np.asarray(matrices, float)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    orthogonal = np.linalg.norm(np.einsum("nji,njk->nik", matrices, matrices) - np.eye(3), axis=(1, 2)) <= atol
    determinant = np.abs(np.linalg.det(matrices) - 1.0) <= atol
    return finite & orthogonal & determinant


def _resample_position(s: np.ndarray, values: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    s = np.asarray(s, float)
    values = np.asarray(values, float)
    unique, index = np.unique(s, return_index=True)
    if len(unique) < 2:
        raise ValueError("trajectory needs at least two distinct normalized-phase samples")
    return np.column_stack([np.interp(grid, unique, values[index, j]) for j in range(3)])


def slerp_rotations(s: np.ndarray, matrices: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """Interpolate rotation matrices on SO(3); raw Euler interpolation is forbidden."""
    s = np.asarray(s, float)
    unique, index = np.unique(s, return_index=True)
    if len(unique) < 2:
        raise ValueError("trajectory needs at least two distinct rotations")
    clipped = np.clip(np.asarray(grid, float), unique[0], unique[-1])
    return Slerp(unique, Rotation.from_matrix(np.asarray(matrices, float)[index]))(clipped).as_matrix()


def rbf_basis(s: np.ndarray, n_basis: int = 12) -> np.ndarray:
    """Normalized Gaussian basis convention used by the G0 ProMP."""
    if n_basis != 12:
        raise ValueError("G0 requires exactly 12 RBF basis functions")
    s = np.asarray(s, float).reshape(-1, 1)
    centers = np.linspace(0.0, 1.0, n_basis)
    spacing = centers[1] - centers[0]
    phi = np.exp(-0.5 * ((s - centers) / spacing) ** 2)
    return phi / phi.sum(axis=1, keepdims=True)


def _robust_position(values: np.ndarray, valid: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool) & np.isfinite(values).all(axis=1)
    if not valid.any():
        raise ValueError(f"missing required {name} context")
    return np.median(values[valid], axis=0)


def _load_record(phase1_dir: Path, entry: dict) -> G0Record:
    with np.load(phase1_dir / entry["file"], allow_pickle=False) as z:
        position = np.asarray(z["tool_position"], float)
        orientation = np.asarray(z["tool_orientation"], float)
        tool_valid = np.asarray(z["tool_pose_valid"], bool) & _finite_pose(position, orientation)
        if tool_valid.sum() < 2:
            raise ValueError("fewer than two valid tracked fork poses")
        valid_index = np.flatnonzero(tool_valid)
        first, final = valid_index[0], valid_index[-1]
        p0, R0 = position[first].copy(), orientation[first].copy()
        if not _proper_so3(R0[None])[0]:
            raise ValueError("initial fork orientation is not proper SO(3)")

        plate = np.asarray(z["plate_position"], float)
        plate_valid = np.asarray(z["plate_position_valid"], bool) & np.isfinite(plate).all(axis=1)
        mouth = np.asarray(z["mouth_target_position"], float)  # upstream name; semantic alias below
        mouth_finite = np.isfinite(mouth).all(axis=1)
        plate_B = _robust_position(plate, plate_valid, "plate")
        mouth_proxy_B = _robust_position(mouth, mouth_finite, "mouth_proxy")

        time = np.asarray(z["time"], float)
        if len(time) < 2 or not np.isfinite(time).all() or time[-1] <= time[0]:
            raise ValueError("invalid segment time")
        duration = float(time[-1] - time[0])
        s_valid = (time[tool_valid] - time[0]) / duration
        relative_position = (R0.T @ (position[tool_valid] - p0).T).T
        relative_orientation = np.einsum("ij,njk->nik", R0.T, orientation[tool_valid])
        relative_rotvec = Rotation.from_matrix(relative_orientation).as_rotvec()
        y_observed = np.column_stack([relative_position, relative_rotvec])

        relative_position_grid = _resample_position(s_valid, relative_position)
        relative_orientation_grid = slerp_rotations(s_valid, relative_orientation)
        # The request pose is authoritative at normalized phase zero.
        relative_position_grid[0] = 0.0
        relative_orientation_grid[0] = np.eye(3)
        gt_position_B = p0 + (R0 @ relative_position_grid.T).T
        gt_orientation_B = np.einsum("ij,njk->nik", R0, relative_orientation_grid)

        mouth_offsets = mouth[mouth_finite] - mouth_proxy_B
        mouth_max_deviation = float(np.max(np.linalg.norm(mouth_offsets, axis=1)))
        audit = {
            "record_id": entry["record_id"], "parent_bite_id": entry["parent_bite_id"],
            "take": entry["source_take"], "phase": entry["phase"], "usable": True,
            "exclusion_reason": "", "initial_tool_position_B_x": p0[0],
            "initial_tool_position_B_y": p0[1], "initial_tool_position_B_z": p0[2],
            "initial_tool_rotation_rotvec_x": Rotation.from_matrix(R0).as_rotvec()[0],
            "initial_tool_rotation_rotvec_y": Rotation.from_matrix(R0).as_rotvec()[1],
            "initial_tool_rotation_rotvec_z": Rotation.from_matrix(R0).as_rotvec()[2],
            **{f"initial_tool_orientation_B_r{row}c{column}": R0[row, column]
               for row in range(3) for column in range(3)},
            "initial_valid_tool_sample_index": int(first),
            "plate_position_B_x": plate_B[0], "plate_position_B_y": plate_B[1],
            "plate_position_B_z": plate_B[2], "mouth_proxy_position_B_x": mouth_proxy_B[0],
            "mouth_proxy_position_B_y": mouth_proxy_B[1], "mouth_proxy_position_B_z": mouth_proxy_B[2],
            "valid_plate_fraction": float(plate_valid.mean()),
            "finite_mouth_proxy_fraction": float(mouth_finite.mean()),
            "mouth_proxy_constant_within_record": bool(mouth_max_deviation <= 1e-12),
            "mouth_proxy_max_deviation_m": mouth_max_deviation,
            "segment_duration_s": duration, "valid_tool_pose_samples": int(tool_valid.sum()),
            "total_samples": int(len(time)),
            "final_fork_to_mouth_proxy_distance_m": float(np.linalg.norm(position[final] - mouth_proxy_B)),
            "max_relative_rotation_rad": float(np.max(np.linalg.norm(relative_rotvec, axis=1))),
        }
    context = np.r_[R0.T @ (plate_B - p0), R0.T @ (mouth_proxy_B - p0)]
    return G0Record(
        entry["record_id"], entry["parent_bite_id"], entry["source_take"], entry["phase"],
        p0, R0, plate_B, mouth_proxy_B, context, duration, s_valid, y_observed,
        relative_position_grid, relative_orientation_grid, gt_position_B, gt_orientation_B, audit,
    )


def load_phase15_records(phase1_dir: Path) -> tuple[list[G0Record], dict, list[dict]]:
    """Sole G0 loader. It rejects every source schema except Phase 1.5."""
    phase1_dir = Path(phase1_dir)
    manifest_path = phase1_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "phase1-v1.5":
        raise ValueError("G0 requires Phase 1.5 only")
    records, exclusions = [], []
    for entry in manifest["records"]:
        try:
            records.append(_load_record(phase1_dir, entry))
        except ValueError as error:
            exclusions.append({
                "record_id": entry.get("record_id"), "parent_bite_id": entry.get("parent_bite_id"),
                "take": entry.get("source_take"), "phase": entry.get("phase"), "usable": False,
                "exclusion_reason": str(error),
            })
    if not records:
        raise ValueError("no G0-usable Phase 1.5 records")
    if sorted({r.take for r in records}) != sorted(manifest["raw_fork_sources"]):
        raise ValueError("G0 usable records do not span the complete seven-take source")
    return records, manifest, exclusions


def _context_scaler(records: list[G0Record]) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([r.context for r in records])
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def training_phase_median_duration(training: list[G0Record], phase: str) -> float:
    """Provisional timing from outer-training records only."""
    durations = [r.duration_s for r in training if r.phase == phase]
    if not durations:
        raise ValueError(f"no outer-training durations for phase {phase}")
    return float(np.median(durations))


def retrieval_generate(query: G0Record, training: list[G0Record], trace: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    candidates = sorted((r for r in training if r.phase == query.phase), key=lambda r: r.record_id)
    if not candidates:
        raise ValueError(f"no same-phase retrieval candidate for {query.record_id}")
    mean, scale = _context_scaler(candidates)
    distances = np.linalg.norm((np.vstack([r.context for r in candidates]) - query.context) / scale, axis=1)
    retrieved = candidates[int(np.argmin(distances))]
    if trace is not None:
        trace.setdefault("retrieval_events", []).append({
            "outer_held": query.take, "query": query.record_id,
            "candidate_takes": sorted({r.take for r in candidates}),
            "candidate_phases": sorted({r.phase for r in candidates}),
            "scaler_takes": sorted({r.take for r in candidates}),
        })
    anchor_slice = slice(3, 6) if query.phase == "transfer" else slice(0, 3)
    delta = query.context[anchor_slice] - retrieved.context[anchor_slice]
    h = 3.0 * GRID ** 2 - 2.0 * GRID ** 3
    relative_position = retrieved.relative_position_grid + h[:, None] * delta
    position_B = query.p0 + (query.R0 @ relative_position.T).T
    orientation_B = np.einsum("ij,njk->nik", query.R0, retrieved.relative_orientation_grid)
    position_B[0] = query.p0
    orientation_B[0] = query.R0
    return position_B, orientation_B, {
        "retrieved_record_id": retrieved.record_id,
        "retrieval_context_distance": float(np.min(distances)),
        "context_scaler_mean": mean, "context_scaler_scale": scale,
    }


def _trajectory_weights(record: G0Record, n_basis: int = 12) -> np.ndarray:
    phi = rbf_basis(record.s_observed, n_basis)
    # A tiny fixed numerical stabilizer only resolves rank/roundoff; it is not tuned.
    lhs = phi.T @ phi + 1e-10 * np.eye(n_basis)
    return np.linalg.solve(lhs, phi.T @ record.y_observed).reshape(-1)


def _fit_context_ridge(records: list[G0Record], alpha: float, trace: dict | None = None,
                       event: str = "fit") -> tuple[Ridge, np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = _context_scaler(records)
    X = (np.vstack([r.context for r in records]) - mean) / scale
    W = np.vstack([_trajectory_weights(r) for r in records])
    model = Ridge(alpha=alpha).fit(X, W)
    residual = W - model.predict(X)
    covariance = np.cov(residual, rowvar=False, ddof=1) if len(records) > 1 else np.zeros((W.shape[1], W.shape[1]))
    if trace is not None:
        trace.setdefault("promp_events", []).append({
            "event": event, "takes": sorted({r.take for r in records}),
            "record_ids": sorted(r.record_id for r in records),
        })
    return model, mean, scale, covariance


def _promp_relative(model: Ridge, mean: np.ndarray, scale: np.ndarray, query: G0Record,
                    n_basis: int = 12) -> tuple[np.ndarray, np.ndarray]:
    weights = model.predict(((query.context - mean) / scale)[None])[0].reshape(n_basis, 6)
    prediction = rbf_basis(GRID, n_basis) @ weights
    relative_position = prediction[:, :3] - prediction[0, :3]
    raw_rotation = Rotation.from_rotvec(prediction[:, 3:]).as_matrix()
    relative_orientation = np.einsum("ij,njk->nik", raw_rotation[0].T, raw_rotation)
    relative_position[0] = 0.0
    relative_orientation[0] = np.eye(3)
    return relative_position, relative_orientation


def choose_promp_alpha(training: list[G0Record], alphas: tuple[float, ...],
                       n_basis: int = 12, trace: dict | None = None) -> tuple[float, list[dict]]:
    """Inner leave-one-training-take-out selection by bite-balanced position RMS."""
    takes = sorted({r.take for r in training})
    scores = []
    for alpha in alphas:
        errors = []
        for held in takes:
            inner_train = [r for r in training if r.take != held]
            validation = [r for r in training if r.take == held]
            model, mean, scale, _ = _fit_context_ridge(inner_train, alpha, trace, "inner_alpha_fit")
            for record in validation:
                position, _ = _promp_relative(model, mean, scale, record, n_basis)
                errors.append(float(np.sqrt(np.mean(np.sum((position - record.relative_position_grid) ** 2, axis=1)))))
        scores.append({"alpha": float(alpha), "inner_position_rms_m": float(np.mean(errors))})
    selected = min(scores, key=lambda row: (row["inner_position_rms_m"], row["alpha"]))["alpha"]
    return float(selected), scores


def promp_generate(query: G0Record, training: list[G0Record], alpha: float,
                   n_basis: int = 12, trace: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    same_phase = [r for r in training if r.phase == query.phase]
    model, mean, scale, covariance = _fit_context_ridge(same_phase, alpha, trace, "outer_fit")
    relative_position, relative_orientation = _promp_relative(model, mean, scale, query, n_basis)
    position_B = query.p0 + (query.R0 @ relative_position.T).T
    orientation_B = np.einsum("ij,njk->nik", query.R0, relative_orientation)
    position_B[0] = query.p0
    orientation_B[0] = query.R0
    return position_B, orientation_B, {
        "retrieved_record_id": "", "retrieval_context_distance": np.nan,
        "context_scaler_mean": mean, "context_scaler_scale": scale,
        "residual_weight_covariance_trace": float(np.trace(covariance)),
        "residual_weight_covariance_shape": list(covariance.shape),
    }


def validate_generated(position: np.ndarray, orientation: np.ndarray, record: G0Record) -> tuple[bool, str]:
    if position.shape != (101, 3) or orientation.shape != (101, 3, 3):
        return False, "INVALID_SAMPLE_SHAPE"
    if not np.isfinite(position).all():
        return False, "NONFINITE_POSITION"
    if not _proper_so3(orientation).all():
        return False, "INVALID_SO3"
    if not np.allclose(position[0], record.p0, atol=1e-12, rtol=0):
        return False, "INITIAL_POSITION_MISMATCH"
    initial_angle = Rotation.from_matrix(record.R0.T @ orientation[0]).magnitude()
    if initial_angle > 1e-12:
        return False, "INITIAL_ORIENTATION_MISMATCH"
    return True, "SUCCESS"


def _smoothness(position: np.ndarray, orientation: np.ndarray) -> dict:
    first = np.gradient(position, GRID, axis=0, edge_order=2)
    second = np.gradient(first, GRID, axis=0, edge_order=2)
    angular = Rotation.from_matrix(np.einsum("nij,njk->nik", orientation[:-1].transpose(0, 2, 1), orientation[1:])).magnitude() / np.diff(GRID)
    return {
        "position_first_derivative_rms_per_normalized_phase": float(np.sqrt(np.mean(np.sum(first ** 2, axis=1)))),
        "position_first_derivative_max_per_normalized_phase": float(np.max(np.linalg.norm(first, axis=1))),
        "position_second_derivative_rms_per_normalized_phase2": float(np.sqrt(np.mean(np.sum(second ** 2, axis=1)))),
        "position_second_derivative_max_per_normalized_phase2": float(np.max(np.linalg.norm(second, axis=1))),
        "angular_increment_mean_rad_per_normalized_phase": float(np.mean(angular)),
        "angular_increment_max_rad_per_normalized_phase": float(np.max(angular)),
    }


def trajectory_metrics(record: G0Record, generator: str, position: np.ndarray,
                       orientation: np.ndarray, generated_duration: float, metadata: dict) -> dict:
    position_error = np.linalg.norm(position - record.gt_position_B, axis=1)
    relative_rotation = np.einsum("nij,njk->nik", orientation.transpose(0, 2, 1), record.gt_orientation_B)
    orientation_error = Rotation.from_matrix(relative_rotation).magnitude()
    generated_path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    gt_path = float(np.linalg.norm(np.diff(record.gt_position_B, axis=0), axis=1).sum())
    generated_displacement = position[-1] - position[0]
    gt_displacement = record.gt_position_B[-1] - record.gt_position_B[0]
    anchor = record.mouth_proxy_B if record.phase == "transfer" else record.plate_B
    generated_endpoint_vector = anchor - position[-1]
    gt_endpoint_vector = anchor - record.gt_position_B[-1]
    result = {
        "record_id": record.record_id, "parent_bite_id": record.parent_bite_id,
        "take": record.take, "phase": record.phase, "generator": generator,
        "outer_fold": record.take, "generation_status": "SUCCESS",
        "rms_position_error_m": float(np.sqrt(np.mean(position_error ** 2))),
        "mean_position_error_m": float(np.mean(position_error)),
        "p95_position_error_m": float(np.percentile(position_error, 95)),
        "max_position_error_m": float(np.max(position_error)),
        "final_position_error_m": float(position_error[-1]),
        "mean_orientation_error_rad": float(np.mean(orientation_error)),
        "rms_orientation_error_rad": float(np.sqrt(np.mean(orientation_error ** 2))),
        "p95_orientation_error_rad": float(np.percentile(orientation_error, 95)),
        "max_orientation_error_rad": float(np.max(orientation_error)),
        "final_orientation_error_rad": float(orientation_error[-1]),
        "generated_path_length_m": generated_path, "gt_path_length_m": gt_path,
        "path_length_absolute_error_m": abs(generated_path - gt_path),
        "path_length_relative_error": abs(generated_path - gt_path) / gt_path if gt_path > 0 else np.nan,
        "translation_displacement_error_m": float(np.linalg.norm(generated_displacement - gt_displacement)),
        "generated_duration_s": generated_duration, "gt_duration_s": record.duration_s,
        "duration_error_s": generated_duration - record.duration_s,
        "retrieved_record_id": metadata.get("retrieved_record_id", ""),
        "retrieval_context_distance": metadata.get("retrieval_context_distance", np.nan),
        "endpoint_anchor": "mouth_proxy" if record.phase == "transfer" else "plate",
    }
    for prefix, vector in (("generated_final_to_anchor", generated_endpoint_vector),
                           ("gt_final_to_anchor", gt_endpoint_vector),
                           ("final_to_anchor_vector_difference", generated_endpoint_vector - gt_endpoint_vector)):
        result.update({f"{prefix}_{axis}_m": float(value) for axis, value in zip("xyz", vector)})
        result[f"{prefix}_norm_m"] = float(np.linalg.norm(vector))
    result.update({f"generated_{key}": value for key, value in _smoothness(position, orientation).items()})
    result.update({f"gt_{key}": value for key, value in _smoothness(record.gt_position_B, record.gt_orientation_B).items()})
    return result


def generated_u_contract_valid(arrays: dict | np.lib.npyio.NpzFile) -> bool:
    required = {
        "normalized_phase", "time_s", "generated_tool_position_B", "generated_tool_orientation_B",
        "initial_tool_position_B", "initial_tool_orientation_B", "phase", "plate_position_B",
        "plate_position_valid", "generation_status",
    }
    if not required.issubset(set(arrays.keys())):
        return False
    return (
        np.asarray(arrays["normalized_phase"]).shape == (101,)
        and np.asarray(arrays["time_s"]).shape == (101,)
        and np.asarray(arrays["generated_tool_position_B"]).shape == (101, 3)
        and np.asarray(arrays["generated_tool_orientation_B"]).shape == (101, 3, 3)
        and str(np.asarray(arrays["phase"]).item()) in PHASES
        and np.asarray(arrays["plate_position_B"]).shape == (3,)
    )


def _save_generated(path: Path, record: G0Record, generator: str, position: np.ndarray,
                    orientation: np.ndarray, duration: float, metadata: dict,
                    training: list[G0Record], source_hash: str, alpha: float | None) -> dict:
    arrays = {
        "record_id": np.array(record.record_id), "parent_bite_id": np.array(record.parent_bite_id),
        "take": np.array(record.take), "phase": np.array(record.phase), "generator": np.array(generator),
        "normalized_phase": GRID, "time_s": GRID * duration,
        "generated_tool_position_B": position, "generated_tool_orientation_B": orientation,
        "initial_tool_position_B": record.p0, "initial_tool_orientation_B": record.R0,
        "plate_position_B": record.plate_B, "plate_position_valid": np.array(True),
        "mouth_proxy_position_B": record.mouth_proxy_B,
        "mouth_proxy_is_independently_measured": np.array(False),
        "deployment_context_validated": np.array(False),
        "retrieved_record_id": np.array(metadata.get("retrieved_record_id", "")),
        "retrieval_context_distance": np.array(metadata.get("retrieval_context_distance", np.nan)),
        "outer_fold": np.array(record.take), "generation_status": np.array("SUCCESS"),
        "training_take_ids": np.array(sorted({r.take for r in training})),
        "training_record_ids": np.array(sorted(r.record_id for r in training if r.phase == record.phase)),
        "held_out_take_excluded": np.array(record.take not in {r.take for r in training}),
        "source_schema_version": np.array("phase1-v1.5"),
        "source_phase1_manifest_sha256": np.array(source_hash),
        "context_scaler_mean": metadata["context_scaler_mean"],
        "context_scaler_scale": metadata["context_scaler_scale"],
        "promp_ridge_alpha": np.array(np.nan if alpha is None else alpha),
        "timing_model": np.array("training_phase_median"),
        "timing_physical_limits_validated": np.array(False),
        "strong_local_feature_contract_version": np.array("strong-local-l3-v1"),
        "strong_local_feature_order": np.array(STRONG_LOCAL_FEATURE_ORDER),
    }
    if not generated_u_contract_valid(arrays):
        raise ValueError(f"generated-U StrongLocal contract failed for {record.record_id}/{generator}")
    np.savez_compressed(path, **arrays)
    return {"record_id": record.record_id, "generator": generator, "path": str(path), "sha256": _sha256(path)}


def _aggregate_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rms_position_error_m", "mean_position_error_m", "p95_position_error_m",
        "max_position_error_m", "final_position_error_m", "mean_orientation_error_rad",
        "rms_orientation_error_rad", "p95_orientation_error_rad", "max_orientation_error_rad",
        "final_orientation_error_rad", "path_length_absolute_error_m", "path_length_relative_error",
        "translation_displacement_error_m",
    ]
    rows = []
    for generator, frame in metrics.groupby("generator", sort=True):
        groups = [("overall", "overall", frame)]
        groups += [("phase", str(key), value) for key, value in frame.groupby("phase", sort=True)]
        groups += [("take", str(key), value) for key, value in frame.groupby("take", sort=True)]
        for grouping, group, subset in groups:
            bite = subset.groupby("parent_bite_id", as_index=False)[columns].mean(numeric_only=True)
            row = {"generator": generator, "grouping": grouping, "group": group,
                   "n_records": len(subset), "n_bites": len(bite)}
            row.update({name: float(bite[name].mean()) for name in columns})
            rows.append(row)
    return pd.DataFrame(rows)


def _bite_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "rms_position_error_m", "rms_orientation_error_rad", "final_position_error_m",
        "path_length_absolute_error_m", "mean_position_error_m", "mean_orientation_error_rad",
    ]
    return metrics.groupby(["generator", "parent_bite_id", "take"], as_index=False)[numeric].mean()


def paired_bootstrap(metrics: pd.DataFrame, seed: int = 20260915, resamples: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metric_names = ("rms_position_error_m", "rms_orientation_error_rad", "final_position_error_m", "path_length_absolute_error_m")
    for phase, frame in [("overall", metrics), *list(metrics.groupby("phase", sort=True))]:
        for metric in metric_names:
            bite = frame.groupby(["generator", "parent_bite_id"])[metric].mean().unstack("generator")
            bite = bite.dropna(subset=list(GENERATORS))
            difference = bite["contextual_promp"].to_numpy() - bite["retrieval"].to_numpy()
            draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
            samples = difference[draws].mean(axis=1)
            rows.append({
                "group": str(phase), "metric": metric,
                "comparison": "contextual_promp_minus_retrieval",
                "convention": "negative favors contextual_promp",
                "mean_difference": float(difference.mean()),
                "ci95_low": float(np.percentile(samples, 2.5)),
                "ci95_high": float(np.percentile(samples, 97.5)),
                "n_parent_bites": int(len(difference)), "seed": seed, "resamples": resamples,
            })
    return pd.DataFrame(rows)


def _representative_examples(metrics: pd.DataFrame) -> dict:
    pivot = metrics.pivot(index=["record_id", "parent_bite_id", "take", "phase"],
                          columns="generator", values="rms_position_error_m").reset_index()
    pivot["retrieval_minus_promp_m"] = pivot["retrieval"] - pivot["contextual_promp"]
    pivot["both_mean_error_m"] = (pivot["retrieval"] + pivot["contextual_promp"]) / 2
    retrieval = pivot.sort_values(["retrieval_minus_promp_m", "record_id"]).iloc[0]
    promp = pivot.sort_values(["retrieval_minus_promp_m", "record_id"], ascending=[False, True]).iloc[0]
    median = float(pivot["retrieval_minus_promp_m"].median())
    typical = pivot.assign(distance=(pivot["retrieval_minus_promp_m"] - median).abs()).sort_values(["distance", "record_id"]).iloc[0]
    selected_phases = {retrieval.phase, promp.phase, typical.phase}
    difficult_pool = pivot if len(selected_phases) == 2 else pivot[pivot.phase != next(iter(selected_phases))]
    difficult = difficult_pool.sort_values(["both_mean_error_m", "record_id"], ascending=[False, True]).iloc[0]
    def pack(row: pd.Series) -> dict:
        return {
            "record_id": row["record_id"], "parent_bite_id": row["parent_bite_id"],
            "take": row["take"], "phase": row["phase"],
            "retrieval_rms_position_error_m": float(row["retrieval"]),
            "contextual_promp_rms_position_error_m": float(row["contextual_promp"]),
            "retrieval_minus_promp_m": float(row["retrieval_minus_promp_m"]),
        }
    return {"retrieval_favorable": pack(retrieval), "contextual_promp_favorable": pack(promp),
            "typical": pack(typical), "difficult_ood": pack(difficult),
            "selection": "deterministic from OOF RMS position errors; no visual selection"}


def _plots(output: Path, aggregates: pd.DataFrame, metrics: pd.DataFrame, examples: dict,
           generated_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], records: dict[str, G0Record]) -> None:
    plots = output / "plots"
    overall = aggregates[aggregates.grouping == "overall"].set_index("generator")
    specs = [
        ("position_error.png", "rms_position_error_m", "bite-balanced trajectory RMS position error (m)"),
        ("orientation_error.png", "rms_orientation_error_rad", "bite-balanced trajectory RMS orientation error (rad)"),
        ("path_length_error.png", "path_length_absolute_error_m", "bite-balanced absolute path-length error (m)"),
    ]
    for filename, column, ylabel in specs:
        fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(list(GENERATORS), overall.loc[list(GENERATORS), column])
        ax.set(ylabel=ylabel); fig.tight_layout(); fig.savefig(plots / filename, dpi=160); plt.close(fig)
    phase = aggregates[aggregates.grouping == "phase"]
    table = phase.pivot(index="group", columns="generator", values="rms_position_error_m").reindex(PHASES)
    fig, ax = plt.subplots(figsize=(6, 4)); table.plot.bar(ax=ax); ax.set(ylabel="RMS position error (m)", xlabel="phase")
    fig.tight_layout(); fig.savefig(plots / "transfer_withdrawal.png", dpi=160); plt.close(fig)

    selected = [examples[key]["record_id"] for key in ("retrieval_favorable", "contextual_promp_favorable", "typical", "difficult_ood")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for ax, record_id in zip(axes.ravel(), selected):
        record = records[record_id]
        ax.plot(record.gt_position_B[:, 0], record.gt_position_B[:, 1], "k--", label="GT")
        for generator in GENERATORS:
            position, _ = generated_cache[(record_id, generator)]
            ax.plot(position[:, 0], position[:, 1], label=generator)
        ax.set(title=f"{record_id} ({record.phase})", xlabel="B-x (m)", ylabel="B-y (m)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(plots / "example_trajectories.png", dpi=160); plt.close(fig)


def process_generator_g0(phase1_dir: Path, output_dir: Path, cfg: GeneratorG0Config,
                         trace: dict | None = None) -> dict:
    phase1_dir, output_dir = Path(phase1_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS:
        (output_dir / "generated" / generator).mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    records, source_manifest, exclusions = load_phase15_records(phase1_dir)
    source_manifest_path = phase1_dir / "manifest.json"
    source_hash = _sha256(source_manifest_path)
    audits = [r.audit for r in records] + exclusions
    pd.DataFrame(audits).to_csv(output_dir / "context_audit.csv", index=False)
    by_id = {r.record_id: r for r in records}
    takes = sorted({r.take for r in records})
    if len(takes) != 7:
        raise ValueError(f"G0 requires seven outer take folds, found {len(takes)}")

    metric_rows, artifact_rows, selected_alphas, alpha_scores = [], [], {}, {}
    generated_cache = {}
    for held in takes:
        outer_train = [r for r in records if r.take != held]
        outer_test = [r for r in records if r.take == held]
        if trace is not None:
            trace["current_outer"] = held
        phase_models = {}
        duration_by_phase = {}
        for phase in PHASES:
            phase_train = [r for r in outer_train if r.phase == phase]
            duration_by_phase[phase] = training_phase_median_duration(outer_train, phase)
            alpha, scores = choose_promp_alpha(phase_train, cfg.alphas, cfg.basis_functions, trace)
            selected_alphas[f"{held}:{phase}"] = alpha
            alpha_scores[f"{held}:{phase}"] = scores
            phase_models[phase] = alpha
        for query in outer_test:
            duration = duration_by_phase[query.phase]
            for generator in GENERATORS:
                if generator == "retrieval":
                    position, orientation, metadata = retrieval_generate(query, outer_train, trace)
                    alpha = None
                else:
                    alpha = phase_models[query.phase]
                    position, orientation, metadata = promp_generate(query, outer_train, alpha, cfg.basis_functions, trace)
                valid, status = validate_generated(position, orientation, query)
                if not valid:
                    raise ValueError(f"{query.record_id}/{generator} generation failed: {status}")
                metric_rows.append(trajectory_metrics(query, generator, position, orientation, duration, metadata))
                path = output_dir / "generated" / generator / f"{query.record_id}.npz"
                artifact_rows.append(_save_generated(path, query, generator, position, orientation, duration,
                                                     metadata, outer_train, source_hash, alpha))
                generated_cache[(query.record_id, generator)] = (position, orientation)

    metrics = pd.DataFrame(metric_rows).sort_values(["record_id", "generator"])
    metrics.to_csv(output_dir / "record_metrics.csv", index=False)
    aggregates = _aggregate_rows(metrics)
    aggregates.to_csv(output_dir / "fold_metrics.csv", index=False)
    bite = _bite_metrics(metrics)
    bite.to_csv(output_dir / "bite_metrics.csv", index=False)
    bootstrap = paired_bootstrap(metrics, cfg.bootstrap_seed, cfg.bootstrap_resamples)
    bootstrap.to_csv(output_dir / "pairwise_bootstrap.csv", index=False)
    duration = metrics.drop_duplicates("record_id")[["record_id", "parent_bite_id", "take", "phase", "generated_duration_s", "gt_duration_s", "duration_error_s"]].copy()
    duration["absolute_duration_error_s"] = duration.duration_error_s.abs()
    duration.to_csv(output_dir / "duration_metrics.csv", index=False)
    examples = _representative_examples(metrics)
    (output_dir / "representative_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True))
    _plots(output_dir, aggregates, metrics, examples, generated_cache, by_id)

    context = pd.DataFrame([r.audit for r in records])
    duration_summary = []
    for group, frame in [("overall", duration), *list(duration.groupby("phase", sort=True))]:
        duration_summary.append({"group": str(group), "n_records": len(frame),
                                 "mae_s": float(frame.absolute_duration_error_s.mean()),
                                 "rmse_s": float(np.sqrt(np.mean(frame.duration_error_s ** 2)))})
    retrieval_distances = metrics[metrics.generator == "retrieval"].retrieval_context_distance
    source_files_read = [str(source_manifest_path.resolve())] + [str((phase1_dir / entry["file"]).resolve()) for entry in source_manifest["records"]]
    summary = {
        "schema_version": "generator-g0-v1",
        "dataset": {"usable_records": len(records), "excluded_records": len(exclusions),
                    "parent_bites": len({r.parent_bite_id for r in records}), "takes": len(takes),
                    "records_by_phase": {phase: sum(r.phase == phase for r in records) for phase in PHASES}},
        "semantic_boundary": {"mouth_proxy_is_independently_measured": False,
                              "deployment_context_validated": False,
                              "physical_mouth_position_generalization_claimed": False,
                              "upstream_field_alias": "mouth_target_position -> mouth_proxy"},
        "input_contract": {"phase": list(PHASES), "initial_fork_pose": "tracked fork rigid-body SE(3)",
                           "context": "[R0.T@(plate-p0), R0.T@(mouth_proxy-p0)]",
                           "excluded": ["take ID", "demonstrator ID", "recipient ID", "absolute coordinates",
                                        "human psi", "robot q", "Exact-SEW", "StrongLocal", "robot outcomes"]},
        "context_audit": {
            "all_mouth_proxy_constant_within_record": bool(context.mouth_proxy_constant_within_record.all()),
            "max_mouth_proxy_deviation_m": float(context.mouth_proxy_max_deviation_m.max()),
            "valid_plate_fraction": {"min": float(context.valid_plate_fraction.min()), "mean": float(context.valid_plate_fraction.mean())},
            "finite_mouth_proxy_fraction": {"min": float(context.finite_mouth_proxy_fraction.min()), "mean": float(context.finite_mouth_proxy_fraction.mean())},
            "final_fork_to_mouth_proxy_distance_m_by_phase": context.groupby("phase").final_fork_to_mouth_proxy_distance_m.agg(["mean", "median", "max"]).to_dict("index"),
            "relative_rotation_rad": {"max": float(context.max_relative_rotation_rad.max()), "pi_margin": float(np.pi - context.max_relative_rotation_rad.max())},
        },
        "generators": {
            "retrieval": {"candidate_pool": "outer-training records of same phase", "context_scaling": "outer-training same-phase only",
                          "distance": "Euclidean in standardized six-dimensional relative context", "tie_break": "stable record_id",
                          "endpoint_adaptation": "translation-only cubic smoothstep; mouth_proxy for transfer, plate for withdrawal",
                          "retrieval_context_distance": {"mean": float(retrieval_distances.mean()), "median": float(retrieval_distances.median()), "p95": float(retrieval_distances.quantile(.95)), "max": float(retrieval_distances.max())}},
            "contextual_promp": {"basis": "12 normalized Gaussian RBFs", "channels": 6,
                                 "conditioner": "standardized relative context -> multi-output ridge weight mean",
                                 "alpha_selection": "inner leave-one-training-take-out, mean trajectory position RMS",
                                 "selected_alphas": selected_alphas, "inner_scores": alpha_scores,
                                 "sampling": "conditional mean only; residual covariance estimated, never sampled"},
        },
        "generation": {"grid_samples": 101, "successes": int((metrics.generation_status == "SUCCESS").sum()),
                       "attempts": len(metrics), "success_rate": float((metrics.generation_status == "SUCCESS").mean()),
                       "initial_pose_exact": True},
        "metrics": {"overall": aggregates[aggregates.grouping == "overall"].to_dict("records"),
                    "per_phase": aggregates[aggregates.grouping == "phase"].to_dict("records"),
                    "per_take": aggregates[aggregates.grouping == "take"].to_dict("records")},
        "paired_parent_bite_bootstrap": bootstrap.to_dict("records"),
        "duration": {"timing_model": "training_phase_median", "timing_physical_limits_validated": False,
                     "diagnostics": duration_summary},
        "representative_examples": examples,
        "generated_artifact_contract": {"paths": "generated/{retrieval,contextual_promp}/<record_id>.npz",
                                        "spatial_samples": 101, "future_feature_contract": "strong-local-l3-v1",
                                        "feature_count_reconstructable": len(STRONG_LOCAL_FEATURE_ORDER),
                                        "contains_pose_time_phase_plate_context": True, "g1_requires_retraining_g0": False},
        "interpretation": {"generation_success_is_numerical_only": True,
                           "physical_task_success_threshold_calibrated": False,
                           "accuracy_is_against_held_out_measured_fork_trajectory": True},
        "scope": {"g1_started": False, "robot_ik_started": False, "calibration_started": False,
                  "collision_started": False, "retiming_started": False, "complex_model_added": False},
        "provenance": {"source_schema": "phase1-v1.5", "source_manifest": str(source_manifest_path.resolve()),
                       "source_manifest_sha256": source_hash, "source_files_read": source_files_read,
                       "outer_folds": takes, "code_revision": _revision(Path(__file__).resolve().parents[2]),
                       "config": asdict(cfg),
                       "code_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False).stdout.strip())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    manifest = {
        "schema_version": "generator-g0-v1", "summary": "summary.json", "context_audit": "context_audit.csv",
        "fold_metrics": "fold_metrics.csv", "record_metrics": "record_metrics.csv", "bite_metrics": "bite_metrics.csv",
        "pairwise_bootstrap": "pairwise_bootstrap.csv", "duration_metrics": "duration_metrics.csv",
        "representative_examples": "representative_examples.json", "generated_artifacts": artifact_rows,
        "plots": [f"plots/{name}" for name in ("position_error.png", "orientation_error.png", "path_length_error.png", "transfer_withdrawal.png", "example_trajectories.png")],
        "source_phase1_manifest_sha256": source_hash, "mouth_proxy_is_independently_measured": False,
        "deployment_context_validated": False, "config": asdict(cfg),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return summary
