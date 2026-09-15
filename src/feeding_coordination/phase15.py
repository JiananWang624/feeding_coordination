"""Phase 1.5: associate recorder fork rigid-body poses with Phase 1 rows."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .phase1 import LANDMARKS, Phase1Config, _revision, process_frame

RAW_COLUMNS = ("host_time_s", "natnet_frame_number", "rigid_body_name", "x", "y", "z", "qx", "qy", "qz", "qw", "tracking_valid")


def _stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values): return {x: None for x in ("mean", "median", "p95", "max")}
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def load_raw_fork(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing: raise ValueError(f"{path}: missing raw fork columns {sorted(missing)}")
    raw = raw.loc[raw.rigid_body_name == "fork"].copy()
    if raw.empty: raise ValueError(f"{path}: no rigid body named fork")
    raw["natnet_frame_number"] = pd.to_numeric(raw.natnet_frame_number, errors="raise").astype(int)
    if raw.natnet_frame_number.duplicated().any(): raise ValueError(f"{path}: fork NatNet frame identifiers are not unique")
    return raw.set_index("natnet_frame_number", drop=False)


def trajectory_metadata(take: str, cfg: Phase1Config) -> dict:
    if take not in cfg.demonstrator_by_take: raise ValueError(f"demonstrator mapping missing take {take}")
    return {"session_id": take, "demonstrator_id": cfg.demonstrator_by_take[take], "recipient_id": None}


def association_and_consistency_status(matching: dict, orientation_stats: dict, hand_stats: dict, cfg: Phase1Config) -> tuple[str, str]:
    association = "verified" if matching["overall"]["exact"] == matching["overall"]["total"] else "partial"
    consistent = orientation_stats["max"] is not None and hand_stats["max"] is not None and orientation_stats["max"] <= cfg.max_orientation_consistency_rad and hand_stats["max"] <= cfg.max_derived_hand_consistency_m
    return association, "verified" if consistent else "discrepant"


def associate_fork(frame: pd.DataFrame, result: dict, raw: pd.DataFrame, cfg: Phase1Config) -> dict:
    """Fill only tool fields; all Phase 1 human and SEW fields remain untouched."""
    n = len(frame); keys = frame.motive_frame.to_numpy(int); raw_rows = raw.reindex(keys)
    matched = raw_rows.natnet_frame_number.notna().to_numpy()
    tracking = np.zeros(n, bool); tracking[matched] = pd.to_numeric(raw_rows.loc[matched, "tracking_valid"], errors="coerce").fillna(0).astype(bool).to_numpy()
    xyz = raw_rows[["x", "y", "z"]].to_numpy(float)
    quat = raw_rows[["qx", "qy", "qz", "qw"]].to_numpy(float)
    raw_valid = np.isfinite(xyz).all(1) & np.isfinite(quat).all(1) & (np.linalg.norm(quat, axis=1) > 0)
    valid = matched & tracking & raw_valid
    status = np.full(n, "unmatched", dtype="<U32")
    status[matched & ~tracking] = "tracking_invalid"
    status[matched & tracking & ~raw_valid] = "invalid_raw_pose"
    status[valid] = "valid"
    position = np.full((n, 3), np.nan); orientation = np.full((n, 3, 3), np.nan)
    if valid.any():
        R_BL = np.asarray(cfg.R_B_L, float)
        position[valid] = xyz[valid] @ R_BL.T + np.asarray(cfg.translation_B_m, float)
        orientation[valid] = R_BL @ Rotation.from_quat(quat[valid] / np.linalg.norm(quat[valid], axis=1)[:, None]).as_matrix()
    result.update({
        "tool_position": position, "tool_orientation": orientation, "tool_pose_valid": valid,
        "tool_pose_status": status, "tool_pose_source": np.full(n, "optitrack_fork_rigid_body"),
        "tool_frame_semantics": np.full(n, "tracked_fork_rigid_body_frame"),
        "tool_match_method": np.full(n, "exact_frame_identifier"),
        "raw_natnet_frame_number": np.where(matched, keys, -1).astype(int),
        "raw_host_time_s": raw_rows.host_time_s.to_numpy(float), "tool_match_exact": matched,
        "tool_match_frame_error": np.where(matched, 0, -1).astype(int),
        "tool_tracking_valid": tracking, "tool_raw_pose_valid": raw_valid,
        "tool_tip_calibrated": np.zeros(n, bool), "tool_tip_calibration_status": np.full(n, "missing_fork_tip_calibration"),
        "missing_calibration": np.zeros(n, bool), "calibration_status": np.where(valid, "fork_rigid_body_available", status),
    })
    diagnostic = valid & result["orientation_valid"] & np.isfinite(result["hand_xyz"]).all(1)
    angle = np.full(n, np.nan); hand_error = np.full(n, np.nan)
    if diagnostic.any():
        relative = orientation[diagnostic].transpose(0, 2, 1) @ result["hand_orientation"][diagnostic]
        angle[diagnostic] = Rotation.from_matrix(relative).magnitude()
        reconstructed_hand = position[diagnostic] - cfg.legacy_hand_offset_m * orientation[diagnostic, :, 1]
        hand_error[diagnostic] = np.linalg.norm(reconstructed_hand - result["hand_xyz"][diagnostic], axis=1)
    result["tool_orientation_consistency_rad"] = angle
    result["tool_derived_hand_consistency_m"] = hand_error
    result["raw_host_minus_source_time_s"] = raw_rows.host_time_s.to_numpy(float) - result["source_time_s"]
    return result


def process_csv(input_path, output_dir, cfg: Phase1Config):
    input_path = Path(input_path).resolve(); output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    demos = output_dir / "demos"; demos.mkdir(exist_ok=True)
    data = pd.read_csv(input_path); source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    takes = sorted(data["take"].unique())
    unknown = set(takes) - set(cfg.demonstrator_by_take)
    if unknown: raise ValueError(f"demonstrator mapping missing takes {sorted(unknown)}")
    raw_by_take = {}; raw_sources = {}
    for take in takes:
        raw_root = Path(cfg.raw_fork_root)
        path = ((raw_root if raw_root.is_absolute() else root / raw_root) / cfg.raw_fork_pattern.format(take=take)).resolve()
        if not path.is_file(): raise ValueError(f"raw fork source does not exist: {path}")
        raw_by_take[take] = load_raw_fork(path)
        raw_sources[take] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rigid_body_name": "fork"}
    records = []; aggregate = {}; lengths = []; shoulder = []; diagnostics = {"orientation": [], "hand": [], "host_offset": []}; per_take = {take: {"total": 0, "exact": 0, "tracking_invalid": 0, "unmatched": 0, "invalid_raw_pose": 0, "valid": 0, "orientation": [], "hand": [], "host_offset": []} for take in takes}
    for trajectory_id, group in data.groupby("trajectory_id", sort=True):
        take = group.iloc[0]["take"]; result = associate_fork(group, process_frame(group, cfg), raw_by_take[take], cfg)
        filename = f"{trajectory_id}.npz"; np.savez_compressed(demos / filename, **result)
        lengths.append(len(group)); shoulder.extend(np.linalg.norm(result["shoulder_displacement"], axis=1)[result["shoulder_valid_observation"]])
        info = per_take[take]; info["total"] += len(group); info["exact"] += int(result["tool_match_exact"].sum()); info["valid"] += int(result["tool_pose_valid"].sum())
        for status in ("tracking_invalid", "unmatched", "invalid_raw_pose"): info[status] += int((result["tool_pose_status"] == status).sum())
        for name, key in (("orientation", "tool_orientation_consistency_rad"), ("hand", "tool_derived_hand_consistency_m"), ("host_offset", "raw_host_minus_source_time_s")):
            values = result[key]; diagnostics[name].extend(values[np.isfinite(values)]); info[name].extend(values[np.isfinite(values)])
        for key in ("psi_valid", "psi_invalid", "psi_near_singular", "tool_pose_valid", "orientation_invalid", *[f"{x.lower()}_{suffix}" for x in LANDMARKS for suffix in ("valid_observation", "interpolated", "long_missing")]): aggregate[key] = aggregate.get(key, 0) + int(np.count_nonzero(result[key]))
        records.append({"record_id": trajectory_id, "source_take": take, "bite_id": int(group.iloc[0].bite_id), "parent_bite_id": f"{take}_bite_{int(group.iloc[0].bite_id):03d}", **trajectory_metadata(take, cfg), "calibration_id": None, "calibration_status": "fork_rigid_body_available", "tool_tip_calibration_status": "missing_fork_tip_calibration", "source_file": str(input_path), "source_hash": source_hash, "raw_fork_source": raw_sources[take], "phase": group.iloc[0].event, "samples": len(group), "processing_status_counts": {str(x): int(np.count_nonzero(result["processing_status"] == x)) for x in np.unique(result["processing_status"])}, "psi_quality_counts": {str(x): int(np.count_nonzero(result["psi_quality"] == x)) for x in np.unique(result["psi_quality"])}, "psi_valid_samples": int(result["psi_valid"].sum()), "tool_pose_valid_samples": int(result["tool_pose_valid"].sum()), "file": f"demos/{filename}"})
    for info in per_take.values():
        info["valid_percent"] = 100 * info["valid"] / info["total"]
        for name in ("orientation", "hand", "host_offset"): info[f"{name}_stats"] = _stats(info.pop(name))
    total = len(data); config = asdict(cfg); valid = aggregate["tool_pose_valid"]
    matching = {"method": "exact_frame_identifier", "frame_error": {"matched": 0, "unmatched": -1}, "overall": {"total": total, "exact": sum(x["exact"] for x in per_take.values()), "tracking_invalid": sum(x["tracking_invalid"] for x in per_take.values()), "unmatched": sum(x["unmatched"] for x in per_take.values()), "invalid_raw_pose": sum(x["invalid_raw_pose"] for x in per_take.values()), "valid": valid, "valid_percent": 100 * valid / total}, "per_take": per_take, "host_minus_source_time_s": _stats(diagnostics["host_offset"])}
    orientation_stats, hand_stats = _stats(diagnostics["orientation"]), _stats(diagnostics["hand"])
    association_status, consistency_status = association_and_consistency_status(matching, orientation_stats, hand_stats, cfg)
    summary = {"rows": total, "demonstrations": len(records), "processed_demonstrations": len(records), "failed_demonstrations": 0, "parent_bites": int(data[["take", "bite_id"]].drop_duplicates().shape[0]), "takes": len(takes), "demonstrator_count": len(set(cfg.demonstrator_by_take.values())), "session_count": len(takes), "recipient_count": None, "psi_valid_samples": aggregate["psi_valid"], "psi_invalid_samples": aggregate["psi_invalid"], "psi_near_singular_samples": aggregate["psi_near_singular"], "psi_valid_percent": 100 * aggregate["psi_valid"] / total, "tool_pose_valid_samples": valid, "tool_pose_valid_percent": 100 * valid / total, "anatomical_hand_to_tool_calibration_required": False, "anatomical_hand_to_tool_calibration_available_demonstrations": 0, "tool_rigid_body_pose_available_demonstrations": len(records), "tool_tip_calibration_available_demonstrations": 0, "tool_tip_calibration_missing_demonstrations": len(records), "source_complete_rows": int((data.is_complete == True).sum()), "source_incomplete_rows": int((data.is_complete == False).sum()), "orientation_invalid_samples": aggregate["orientation_invalid"], "landmark_masks": {key: aggregate[key] for key in aggregate if any(key.startswith(x.lower()) for x in LANDMARKS)}, "length_frames": {"min": int(min(lengths)), "median": float(np.median(lengths)), "mean": float(np.mean(lengths)), "max": int(max(lengths))}, "shoulder_displacement_m": _stats(shoulder), "tool_consistency": {"orientation_geodesic_rad": orientation_stats, "derived_hand_position_m": hand_stats}}
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    manifest = {"schema_version": "phase1-v1.5", "source_file": str(input_path), "source_hash": source_hash, "raw_fork_sources": raw_sources, "config": config, "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "code_revision": _revision(root), "code_worktree_dirty": dirty, "coordinate_convention": {"landmark_source": cfg.source_frame, "raw_fork_source": "L: Motive Global, metres", "working": cfg.working_frame, "human_point_transform": "p_B=scale*R_B_L*p_L+t_B", "raw_fork_point_transform": "p_B=R_B_L*p_L+t_B (raw recorder metres; no scale)"}, "tool_convention": {"source": "optitrack_fork_rigid_body", "frame_semantics": "tracked_fork_rigid_body_frame", "rotation": "R_B_U=R_B_L*R_L_U; raw quaternion xyzw normalized", "tip_calibration": "missing_fork_tip_calibration"}, "stereo_reference": {"source_frame": "frozen Exact-SEW native Gen3 base", "working_frame": cfg.working_frame, "transform": "R_B_from_exact_sew_base"}, "orientation_convention": "upstream quaternion-derived extrinsic xyz degrees; scipy from_euler('xyz', degrees=True)", "association_status": association_status, "consistency_status": consistency_status, "consistency_thresholds": {"orientation_geodesic_rad_max": cfg.max_orientation_consistency_rad, "derived_hand_position_m_max": cfg.max_derived_hand_consistency_m}, "matching": matching, "records": records, "summary": summary}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)); return manifest
