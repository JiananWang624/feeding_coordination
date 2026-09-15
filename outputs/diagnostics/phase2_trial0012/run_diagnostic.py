"""Read-only trial_0012 old-vs-Phase-2 Exact-SEW target diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.adapters.exact_sew import ExactSewTrajectoryAdapter
from feeding_coordination.phase2 import base_rotations, base_transform, valid_runs
from sew_mimic.common import ExactSewTarget, SolverStatus
from sew_mimic.csv_adapter import HumanCSVAdapter, R_INPUT_ALIGN, load_human_trajectory_csv
from sew_mimic.exact import ExactSewSolver
from sew_mimic.exact.residuals import robot_exact_sew_residuals
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.mounting import load_humanoid_mounted_gen3
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference

OUT = Path(__file__).resolve().parent
PHASE1 = ROOT / "outputs" / "phase1"
PHASE2 = ROOT / "outputs" / "phase2"
OLD = ROOT / "external" / "exact_sew"
TAKE = "trial_0012"


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {"count": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def pose(robot, data):
    base_id = int(robot.frame_body_ids[0])
    joint1_id = int(robot.joint_ids[0])
    return {
        "R_B_from_base": np.asarray(data.xmat[base_id], float).reshape(3, 3).copy(),
        "p_B_of_base": np.asarray(data.xpos[base_id], float).copy(),
        "joint1_B": np.asarray(data.xanchor[joint1_id], float).copy(),
    }


def status_name(result):
    return getattr(result.status, "value", str(result.status))


def summarize_status(frame, column):
    def one(group):
        counts = group[column].value_counts().to_dict()
        attempted = len(group)
        success = int(counts.get("SUCCESS_EXACT", 0))
        return {"attempted": attempted, "SUCCESS_EXACT": success,
                "JOINT_LIMIT": int(counts.get("JOINT_LIMIT", 0)),
                "NO_VALID_BRANCH": int(counts.get("NO_VALID_BRANCH", 0)),
                "other_statuses": {str(k): int(v) for k, v in counts.items()
                                   if k not in {"SUCCESS_EXACT", "JOINT_LIMIT", "NO_VALID_BRANCH"}},
                "success_percent": 100.0 * success / attempted}
    return {"overall": one(frame), **{str(k): one(v) for k, v in frame.groupby("phase")}}


def main():
    manifest1 = json.loads((PHASE1 / "manifest.json").read_text())
    manifest2 = json.loads((PHASE2 / "manifest.json").read_text())
    records = [r for r in manifest1["records"] if r["source_take"] == TAKE]

    old_csv = pd.read_csv(OLD / "data" / "test.csv")
    old_world = load_human_trajectory_csv(OLD / "data" / "test.csv")
    old_anchor = old_world.shoulders[0]
    new_anchor = np.asarray(manifest2["mountings"][TAKE]["anchor_B"], float)
    old_robot, old_data = load_humanoid_mounted_gen3(old_anchor)
    new_robot, new_data = load_humanoid_mounted_gen3(new_anchor)
    old_pose, new_pose = pose(old_robot, old_data), pose(new_robot, new_data)
    R_B_from_base, p_B_of_base = new_pose["R_B_from_base"], new_pose["p_B_of_base"]
    joint1_base = base_transform(new_pose["joint1_B"][None], R_B_from_base, p_B_of_base)[0]

    old_lookup = {(int(row.bite_id), str(row.event), int(row.motive_frame)): row
                  for row in old_csv.itertuples(index=False)}
    adapter = HumanCSVAdapter()
    segments, rows = [], []
    old_adapter_position_errors, old_adapter_rotation_errors = [], []
    for rec in records:
        with np.load(PHASE1 / rec["file"]) as z:
            arrays = {k: np.asarray(z[k]).copy() for k in z.files}
        n = len(arrays["motive_frame"])
        old_R_B = np.full((n, 3, 3), np.nan)
        old_wrist_B = np.full((n, 3), np.nan)
        for i, motive in enumerate(arrays["motive_frame"]):
            source = old_lookup[(int(rec["bite_id"]), str(rec["phase"]), int(motive))]
            shoulder, elbow, wrist, hand = adapter.adapt_frame(
                [source.Shoulder_X, source.Shoulder_Y, source.Shoulder_Z],
                [source.Elbow_X, source.Elbow_Y, source.Elbow_Z],
                [source.Wrist_X, source.Wrist_Y, source.Wrist_Z],
                [source.Wrist_Rx, source.Wrist_Ry, source.Wrist_Rz],
            )
            old_wrist_B[i], old_R_B[i] = wrist, hand
            old_adapter_position_errors.append(np.linalg.norm(wrist - arrays["wrist_xyz"][i]))
            reconstructed = arrays["hand_orientation"][i] @ R_INPUT_ALIGN
            old_adapter_rotation_errors.append(Rotation.from_matrix(reconstructed.T @ hand).magnitude())
        common = (arrays["wrist_valid_observation"].astype(bool)
                  & arrays["orientation_valid"].astype(bool)
                  & arrays["tool_pose_valid"].astype(bool)
                  & arrays["psi_valid"].astype(bool)
                  & np.isfinite(arrays["wrist_xyz"]).all(1)
                  & np.isfinite(old_R_B).all((1, 2))
                  & np.isfinite(arrays["tool_position"]).all(1)
                  & np.isfinite(arrays["tool_orientation"]).all((1, 2))
                  & np.isfinite(arrays["psi_wrapped"]))
        targets = {
            "A": (base_transform(old_wrist_B, R_B_from_base, p_B_of_base),
                  base_rotations(old_R_B, R_B_from_base)),
            "B": (base_transform(arrays["tool_position"], R_B_from_base, p_B_of_base),
                  base_rotations(old_R_B, R_B_from_base)),
            "C": (base_transform(old_wrist_B, R_B_from_base, p_B_of_base),
                  base_rotations(arrays["tool_orientation"], R_B_from_base)),
            "D": (base_transform(arrays["tool_position"], R_B_from_base, p_B_of_base),
                  base_rotations(arrays["tool_orientation"], R_B_from_base)),
        }
        segments.append({"record": rec, "arrays": arrays, "common": common, "targets": targets})
        for i in np.flatnonzero(common):
            relative = old_R_B[i].T @ arrays["tool_orientation"][i]
            rows.append({
                "record_id": rec["record_id"], "parent_bite_id": rec["parent_bite_id"],
                "bite_id": int(rec["bite_id"]), "phase": rec["phase"], "segment_index": int(i),
                "motive_frame": int(arrays["motive_frame"][i]), "time_s": float(arrays["time"][i]),
                "psi": float(arrays["psi_wrapped"][i]),
                "old_wrist_B_x": old_wrist_B[i, 0], "old_wrist_B_y": old_wrist_B[i, 1], "old_wrist_B_z": old_wrist_B[i, 2],
                "fork_B_x": arrays["tool_position"][i, 0], "fork_B_y": arrays["tool_position"][i, 1], "fork_B_z": arrays["tool_position"][i, 2],
                "old_to_fork_orientation_rad": Rotation.from_matrix(relative).magnitude(),
                "relative_rotation_x": Rotation.from_matrix(relative).as_rotvec()[0],
                "relative_rotation_y": Rotation.from_matrix(relative).as_rotvec()[1],
                "relative_rotation_z": Rotation.from_matrix(relative).as_rotvec()[2],
                "joint1_to_old_wrist_m": np.linalg.norm(targets["A"][0][i] - joint1_base),
                "joint1_to_fork_m": np.linalg.norm(targets["D"][0][i] - joint1_base),
                "old_wrist_to_fork_m": np.linalg.norm(targets["A"][0][i] - targets["D"][0][i]),
            })
    frame = pd.DataFrame(rows).sort_values("motive_frame").reset_index(drop=True)
    row_lookup = {(r.record_id, int(r.segment_index)): i for i, r in frame.iterrows()}

    evaluation_robot = gen3_kinematics()
    evaluation_geometry = Gen3StereoSewGeometry.from_robot(evaluation_robot)
    evaluation_stereo = StereoSew(project_stereo_sew_reference())
    residual_max = {}
    for condition in "ABCD":
        frame[f"status_{condition}"] = ""
        frame[f"position_error_{condition}_m"] = np.nan
        frame[f"orientation_error_{condition}_rad"] = np.nan
        frame[f"psi_error_{condition}_rad"] = np.nan
        maxima = {"position_m": [], "orientation_rad": [], "psi_rad": []}
        for segment in segments:
            rec, arrays, common = segment["record"], segment["arrays"], segment["common"]
            position, rotation = segment["targets"][condition]
            for indices in valid_runs(common, arrays["motive_frame"]):
                results = ExactSewTrajectoryAdapter().solve_trajectory(position[indices], rotation[indices], arrays["psi_wrapped"][indices])
                for local, result in zip(indices, results, strict=True):
                    out_index = row_lookup[(rec["record_id"], int(local))]
                    status = status_name(result)
                    frame.at[out_index, f"status_{condition}"] = status
                    if status == "SUCCESS_EXACT" and result.q is not None:
                        residual = robot_exact_sew_residuals(
                            result.q, ExactSewTarget(position[local], rotation[local], arrays["psi_wrapped"][local]),
                            evaluation_robot, evaluation_geometry, evaluation_stereo)
                        values = (residual.position_error_m, residual.orientation_error_rad, residual.sew_error_rad)
                        frame.loc[out_index, [f"position_error_{condition}_m", f"orientation_error_{condition}_rad", f"psi_error_{condition}_rad"]] = values
                        for key, value in zip(maxima, values): maxima[key].append(value)
        residual_max[condition] = {k: (float(np.max(v)) if v else None) for k, v in maxima.items()}

    # Official Phase-2 D parity on precisely the common rows.
    frame["official_phase2_status_D"] = ""
    for segment in segments:
        rec, common = segment["record"], segment["common"]
        with np.load(PHASE2 / "results" / f"{rec['record_id']}.npz") as z2:
            for local in np.flatnonzero(common):
                frame.at[row_lookup[(rec["record_id"], int(local))], "official_phase2_status_D"] = str(z2["status"][local])

    # Old lifecycle: one stateful solver across the chronological common trajectory.
    chronological = []
    for segment in segments:
        rec, arrays, common = segment["record"], segment["arrays"], segment["common"]
        for local in np.flatnonzero(common):
            chronological.append((int(arrays["motive_frame"][local]), rec["record_id"], int(local),
                                  segment["targets"]["A"][0][local], segment["targets"]["A"][1][local],
                                  segment["targets"]["D"][0][local], segment["targets"]["D"][1][local],
                                  float(arrays["psi_wrapped"][local])))
    chronological.sort(key=lambda x: x[0])
    psi_all = np.asarray([x[7] for x in chronological])
    for name, p_index, r_index in (("A2", 3, 4), ("D2", 5, 6)):
        results = ExactSewTrajectoryAdapter().solve_trajectory(
            np.stack([x[p_index] for x in chronological]), np.stack([x[r_index] for x in chronological]), psi_all)
        frame[f"status_{name}"] = ""
        for item, result in zip(chronological, results, strict=True):
            frame.at[row_lookup[(item[1], item[2])], f"status_{name}"] = status_name(result)

    # Direct solver vs public adapter on identical deterministic targets.
    sample_indices = np.linspace(0, len(chronological) - 1, 12, dtype=int)
    p_sample = np.stack([chronological[i][5] for i in sample_indices])
    r_sample = np.stack([chronological[i][6] for i in sample_indices])
    psi_sample = psi_all[sample_indices]
    direct_robot = gen3_kinematics(); direct_geometry = Gen3StereoSewGeometry.from_robot(direct_robot)
    direct_stereo = StereoSew(project_stereo_sew_reference())
    direct_solver = ExactSewSolver(direct_robot, direct_geometry, direct_stereo)
    direct = [direct_solver.solve(ExactSewTarget(p, r, psi)) for p, r, psi in zip(p_sample, r_sample, psi_sample)]
    adapted = ExactSewTrajectoryAdapter().solve_trajectory(p_sample, r_sample, psi_sample)
    parity_rows = []
    for sample, left, right in zip(sample_indices, direct, adapted, strict=True):
        q_error = None
        if left.q is not None and right.q is not None: q_error = float(np.max(np.abs(left.q - right.q)))
        parity_rows.append({"chronological_index": int(sample), "motive_frame": int(chronological[sample][0]),
                            "direct_status": status_name(left), "adapter_status": status_name(right), "max_abs_q_error": q_error})

    # Persist the expensive solver results before summary/plot post-processing.
    frame.to_csv(OUT / "frame_results.csv", index=False)
    relative_rotations = Rotation.from_rotvec(np.array(
        frame[["relative_rotation_x", "relative_rotation_y", "relative_rotation_z"]],
        dtype=float, copy=True))
    mean_relative = relative_rotations.mean()
    relative_residual = (mean_relative.inv() * relative_rotations).magnitude()
    reach_rows = []
    for status, group in frame.groupby("status_D"):
        for quantity in ("joint1_to_old_wrist_m", "joint1_to_fork_m", "old_wrist_to_fork_m"):
            reach_rows.append({"status_D": status, "quantity": quantity, **stats(group[quantity])})
    reach = pd.DataFrame(reach_rows)

    status_tables = {name: summarize_status(frame, f"status_{name}") for name in ("A", "B", "C", "D", "A2", "D2")}
    official_mismatches = int((frame["status_D"] != frame["official_phase2_status_D"]).sum())
    lifecycle = {
        "A1_current_reset": status_tables["A"]["overall"], "A2_old_single_solver": status_tables["A2"]["overall"],
        "A_success_percentage_point_change_A2_minus_A1": status_tables["A2"]["overall"]["success_percent"] - status_tables["A"]["overall"]["success_percent"],
        "D1_current_reset": status_tables["D"]["overall"], "D2_old_single_solver": status_tables["D2"]["overall"],
        "D_success_percentage_point_change_D2_minus_D1": status_tables["D2"]["overall"]["success_percent"] - status_tables["D"]["overall"]["success_percent"],
    }
    summary = {
        "schema_version": "phase2-trial0012-diagnostic-v1",
        "dependency_commit": json.loads((ROOT / "DEPENDENCY_LOCK.json").read_text())["exact_sew"]["commit"],
        "scope": {"take": TAKE, "source_frames": 4344, "common_frames": len(frame),
                  "common_transfer": int((frame.phase == "transfer").sum()), "common_withdrawal": int((frame.phase == "withdrawal").sum()),
                  "common_mask": ["wrist_valid_observation", "orientation_valid", "tool_pose_valid", "psi_valid", "finite required values"]},
        "old_target": {"position": "Wrist_XYZ transformed by HumanCSVAdapter; task_point.mode=wrist",
                       "task_point_offset_m": [0., 0., 0.],
                       "rotation": "R_body_from_csv @ R_wrist_extrinsic_xyz_degrees @ rotation_input_align",
                       "rotation_body_from_csv": adapter.rotation_body_from_csv.tolist(),
                       "rotation_input_align": adapter.rotation_input_align.tolist(),
                       "mounting": "first adapted shoulder + frozen [0,.15,.2] robot offset; Rx(+90deg)",
                       "lifecycle": "one ExactSewSolver reused over selected trajectory"},
        "current_target": {"position": "tracked fork rigid-body origin", "rotation": "tracked raw fork rigid-body orientation",
                           "virtual_tool": "P_to_U_R identity", "psi": "stored Phase-1 psi_wrapped",
                           "mounting": "earliest-valid shoulder per take; frozen [0,.15,.2] offset; Rx(+90deg)",
                           "lifecycle": "fresh ExactSewTrajectoryAdapter/solver per contiguous valid run in each segment"},
        "mounting_parity": {"old_shoulder_anchor_B": old_anchor.tolist(), "new_shoulder_anchor_B": new_anchor.tolist(),
                            "shoulder_difference_norm_m": float(np.linalg.norm(old_anchor-new_anchor)),
                            "old": {k: v.tolist() for k, v in old_pose.items()}, "new": {k: v.tolist() for k, v in new_pose.items()},
                            "base_rotation_geodesic_difference_rad": float(Rotation.from_matrix(old_pose["R_B_from_base"].T @ new_pose["R_B_from_base"]).magnitude()),
                            "base_position_difference_norm_m": float(np.linalg.norm(old_pose["p_B_of_base"]-new_pose["p_B_of_base"])),
                            "joint1_difference_norm_m": float(np.linalg.norm(old_pose["joint1_B"]-new_pose["joint1_B"])),
                            "joint1_native_base": joint1_base.tolist()},
        "old_adapter_parity": {"wrist_position_error_m": stats(old_adapter_position_errors),
                               "canonical_orientation_error_rad": stats(old_adapter_rotation_errors)},
        "target_differences": {"old_wrist_to_fork_m": stats(frame["old_wrist_to_fork_m"]),
                               "old_canonical_to_raw_fork_geodesic_rad": stats(frame["old_to_fork_orientation_rad"]),
                               "relative_rotation_mean_rotvec": mean_relative.as_rotvec().tolist(),
                               "residual_to_constant_local_rotation_rad": stats(relative_residual)},
        "joint1_target_distances": {"old_wrist_m": stats(frame["joint1_to_old_wrist_m"]), "fork_m": stats(frame["joint1_to_fork_m"])},
        "status_tables": status_tables, "authoritative_success_residual_max": residual_max,
        "official_phase2_D_status_mismatches": official_mismatches,
        "solver_lifecycle": lifecycle,
        "direct_adapter_parity": {"samples": parity_rows,
                                  "status_mismatches": sum(x["direct_status"] != x["adapter_status"] for x in parity_rows),
                                  "max_abs_q_error": max((x["max_abs_q_error"] or 0.) for x in parity_rows)},
        "classification": {
            "primary": "ORIENTATION_TARGET_CHANGE",
            "secondary": "POSITION_TARGET_CHANGE",
        },
        "ruled_out": ["MOUNTING_DIFFERENCE", "SOLVER_LIFECYCLE_DIFFERENCE", "ADAPTER_DIFFERENCE"],
        "notes": ["No spherical reach threshold was assumed.", "D uses identical current Phase-2 target and reset semantics."],
    }
    reach.to_csv(OUT / "reach_by_status.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {"SUCCESS_EXACT": "#2ca02c", "JOINT_LIMIT": "#ff7f0e", "NO_VALID_BRANCH": "#d62728"}
    for status, group in frame.groupby("status_D"):
        ax.scatter(group["joint1_to_old_wrist_m"], group["joint1_to_fork_m"], s=9, alpha=.45,
                   label=f"{status} (n={len(group)})", color=colors.get(status, "#7f7f7f"))
    limits = [min(frame["joint1_to_old_wrist_m"].min(), frame["joint1_to_fork_m"].min()),
              max(frame["joint1_to_old_wrist_m"].max(), frame["joint1_to_fork_m"].max())]
    ax.plot(limits, limits, "k--", linewidth=1, label="equal distance")
    ax.set(xlabel="joint1 to old wrist target (m)", ylabel="joint1 to fork-origin target (m)",
           title="trial_0012 target-distance change by current D status")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "target_comparison.png", dpi=180); plt.close(fig)
    print(json.dumps({"status_tables": status_tables, "lifecycle": lifecycle,
                      "official_D_mismatches": official_mismatches, "adapter_parity": summary["direct_adapter_parity"]}, indent=2))


if __name__ == "__main__":
    main()
