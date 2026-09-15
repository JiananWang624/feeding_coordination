"""Robot R1: continuity-clean replay analysis and causal RobotSmooth baseline.

R1 deliberately consumes Robot R0 artifacts.  The seven R0 strategies are never
solved again; only RobotSmooth makes new frozen Exact-SEW calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase2 import INPUT_NOT_SOLVED, SUCCESS_EXACT
from .robot_r0 import R_INPUT_ALIGN, STRATEGIES as R0_STRATEGIES, _rotation_error

CONTINUITY_THRESHOLD_RAD = 0.5
NEAR_PI_THRESHOLD_RAD = 0.95 * np.pi
ROBOT_SMOOTH = "RobotSmooth"
STRATEGIES = (*R0_STRATEGIES, ROBOT_SMOOTH)
PRIMARY_PAIRS = (("StrongLocal", "B0"), ("StrongLocal", "B2"), ("StrongLocal", "B3"),
                 ("StrongLocal", "H_star"), ("StrongLocal", "HumanGT"),
                 ("StrongLocal", ROBOT_SMOOTH), ("HumanGT", ROBOT_SMOOTH))


def comparison_pairs() -> tuple[tuple[str, str], ...]:
    """All unordered comparisons, with primary pairs in their requested direction."""
    represented = {frozenset(pair) for pair in PRIMARY_PAIRS}
    remainder = tuple((left, right) for index, left in enumerate(STRATEGIES)
                      for right in STRATEGIES[index + 1:]
                      if frozenset((left, right)) not in represented)
    return PRIMARY_PAIRS + remainder
LOCAL_OFFSETS_RAD = np.array([0., .05, -.05, .10, -.10, .20, -.20, .40, -.40])


@dataclass(frozen=True)
class RobotR1Config:
    robot_r0_output_path: str = "outputs/robot_r0"
    output_path: str = "outputs/robot_r1"
    continuity_threshold_rad: float = CONTINUITY_THRESHOLD_RAD
    bootstrap_seed: int = 20260915
    bootstrap_resamples: int = 2000

    @classmethod
    def load(cls, path: Path) -> "RobotR1Config":
        value = cls(**json.loads(Path(path).read_text()))
        if value.continuity_threshold_rad != CONTINUITY_THRESHOLD_RAD:
            raise ValueError("Robot R1 continuity threshold is frozen to 0.5 rad/frame")
        if value.bootstrap_resamples != 2000 or value.bootstrap_seed != 20260915:
            raise ValueError("Robot R1 bootstrap is frozen to seed 20260915 and 2000 resamples")
        return value


def wrap(angle: np.ndarray | float) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict-JSON values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def continuity_labels(q: np.ndarray, success: np.ndarray, motive_frame: np.ndarray,
                      run_id: np.ndarray) -> dict[str, np.ndarray]:
    """Mark the *current* frame of every >0.5 rad successful source edge."""
    n = len(success); step = np.full((n, 7), np.nan); violation = np.zeros(n, bool)
    valid_edge = np.zeros(n, bool)
    for current in range(1, n):
        previous = current - 1
        if not (success[previous] and success[current] and run_id[previous] >= 0
                and run_id[previous] == run_id[current] and motive_frame[current] == motive_frame[previous] + 1):
            continue
        step[current] = wrap(q[current] - q[previous]); valid_edge[current] = True
        violation[current] = bool(np.max(np.abs(step[current])) > CONTINUITY_THRESHOLD_RAD)
    # A violation begins a new segment and is itself usable as that segment's first frame.
    continuity_valid = np.asarray(success, bool) & ~violation
    return {"wrapped_delta_q_rad": step, "continuity_edge": valid_edge,
            "continuity_violation": violation, "continuity_valid": continuity_valid}


def continuous_segments(success: np.ndarray, continuity_violation: np.ndarray,
                        motive_frame: np.ndarray, run_id: np.ndarray) -> list[np.ndarray]:
    """Successful pieces, split before violation current frames and all source gaps."""
    segments: list[list[int]] = []; current: list[int] = []
    for index in range(len(success)):
        contiguous = bool(current and motive_frame[index] == motive_frame[current[-1]] + 1
                          and run_id[index] == run_id[current[-1]])
        if not success[index]:
            if current: segments.append(current); current = []
        elif not contiguous or continuity_violation[index]:
            if current: segments.append(current)
            current = [index]
        else:
            current.append(index)
    if current: segments.append(current)
    return [np.asarray(x, int) for x in segments]


def clean_motion(q: np.ndarray, time_s: np.ndarray, motive_frame: np.ndarray, run_id: np.ndarray,
                 success: np.ndarray, violation: np.ndarray) -> dict[str, Any]:
    """Finite differences within continuity-clean segments only."""
    n = len(q); dq = np.full((n, 7), np.nan); velocity = dq.copy(); acceleration = dq.copy(); jerk = dq.copy()
    segments = continuous_segments(success, violation, motive_frame, run_id)
    for indices in segments:
        for prev, cur in zip(indices[:-1], indices[1:]):
            dt = time_s[cur] - time_s[prev]
            if np.isfinite(dt) and dt > 0:
                dq[cur] = wrap(q[cur] - q[prev]); velocity[cur] = dq[cur] / dt
        for prev, cur in zip(indices[:-1], indices[1:]):
            dt = time_s[cur] - time_s[prev]
            if np.isfinite(dt) and dt > 0 and np.isfinite(velocity[[prev, cur]]).all():
                acceleration[cur] = (velocity[cur] - velocity[prev]) / dt
        for prev, cur in zip(indices[:-1], indices[1:]):
            dt = time_s[cur] - time_s[prev]
            if np.isfinite(dt) and dt > 0 and np.isfinite(acceleration[[prev, cur]]).all():
                jerk[cur] = (acceleration[cur] - acceleration[prev]) / dt
    return {"continuous_segments": segments, "clean_wrapped_delta_q_rad": dq,
            "clean_joint_velocity_rad_s": velocity, "clean_joint_acceleration_rad_s2": acceleration,
            "clean_joint_jerk_rad_s3": jerk}


def _diag(result: Any) -> tuple[str, str, str, str, float]:
    diagnostics = result.diagnostics; status = getattr(result.status, "value", str(result.status))
    raw = diagnostics.to_dict(); meta = raw.get("metadata", {})
    # `is not None` preserves a legitimate measured zero solve time.
    elapsed = diagnostics.solve_time_ms
    return status, json.dumps(raw, sort_keys=True, default=str), str(diagnostics.branch_id or ""), str(meta.get("search_branch", "")), float(elapsed) if elapsed is not None else np.nan


def _candidate_key(q: np.ndarray, previous_q: np.ndarray, margin: float, psi: float, previous_psi: float,
                   candidate_order: int) -> tuple[float, float, float, int]:
    return (float(np.linalg.norm(wrap(q - previous_q))), -float(margin),
            float(abs(wrap(psi - previous_psi))), candidate_order)


def robot_smooth_run(positions: np.ndarray, rotations: np.ndarray, psi0: float,
                     initial_q: np.ndarray | None, initial_status: str, solver: Callable[..., Any],
                     margin_fn: Callable[[np.ndarray], float]) -> dict[str, Any]:
    """Causal candidate search. `solver` receives (p, R, psi, q_previous)."""
    n = len(positions); status = np.full(n, INPUT_NOT_SOLVED, dtype="<U128"); evaluation = status.copy(); q = np.full((n, 7), np.nan)
    psi = np.full(n, np.nan); mode = np.full(n, "NOT_EVALUATED_AFTER_FAILURE", dtype="<U128")
    evaluated = np.zeros(n, int); chosen_delta = np.full(n, np.nan); candidates_status = ["[]" for _ in range(n)]
    local_statuses = ["[]" for _ in range(n)]; global_statuses = ["[]" for _ in range(n)]
    branch = [""] * n; search_branch = [""] * n; diagnostics = ["{}"] * n
    solver_messages = [""] * n; solve_ms = np.full(n, np.nan)
    # R0 owns the canonical initial target.  Reuse it, never re-solve it.
    status[0] = initial_status; evaluation[0] = initial_status; psi[0] = psi0; mode[0] = "INITIAL"
    if initial_status != SUCCESS_EXACT or initial_q is None or not np.isfinite(initial_q).all():
        return {"solver_status": status, "evaluation_status": evaluation, "q": q, "strategy_psi": psi, "selection_status": mode,
                "search_mode": mode, "candidates_evaluated": evaluated, "chosen_abs_delta_psi": chosen_delta,
                "candidate_statuses_json": candidates_status, "local_candidate_statuses_json": local_statuses, "global_candidate_statuses_json": global_statuses, "branch_id": branch, "search_branch": search_branch, "solver_message": solver_messages,
                "solver_diagnostics_json": diagnostics, "solve_time_ms": solve_ms}
    q[0] = initial_q; previous_q = initial_q.copy(); previous_psi = float(psi0); terminated = False
    # Frozen calls have no mutable solver state; each gets an independent q copy.
    # map preserves the prescribed candidate order despite parallel execution.
    with ThreadPoolExecutor(max_workers=9) as executor:
      for index in range(1, n):
        if terminated:
            continue
        def evaluate(values: np.ndarray, label: str) -> list[tuple[tuple[float, float, float, int], Any, float]]:
            found = []; attempted = []
            candidate_psi_values = [float(wrap(value)) for value in values]
            def invoke(candidate_psi: float) -> Any:
                return solver(positions[index], rotations[index], candidate_psi, previous_q.copy())
            for order, (candidate_psi, result) in enumerate(zip(candidate_psi_values, executor.map(invoke, candidate_psi_values))):
                candidate_status, _, _, _, _ = _diag(result); attempted.append(candidate_status)
                if candidate_status == SUCCESS_EXACT and result.q is not None:
                    candidate_q = np.asarray(result.q, float)
                    if candidate_q.shape != (7,) or not np.isfinite(candidate_q).all():
                        raise ValueError("Exact-SEW returned SUCCESS_EXACT with an invalid 7-joint solution")
                    margin = float(margin_fn(candidate_q))
                    if not np.isfinite(margin):
                        raise ValueError("joint-limit margin is non-finite for a successful Exact-SEW result")
                    found.append((_candidate_key(candidate_q, previous_q, margin, candidate_psi, previous_psi, order), result, candidate_psi))
            candidates_status[index] = json.dumps(attempted); evaluated[index] += len(values)
            if label == "LOCAL_SEARCH": local_statuses[index] = candidates_status[index]
            else: global_statuses[index] = candidates_status[index]
            return found
        local = evaluate(previous_psi + LOCAL_OFFSETS_RAD, "LOCAL_SEARCH")
        choices = local; selected_mode = "LOCAL_SEARCH"
        if not choices:
            choices = evaluate(np.linspace(-np.pi, np.pi, 32, endpoint=False), "GLOBAL_RECOVERY")
            selected_mode = "GLOBAL_RECOVERY"
        if not choices:
            # No frozen result was selected: do not manufacture a frozen status.
            evaluation[index] = "NO_FEASIBLE_PSI"; mode[index] = "NO_FEASIBLE_PSI"; terminated = True; continue
        _, result, selected_psi = min(choices, key=lambda item: item[0])
        result_status, raw_diag, bid, sb, elapsed = _diag(result)
        status[index] = result_status; evaluation[index] = result_status; q[index] = result.q; psi[index] = selected_psi; mode[index] = selected_mode
        chosen_delta[index] = abs(wrap(selected_psi - previous_psi)); branch[index] = bid; search_branch[index] = sb
        diagnostics[index] = raw_diag; solver_messages[index] = getattr(result, "message", None) or ""
        solve_ms[index] = elapsed; previous_q = q[index].copy(); previous_psi = selected_psi
    return {"solver_status": status, "evaluation_status": evaluation, "q": q, "strategy_psi": psi, "selection_status": mode,
            "search_mode": mode, "candidates_evaluated": evaluated, "chosen_abs_delta_psi": chosen_delta,
            "candidate_statuses_json": candidates_status, "local_candidate_statuses_json": local_statuses, "global_candidate_statuses_json": global_statuses, "branch_id": branch, "search_branch": search_branch, "solver_message": solver_messages,
            "solver_diagnostics_json": diagnostics, "solve_time_ms": solve_ms}


def continuity_events(record: dict[str, Any], strategy: str, q: np.ndarray, psi: np.ndarray, success: np.ndarray,
                      motive_frame: np.ndarray, run_id: np.ndarray, branch: np.ndarray, search_branch: np.ndarray,
                      diagnostics: np.ndarray, tool_position: np.ndarray, tool_rotation: np.ndarray) -> list[dict[str, Any]]:
    labels = continuity_labels(q, success, motive_frame, run_id); rows = []
    for cur in np.flatnonzero(labels["continuity_violation"]):
        prev = cur - 1; delta = labels["wrapped_delta_q_rad"][cur]; max_joint = int(np.argmax(np.abs(delta)))
        try: fallback = bool(json.loads(str(diagnostics[cur])).get("metadata", {}).get("fallback_used", False))
        except (ValueError, TypeError): fallback = False
        sb_change = bool(search_branch[prev] and search_branch[cur] and search_branch[prev] != search_branch[cur])
        bid_change = bool(branch[prev] and branch[cur] and branch[prev] != branch[cur])
        categories = []
        if sb_change: categories.append("SEARCH_BRANCH_CHANGE")
        if bid_change: categories.append("BRANCH_ID_CHANGE")
        if not (sb_change or bid_change):
            categories.append("GLOBAL_RECOVERY_WITHOUT_RECORDED_BRANCH_CHANGE" if fallback else "UNCLASSIFIED")
        dpsi = float(wrap(psi[cur] - psi[prev]))
        if abs(dpsi) > .5: categories.append("LARGE_PSI_STEP")
        translation = float(np.linalg.norm(tool_position[cur] - tool_position[prev]))
        rotation = _rotation_error(tool_rotation[cur], tool_rotation[prev])
        if rotation > .5: categories.append("LARGE_TOOL_STEP")
        rows.append({"take": record["take"], "record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"],
                     "phase": str(record["phase"][cur]), "strategy": strategy, "previous_index": int(prev), "current_index": int(cur),
                     "previous_motive_frame": int(motive_frame[prev]), "current_motive_frame": int(motive_frame[cur]),
                     "max_wrapped_joint_step_rad": float(np.max(np.abs(delta))), "joint_index": max_joint,
                     "psi_previous": float(psi[prev]), "psi_current": float(psi[cur]), "wrapped_delta_psi": dpsi,
                     "tool_translation_step_m": translation, "tool_rotation_geodesic_step_rad": rotation,
                     "search_branch_previous": str(search_branch[prev]), "search_branch_current": str(search_branch[cur]),
                     "branch_id_previous": str(branch[prev]), "branch_id_current": str(branch[cur]),
                     "search_branch_changed": sb_change, "branch_id_changed": bid_change,
                     "near_pi_jump": bool(np.max(np.abs(delta)) >= NEAR_PI_THRESHOLD_RAD), "categories": "|".join(categories),
                     "primary_category": categories[0]})
    return rows


def _distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x): return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    return {"count": int(len(x)), "min": float(x.min()), "mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _metrics(q: np.ndarray, margin: np.ndarray, time_s: np.ndarray, frames: np.ndarray, run_id: np.ndarray,
             success: np.ndarray, violation: np.ndarray, branch: np.ndarray, search_branch: np.ndarray | None = None) -> dict[str, Any]:
    clean = clean_motion(q, time_s, frames, run_id, success, violation)
    out = {"solver_success_frames": int(success.sum()), "continuity_violations": int(violation.sum()),
           "continuous_frame_percent": float(100 * (success & ~violation).sum() / max(1, success.sum())),
           "continuous_segments": int(len(clean["continuous_segments"])),
           "longest_continuous_segment_frames": max((len(x) for x in clean["continuous_segments"]), default=0)}
    for name, array in (("wrapped_joint_travel_rad", clean["clean_wrapped_delta_q_rad"]),
                        ("joint_velocity_rad_s", clean["clean_joint_velocity_rad_s"]),
                        ("joint_acceleration_rad_s2", clean["clean_joint_acceleration_rad_s2"]),
                        ("joint_jerk_rad_s3", clean["clean_joint_jerk_rad_s3"])):
        vals = np.abs(array[np.isfinite(array)]); out[f"{name}_distribution"] = _distribution(vals)
        out[f"rms_{name}"] = float(np.sqrt(np.mean(vals**2))) if len(vals) else None
    out["total_wrapped_joint_travel_rad"] = float(np.nansum(np.abs(clean["clean_wrapped_delta_q_rad"])))
    out["joint_limit_margin_rad"] = _distribution(margin[success])
    out["clean_branch_changes"] = int(sum(bool(branch[b] and branch[a] and branch[b] != branch[a])
                                           for seg in clean["continuous_segments"] for b, a in zip(seg[:-1], seg[1:])))
    search_branch = branch if search_branch is None else search_branch
    out["clean_search_branch_changes"] = int(sum(bool(search_branch[b] and search_branch[a] and search_branch[b] != search_branch[a])
                                                  for seg in clean["continuous_segments"] for b, a in zip(seg[:-1], seg[1:])))
    return out | clean


def paired_bootstrap(parent_values: pd.DataFrame, metric: str, seed: int = 20260915,
                     resamples: int = 2000) -> dict[str, Any]:
    """Paired resampling units are parent bites, never frames."""
    values = parent_values[["parent_bite_id", metric]].dropna().groupby("parent_bite_id")[metric].mean().to_numpy(float)
    if not len(values): return {"units": 0, "difference": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed); draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {"units": int(len(values)), "difference": float(values.mean()), "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))}


def _default_solver() -> tuple[Callable[..., Any], Callable[[np.ndarray], float]]:
    from sew_mimic.common import ExactSewTarget, joint_limit_margin
    from sew_mimic.exact import solve_exact_sew
    from sew_mimic.kinematics import gen3_kinematics
    from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
    robot = gen3_kinematics(); geometry = Gen3StereoSewGeometry.from_robot(robot); stereo = StereoSew(project_stereo_sew_reference())
    return (lambda p, r, psi, qprev: solve_exact_sew(ExactSewTarget(p, r, psi), robot, geometry, stereo,
                                                       branch_policy="continuous", q_previous=qprev),
            lambda q: joint_limit_margin(q, robot))


def process_robot_r1(robot_r0_dir: Path, output_dir: Path, cfg: RobotR1Config,
                     candidate_solver: Callable[..., Any] | None = None,
                     margin_fn: Callable[[np.ndarray], float] | None = None) -> dict[str, Any]:
    robot_r0_dir = Path(robot_r0_dir); output_dir = Path(output_dir); out_results = output_dir / "results"; plots = output_dir / "plots"
    out_results.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    manifest0_path = robot_r0_dir / "manifest.json"; manifest0 = json.loads(manifest0_path.read_text())
    if manifest0.get("schema_version") != "robot-r0-v1": raise ValueError("Robot R1 requires frozen Robot R0 v1 artifacts")
    transform = manifest0.get("virtual_simulation_tool_transform", {})
    if (transform.get("external_R_robot_align_applied") is not False
            or not np.array_equal(np.asarray(transform.get("translation_P_to_U_m"), float), np.zeros(3))
            or not np.array_equal(np.asarray(transform.get("rotation_P_to_U"), float), R_INPUT_ALIGN.T)
            or not np.array_equal(np.asarray(transform.get("R_input_align"), float), R_INPUT_ALIGN)):
        raise ValueError("Robot R0 virtual-tool provenance does not match R1")
    production_solver = candidate_solver is None
    if candidate_solver is None: candidate_solver, default_margin = _default_solver(); margin_fn = margin_fn or default_margin
    if margin_fn is None: raise ValueError("margin_fn required with custom candidate_solver")
    all_items = []; event_rows = []; segment_rows = []
    for entry in manifest0["records"]:
        with np.load(robot_r0_dir / entry["file"], allow_pickle=False) as data:
            raw = {key: data[key].copy() for key in data.files}
        n = len(raw["time_s"])
        record = {"take": str(raw["take"][0]), "record_id": str(raw["record_id"][0]), "parent_bite_id": str(raw["parent_bite_id"][0]), "phase": raw["phase"]}
        success0 = raw["B0_solver_status"] == SUCCESS_EXACT; run_id = raw["run_id"]; frames = raw["motive_frame"]
        # Solve each original source run independently, beginning from R0's canonical first frame.
        # Sized generously before assignment; frozen diagnostics are JSON and must
        # not be silently shortened by the small R0 display dtypes.
        smooth_parts: dict[str, Any] = {"solver_status": np.full(n, INPUT_NOT_SOLVED, dtype=object), "evaluation_status": np.full(n, INPUT_NOT_SOLVED, dtype=object), "q": np.full((n,7), np.nan),
            "strategy_psi": np.full(n, np.nan), "search_mode": np.full(n,"NOT_EVALUATED_AFTER_FAILURE",dtype=object),
            "selection_status": np.full(n,"NOT_EVALUATED_AFTER_FAILURE",dtype=object), "candidates_evaluated": np.zeros(n,int),
            "chosen_abs_delta_psi": np.full(n,np.nan), "candidate_statuses_json": np.full(n,"[]",dtype=object),
            "local_candidate_statuses_json": np.full(n,"[]",dtype=object), "global_candidate_statuses_json": np.full(n,"[]",dtype=object),
            "branch_id": np.full(n,"",dtype=object), "search_branch": np.full(n,"",dtype=object),
            "solver_diagnostics_json": np.full(n,"{}",dtype=object), "solver_message": np.full(n,"",dtype=object), "solve_time_ms": np.full(n,np.nan)}
        for rid in sorted(set(run_id[run_id >= 0])):
            idx = np.flatnonzero(run_id == rid); first = idx[0]
            part = robot_smooth_run(raw["desired_pinch_position_base"][idx], raw["desired_pinch_rotation_base"][idx],
                                    float(raw["B0_strategy_psi"][first]), raw["B0_q"][first] if success0[first] else None,
                                    str(raw["B0_solver_status"][first]), candidate_solver, margin_fn)
            for key, value in part.items(): smooth_parts[key][idx] = value
            initial_psis=np.array([raw[f"{strategy}_strategy_psi"][first] for strategy in R0_STRATEGIES])
            if (not np.array_equal(initial_psis,np.full(len(initial_psis),initial_psis[0]))
                    or smooth_parts["strategy_psi"][first] != initial_psis[0]
                    or smooth_parts["solver_status"][first] != raw["B0_solver_status"][first]
                    or (success0[first] and not np.allclose(smooth_parts["q"][first], raw["B0_q"][first], atol=1e-10))):
                raise RuntimeError("RobotSmooth first frame differs from R0 canonical target/q")
        smooth_parts["joint_limit_margin_rad"] = np.array([margin_fn(x) if np.isfinite(x).all() else np.nan for x in smooth_parts["q"]])
        # The initial result is literally R0's canonical evaluation.  New selected
        # results are evaluated below with the same frozen authoritative evaluator.
        smooth_parts["initial_reused_verified"] = np.zeros(n, bool)
        for rid in sorted(set(run_id[run_id >= 0])):
            first = np.flatnonzero(run_id == rid)[0]
            smooth_parts["initial_reused_verified"][first] = True
            for key in ("branch_id","search_branch","solver_diagnostics_json","solver_message","solve_time_ms"):
                smooth_parts[key][first] = raw[f"B0_{key}"][first]
        if production_solver:
            from sew_mimic.common import ExactSewTarget
            from sew_mimic.exact.residuals import robot_exact_sew_residuals
            from sew_mimic.kinematics import gen3_kinematics
            from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
            eval_robot=gen3_kinematics(); eval_geometry=Gen3StereoSewGeometry.from_robot(eval_robot); eval_stereo=StereoSew(project_stereo_sew_reference())
            for name,shape in (("actual_pinch_position_base",(n,3)),("actual_pinch_rotation_base",(n,3,3)),("actual_virtual_U_position_base",(n,3)),("actual_virtual_U_rotation_base",(n,3,3))): smooth_parts[name]=np.full(shape,np.nan)
            for name in ("actual_psi","aligned_pinch_position_error_m","aligned_pinch_orientation_error_rad","psi_error_rad","virtual_utensil_position_error_m","virtual_utensil_orientation_error_rad"): smooth_parts[name]=np.full(n,np.nan)
            for index in np.flatnonzero(smooth_parts["evaluation_status"] == SUCCESS_EXACT):
                target=ExactSewTarget(raw["desired_pinch_position_base"][index],raw["desired_pinch_rotation_base"][index],smooth_parts["strategy_psi"][index])
                residual=robot_exact_sew_residuals(smooth_parts["q"][index],target,eval_robot,eval_geometry,eval_stereo)
                smooth_parts["actual_pinch_position_base"][index]=residual.actual_position; smooth_parts["actual_pinch_rotation_base"][index]=residual.actual_rotation
                smooth_parts["actual_psi"][index]=np.nan if residual.actual_psi is None else residual.actual_psi
                smooth_parts["aligned_pinch_position_error_m"][index]=residual.position_error_m; smooth_parts["aligned_pinch_orientation_error_rad"][index]=residual.orientation_error_rad; smooth_parts["psi_error_rad"][index]=np.nan if residual.sew_error_rad is None else residual.sew_error_rad
                smooth_parts["actual_virtual_U_position_base"][index]=residual.actual_position; actual_u_rotation=residual.actual_rotation @ R_INPUT_ALIGN.T; smooth_parts["actual_virtual_U_rotation_base"][index]=actual_u_rotation
                smooth_parts["virtual_utensil_position_error_m"][index]=np.linalg.norm(residual.actual_position-raw["desired_U_position_base"][index]); smooth_parts["virtual_utensil_orientation_error_rad"][index]=_rotation_error(actual_u_rotation,raw["desired_U_rotation_base"][index])
        payloads = {s: {key[len(s)+1:]: raw[key] for key in raw if key.startswith(s + "_")} for s in R0_STRATEGIES}
        for payload in payloads.values():
            payload["evaluation_status"] = payload["solver_status"].copy()
        payloads[ROBOT_SMOOTH] = smooth_parts
        for strategy, payload in payloads.items():
            ok = payload["evaluation_status"] == SUCCESS_EXACT
            labels = continuity_labels(payload["q"], ok, frames, run_id); payload.update(labels)
            margin = payload.get("joint_limit_margin_rad", np.full(n,np.nan))
            payload["clean_metrics"] = _metrics(payload["q"], margin, raw["time_s"], frames, run_id, ok, labels["continuity_violation"], payload.get("branch_id", np.full(n,"")), payload.get("search_branch", np.full(n,"")))
            clean=payload["clean_metrics"]
            payload["clean_wrapped_delta_q_rad"] = clean["clean_wrapped_delta_q_rad"]
            payload["clean_joint_velocity_rad_s"] = clean["clean_joint_velocity_rad_s"]
            payload["clean_joint_acceleration_rad_s2"] = clean["clean_joint_acceleration_rad_s2"]
            payload["clean_joint_jerk_rad_s3"] = clean["clean_joint_jerk_rad_s3"]
            segment_id=np.full(n,-1,int)
            for sid, segment in enumerate(clean["continuous_segments"]): segment_id[segment]=sid
            payload["continuous_segment_id"] = segment_id
            event_rows.extend(continuity_events(record, strategy, payload["q"], payload["strategy_psi"], ok, frames, run_id,
                                                np.asarray(payload.get("branch_id", [""]*n)), np.asarray(payload.get("search_branch", [""]*n)),
                                                np.asarray(payload.get("solver_diagnostics_json", ["{}"]*n)), raw["desired_U_position_base"], raw["desired_U_rotation_base"]))
            for sid, segment in enumerate(payload["clean_metrics"]["continuous_segments"]):
                local_q=payload["q"]; local_time=raw["time_s"]; local_margin=margin[segment]
                local_motion=clean_motion(local_q,local_time,frames,run_id,np.isin(np.arange(n),segment),np.zeros(n,bool))
                margin_stats = _distribution(local_margin)
                row={"take": record["take"], "record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"], "phase": str(raw["phase"][segment[0]]), "strategy": strategy, "segment_id": sid, "frames":len(segment), "start_index":int(segment[0]), "end_index":int(segment[-1]), "start_time_s":float(local_time[segment[0]]), "end_time_s":float(local_time[segment[-1]]), "start_motive_frame":int(frames[segment[0]]), "end_motive_frame":int(frames[segment[-1]]), "joint_limit_margin_min_rad":margin_stats["min"], "joint_limit_margin_mean_rad":margin_stats["mean"], "joint_limit_margin_p95_rad":margin_stats["p95"], "joint_limit_margin_max_rad":margin_stats["max"]}
                for label,array in (("travel",local_motion["clean_wrapped_delta_q_rad"]),("velocity",local_motion["clean_joint_velocity_rad_s"]),("acceleration",local_motion["clean_joint_acceleration_rad_s2"]),("jerk",local_motion["clean_joint_jerk_rad_s3"])):
                    values=np.abs(array[np.isfinite(array)]); row[f"{label}_total_rad"] = float(values.sum()) if label=="travel" else None; row[f"{label}_rms"] = float(np.sqrt(np.mean(values**2))) if len(values) else None; row[f"{label}_p95"] = float(np.percentile(values,95)) if len(values) else None; row[f"{label}_max"] = float(values.max()) if len(values) else None
                segment_rows.append(row)
        for strategy, payload in payloads.items():
            for key, value in payload.items():
                if key != "clean_metrics": raw[f"{strategy}_{key}"] = value
        # NPZ cannot be loaded safely with pickle; convert object string columns
        # only at serialization time, after dynamic-width accumulation.
        for key, value in tuple(raw.items()):
            if isinstance(value, np.ndarray) and value.dtype == object:
                if all(isinstance(x, str) for x in value.ravel()): raw[key] = np.asarray(value, dtype=f"<U{max(1,max(map(len,value.ravel())))}")
                else: raise ValueError(f"non-string object result field {key}")
        np.savez_compressed(out_results / f"{record['record_id']}.npz", **raw)
        # Keep only arrays required by aggregate statistics.  A full R0 record is
        # about 23 MB after loading because of wide diagnostic strings; retaining
        # all 254 records would consume several GB without affecting any metric.
        aggregate_raw = {key: raw[key] for key in
                         ("time_s", "motive_frame", "run_id", "common_source_valid", "phase")}
        aggregate_keys = ("q", "solver_status", "evaluation_status", "continuity_violation",
                          "continuity_valid", "joint_limit_margin_rad", "branch_id", "search_branch",
                          "clean_metrics", "selection_status", "search_mode", "candidates_evaluated",
                          "chosen_abs_delta_psi", "strategy_psi", "local_candidate_statuses_json",
                          "global_candidate_statuses_json", "virtual_utensil_position_error_m",
                          "virtual_utensil_orientation_error_rad")
        aggregate_payloads = {strategy: {key: payload[key] for key in aggregate_keys if key in payload}
                              for strategy, payload in payloads.items()}
        all_items.append({"record": record, "raw": aggregate_raw, "payloads": aggregate_payloads})
    events = pd.DataFrame(event_rows); events.to_csv(output_dir / "continuity_events.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(output_dir / "continuous_segment_metrics.csv", index=False)
    strategy_rows=[]
    for grouping, groups in (("overall", {"overall": all_items}), ("phase", {p:[x for x in all_items] for p in ("transfer","withdrawal")}), ("take", {t:[x for x in all_items if x["record"]["take"]==t] for t in sorted({x["record"]["take"] for x in all_items})})):
        for group, items in groups.items():
            for strategy in STRATEGIES:
                selected=[]
                for item in items:
                    source=item["raw"]["common_source_valid"].astype(bool)
                    mask = source if grouping != "phase" else source & (item["raw"]["phase"] == group)
                    p=item["payloads"][strategy]; ok=p["evaluation_status"] == SUCCESS_EXACT
                    selected.append((item,mask,p,ok))
                status=np.concatenate([p["solver_status"][m] for _,m,p,_ in selected]); success=np.concatenate([ok[m] for _,m,_,ok in selected])
                violations=sum(int((p["continuity_violation"] & m).sum()) for _,m,p,_ in selected)
                evaluation=np.concatenate([p["evaluation_status"][m] for _,m,p,_ in selected])
                counts={key:int((evaluation == key).sum()) for key in (SUCCESS_EXACT,"JOINT_LIMIT","NO_VALID_BRANCH","NO_FEASIBLE_PSI",INPUT_NOT_SOLVED)}
                counts["other_status"] = int(len(evaluation)-sum(counts.values()))
                continuous=int(sum(((p["continuity_valid"] & m)).sum() for _,m,p,_ in selected))
                search={}
                if strategy == ROBOT_SMOOTH:
                    modes=np.concatenate([p["search_mode"][m] for _,m,p,_ in selected]); candidates=np.concatenate([p["candidates_evaluated"][m] for _,m,p,_ in selected]); deltas=np.concatenate([p["chosen_abs_delta_psi"][m] for _,m,p,_ in selected])
                    attempted_search=np.isin(modes,["LOCAL_SEARCH","GLOBAL_RECOVERY","NO_FEASIBLE_PSI"])
                    search_attempts=int(attempted_search.sum()); local_selected=int((modes=="LOCAL_SEARCH").sum()); global_selected=int((modes=="GLOBAL_RECOVERY").sum())
                    psi_travel=0.; candidate_counts={SUCCESS_EXACT:0,"JOINT_LIMIT":0,"NO_VALID_BRANCH":0,"other":0}
                    for item,m,p,ok in selected:
                        psi_values=p["strategy_psi"]
                        for current in range(1,len(m)):
                            previous=current-1
                            if (m[previous] and m[current] and ok[previous] and ok[current]
                                    and item["raw"]["run_id"][previous] == item["raw"]["run_id"][current]
                                    and item["raw"]["motive_frame"][current] == item["raw"]["motive_frame"][previous]+1):
                                psi_travel += float(abs(wrap(psi_values[current]-psi_values[previous])))
                        for field in ("local_candidate_statuses_json","global_candidate_statuses_json"):
                            for encoded in np.asarray(p[field])[m]:
                                for candidate_status in json.loads(str(encoded)):
                                    key=candidate_status if candidate_status in candidate_counts else "other"
                                    candidate_counts[key] += 1
                    search={"search_attempted_frames":search_attempts,"local_selected_frames":local_selected,
                            "local_candidate_selected_percent":float(100*local_selected/max(1,search_attempts)),
                            "global_recovery_frames":global_selected,
                            "global_recovery_percent":float(100*global_selected/max(1,search_attempts)),
                            "no_feasible_psi_frames":int((modes=="NO_FEASIBLE_PSI").sum()),
                            "average_candidates_evaluated_among_search_attempts":float(candidates[attempted_search].mean()) if attempted_search.any() else None,
                            **{f"chosen_abs_delta_psi_rad_{key}":value for key,value in _distribution(deltas).items()},
                            "total_psi_travel_rad":psi_travel,
                            "candidate_SUCCESS_EXACT":candidate_counts[SUCCESS_EXACT],
                            "candidate_JOINT_LIMIT":candidate_counts["JOINT_LIMIT"],
                            "candidate_NO_VALID_BRANCH":candidate_counts["NO_VALID_BRANCH"],
                            "candidate_other_status":candidate_counts["other"]}
                strategy_rows.append({"scope":"clean_motion", "grouping":grouping,"group":group,"strategy":strategy,"source_valid_frames":int(len(status)),"attempted_frames":int(sum((p["selection_status"][m] != "NOT_EVALUATED_AFTER_FAILURE").sum() if strategy==ROBOT_SMOOTH else m.sum() for _,m,p,_ in selected)),"SUCCESS_EXACT":int(success.sum()),"success_percent":float(100*success.mean()),"continuous_frame_percent":float(100*continuous/max(1,success.sum())),"continuity_violations":violations,**counts,**search, **{k:v for k,v in _metrics_aggregate(selected).items()}})
    metrics=pd.DataFrame(strategy_rows); metrics.to_csv(output_dir / "strategy_metrics.csv",index=False)
    # Recompute public aggregates from raw common-clean derivatives; summing
    # record-level RMS/minimum statistics is mathematically invalid.
    aggregate_rows=[]; bite_rows=[]
    takes=sorted({x["record"]["take"] for x in all_items})
    for left,right in comparison_pairs():
        aggregate_rows.append(_pooled_pair_row(all_items,left,right,[x["raw"]["common_source_valid"].astype(bool) for x in all_items],"overall","overall"))
        for phase in ("transfer","withdrawal"):
            aggregate_rows.append(_pooled_pair_row(all_items,left,right,[x["raw"]["common_source_valid"].astype(bool)&(x["raw"]["phase"]==phase) for x in all_items],"phase",phase))
        for take in takes:
            selected=[x for x in all_items if x["record"]["take"]==take]
            aggregate_rows.append(_pooled_pair_row(selected,left,right,[x["raw"]["common_source_valid"].astype(bool) for x in selected],"take",take))
    for left,right in PRIMARY_PAIRS:
        for bite in sorted({x["record"]["parent_bite_id"] for x in all_items}):
            selected=[x for x in all_items if x["record"]["parent_bite_id"]==bite]
            row=_pooled_pair_row(selected,left,right,[x["raw"]["common_source_valid"].astype(bool) for x in selected],"bite",bite)
            row["parent_bite_id"]=bite; bite_rows.append(row)
    pairs=pd.DataFrame(aggregate_rows)
    pairs.to_csv(output_dir / "pairwise_metrics.csv",index=False)
    bootstrap={}
    bite_detail=pd.DataFrame(bite_rows)
    for a,b in PRIMARY_PAIRS:
        subset=bite_detail[(bite_detail.strategy_a==a)&(bite_detail.strategy_b==b)]
        bootstrap[f"{a}_vs_{b}"]={metric:paired_bootstrap(subset.rename(columns={metric:"value"}),"value") for metric in ("travel_difference_a_minus_b","rms_velocity_difference_a_minus_b","rms_acceleration_difference_a_minus_b","rms_jerk_difference_a_minus_b","min_margin_difference_a_minus_b_positive_is_higher","violations_difference_a_minus_b")}
    if len(events):
        exploded=events.assign(category=events["categories"].str.split("|")).explode("category")
        event_summary=exploded.groupby(["strategy","category"],as_index=False).size().rename(columns={"size":"count"}).to_dict("records")
        near_pi_by_strategy=(events.groupby("strategy").apply(
            lambda frame: pd.Series({"near_pi_events":int(frame.near_pi_jump.sum()),
                "near_pi_with_branch_or_search_change":int((frame.near_pi_jump & (frame.branch_id_changed|frame.search_branch_changed)).sum())}),
            include_groups=False).reset_index().to_dict("records"))
    else:
        event_summary=[]; near_pi_by_strategy=[]
    smooth_payloads=[item["payloads"][ROBOT_SMOOTH] for item in all_items]
    task_equivalence={"desired_tool_trajectory_reused_from_robot_r0":True,"authoritative_evaluation_performed":production_solver}
    if production_solver:
        for field in ("virtual_utensil_position_error_m","virtual_utensil_orientation_error_rad"):
            values=np.concatenate([payload[field] for payload in smooth_payloads])
            task_equivalence[field]=_distribution(values)
    overall_records=metrics.query("grouping == 'overall'").to_dict("records")
    phase_records={group:metrics[(metrics.grouping=="phase")&(metrics.group==group)].to_dict("records")
                   for group in ("transfer","withdrawal")}
    summary={"schema_version":"robot-r1-v1","robot_r0_manifest_sha256":hashlib.sha256(manifest0_path.read_bytes()).hexdigest(),
             "continuity_threshold_rad":CONTINUITY_THRESHOLD_RAD,"strategies":list(STRATEGIES),
             "overall":overall_records,"per_phase":phase_records,
             "per_take":{group:metrics[(metrics.grouping=="take")&(metrics.group==group)].to_dict("records")
                         for group in metrics.loc[metrics.grouping=="take","group"].unique()},
             "continuity_events":{"categories":event_summary,"near_pi_events":int(events.near_pi_jump.sum()) if len(events) else 0,
                 "near_pi_with_branch_or_search_change":int((events.near_pi_jump & (events.branch_id_changed|events.search_branch_changed)).sum()) if len(events) else 0,
                 "near_pi_definition":f"max wrapped joint step >= 0.95*pi ({NEAR_PI_THRESHOLD_RAD:.12g} rad)",
                 "near_pi_by_strategy":near_pi_by_strategy},
             "robot_smooth_task_equivalence":task_equivalence,
             "primary_pairwise_overall":[row for row in pairs.query("grouping == 'overall'").to_dict("records")
                                          if (row["strategy_a"],row["strategy_b"]) in PRIMARY_PAIRS],
             "human_gt_vs_learned_clean_metrics":[row for row in overall_records
                                                   if row["strategy"] in ("HumanGT","StrongLocal","B2","B3","H_star")],
             "transfer_vs_withdrawal_clean_metrics":phase_records,
             "paired_bootstrap":bootstrap,"bootstrap_unit":"parent_bite (paired; never frames)",
             "scope_exclusions":["calibration","collision checking","retiming","human learning"]}
    (output_dir/"summary.json").write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False))
    (output_dir/"manifest.json").write_text(json.dumps({"schema_version":"robot-r1-v1","config":asdict(cfg),"robot_r0_manifest":str(manifest0_path),"robot_r0_manifest_sha256":summary["robot_r0_manifest_sha256"],"records":manifest0["records"]},indent=2))
    _plots(metrics, events, plots); return summary


def _metrics_aggregate(selected: list[tuple[Any,...]]) -> dict[str,Any]:
    # Aggregate clean per-frame arrays; no derivative can cross a stored split.
    names=("clean_wrapped_delta_q_rad","clean_joint_velocity_rad_s","clean_joint_acceleration_rad_s2","clean_joint_jerk_rad_s3")
    out={}
    for name in names:
        values=np.concatenate([p["clean_metrics"][name][m].ravel() for _,m,p,_ in selected]); values=np.abs(values[np.isfinite(values)])
        out[name+"_support_values"]=int(len(values)); out[name+"_p95"]=float(np.percentile(values,95)) if len(values) else None; out[name+"_max"]=float(values.max()) if len(values) else None; out[name+"_rms"]=float(np.sqrt(np.mean(values**2))) if len(values) else None
    margins=np.concatenate([p.get("joint_limit_margin_rad",np.full(len(m),np.nan))[m&ok] for _,m,p,ok in selected])
    out.update({f"joint_limit_margin_rad_{key}":value for key,value in _distribution(margins).items()})
    out["total_wrapped_joint_travel_rad"]=float(sum(np.nansum(np.abs(p["clean_metrics"]["clean_wrapped_delta_q_rad"][m])) for _,m,p,_ in selected))
    segments=[]; branch_changes=0; search_changes=0
    for item,m,p,ok in selected:
        scoped=clean_motion(p["q"],item["raw"]["time_s"],item["raw"]["motive_frame"],item["raw"]["run_id"],ok&m,p["continuity_violation"])
        segments.extend(scoped["continuous_segments"]); branch=np.asarray(p.get("branch_id",[""]*len(m))); search=np.asarray(p.get("search_branch",[""]*len(m)))
        for segment in scoped["continuous_segments"]:
            for previous,current in zip(segment[:-1],segment[1:]):
                branch_changes += int(bool(branch[previous] and branch[current] and branch[previous] != branch[current]))
                search_changes += int(bool(search[previous] and search[current] and search[previous] != search[current]))
    out["continuous_segments"]=int(len(segments)); out["longest_continuous_segment_frames"]=max((len(x) for x in segments),default=0)
    out["clean_branch_changes"]=branch_changes; out["clean_search_branch_changes"]=search_changes
    return out


def _pair_clean_metrics(item: dict[str, Any], left: str, right: str, mask: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, int]:
    """Recompute both motions on the identical continuity-clean frame population."""
    raw=item["raw"]; a=item["payloads"][left]; b=item["payloads"][right]
    solver_common=(a["evaluation_status"] == SUCCESS_EXACT) & (b["evaluation_status"] == SUCCESS_EXACT) & mask
    pair_violation=a["continuity_violation"] | b["continuity_violation"]
    ma=_metrics(a["q"], a.get("joint_limit_margin_rad",np.full(len(solver_common),np.nan)), raw["time_s"],raw["motive_frame"],raw["run_id"],solver_common,pair_violation,a.get("branch_id",np.full(len(solver_common),"")))
    mb=_metrics(b["q"], b.get("joint_limit_margin_rad",np.full(len(solver_common),np.nan)), raw["time_s"],raw["motive_frame"],raw["run_id"],solver_common,pair_violation,b.get("branch_id",np.full(len(solver_common),"")))
    intervals=sum(max(0,len(x)-1) for x in ma["continuous_segments"])
    return ma,mb,solver_common,intervals


def _pooled_pair_row(items: list[dict[str, Any]], left: str, right: str, masks: list[np.ndarray],
                     grouping: str, group: str) -> dict[str, Any]:
    """Pool raw values after applying exactly the same pair-clean support."""
    left_motion=[]; right_motion=[]; left_margin=[]; right_margin=[]; common_frames=0; clean_frames=0; intervals=0
    violations_left=violations_right=0
    for item, mask in zip(items,masks):
        ma,mb,common,nintervals=_pair_clean_metrics(item,left,right,mask)
        left_motion.append(ma); right_motion.append(mb); common_frames += int(common.sum()); intervals += nintervals
        pa,pb=item["payloads"][left],item["payloads"][right]
        pair_clean=common & ~ (pa["continuity_violation"] | pb["continuity_violation"])
        clean_frames += int(pair_clean.sum())
        left_margin.append(pa.get("joint_limit_margin_rad",np.full(len(mask),np.nan))[pair_clean]); right_margin.append(pb.get("joint_limit_margin_rad",np.full(len(mask),np.nan))[pair_clean])
        violations_left += int((pa["continuity_violation"] & mask).sum()); violations_right += int((pb["continuity_violation"] & mask).sum())
    row={"scope":"pairwise_continuity_clean","grouping":grouping,"group":group,"strategy_a":left,"strategy_b":right,"solver_common_success_frames":common_frames,"clean_common_frames":clean_frames,"clean_common_intervals":intervals,"strategy_a_continuity_violations":violations_left,"strategy_b_continuity_violations":violations_right,"violations_difference_a_minus_b":violations_left-violations_right}
    metric_names=(("travel","clean_wrapped_delta_q_rad"),("velocity","clean_joint_velocity_rad_s"),("acceleration","clean_joint_acceleration_rad_s2"),("jerk","clean_joint_jerk_rad_s3"))
    for label,key in metric_names:
        av=np.concatenate([x[key].ravel() for x in left_motion]); bv=np.concatenate([x[key].ravel() for x in right_motion]); av=np.abs(av[np.isfinite(av)]); bv=np.abs(bv[np.isfinite(bv)])
        for prefix,values in (("strategy_a",av),("strategy_b",bv)):
            row[f"{prefix}_{label}_support_values"]=int(len(values)); row[f"{prefix}_{label}_total"]=float(values.sum()) if label=="travel" else None; row[f"{prefix}_{label}_rms"]=float(np.sqrt(np.mean(values**2))) if len(values) else None; row[f"{prefix}_{label}_p95"]=float(np.percentile(values,95)) if len(values) else None; row[f"{prefix}_{label}_max"]=float(values.max()) if len(values) else None
        if label == "travel": row["travel_difference_a_minus_b"]=row["strategy_a_travel_total"]-row["strategy_b_travel_total"]
        else: row[f"rms_{label}_difference_a_minus_b"] = None if not len(av) or not len(bv) else row[f"strategy_a_{label}_rms"]-row[f"strategy_b_{label}_rms"]
    ma=np.concatenate(left_margin) if left_margin else np.array([]); mb=np.concatenate(right_margin) if right_margin else np.array([]); da=_distribution(ma); db=_distribution(mb)
    for prefix,distribution in (("strategy_a_joint_limit_margin_rad",da),("strategy_b_joint_limit_margin_rad",db)):
        row.update({f"{prefix}_{key}":value for key,value in distribution.items()})
    row["min_margin_difference_a_minus_b_positive_is_higher"]=None if da["min"] is None or db["min"] is None else da["min"]-db["min"]
    return row


def _plots(metrics: pd.DataFrame, events: pd.DataFrame, directory: Path) -> None:
    overall=metrics.query("grouping == 'overall'");
    for filename,column,ylabel in (("feasibility.png","success_percent","SUCCESS_EXACT (%)"),("continuity_violations.png","continuity_violations","violations"),("clean_joint_motion.png","total_wrapped_joint_travel_rad","clean travel (rad)")):
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(overall.strategy,overall[column]); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(directory/filename,dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); subset=overall[overall.strategy.isin(["StrongLocal","HumanGT",ROBOT_SMOOTH])]; ax.bar(subset.strategy,subset.total_wrapped_joint_travel_rad); ax.set_ylabel("clean joint travel (rad)"); fig.tight_layout(); fig.savefig(directory/"robot_smooth_comparison.png",dpi=160); plt.close(fig)
