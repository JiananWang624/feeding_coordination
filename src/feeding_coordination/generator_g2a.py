"""G2A: position-only endpoint constrained transfer trajectories.

This module deliberately has no robot, IK, or calibration dependency.  The
``mouth_proxy`` is frozen Phase 1.5 context.  It was constructed upstream from
the final transfer pose, so this is an *offline provisional* target rather than
a deployable independent mouth measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .generator_g0 import (GRID, G0Record, _proper_so3, _sha256,
                           load_phase15_records, promp_generate)


START_CANDIDATES = (0.50, 0.65, 0.80)
BOOTSTRAP_SEED = 20260915
BOOTSTRAP_RESAMPLES = 2000
OFFSET_M = 0.18


@dataclass(frozen=True)
class GeneratorG2AConfig:
    """Frozen G2A settings; target orientation is intentionally unavailable."""
    phase1_output_path: str = "outputs/phase1"
    g0_output_path: str = "outputs/generator_g0"
    output_path: str = "outputs/generator_g2a"
    start_candidates: tuple[float, ...] = START_CANDIDATES
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES


def quintic_smoothstep(u: np.ndarray | float) -> np.ndarray:
    u = np.asarray(u, float)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def quintic_smoothstep_first(u: np.ndarray | float) -> np.ndarray:
    u = np.asarray(u, float)
    return 30 * u**2 - 60 * u**3 + 30 * u**4


def quintic_smoothstep_second(u: np.ndarray | float) -> np.ndarray:
    u = np.asarray(u, float)
    return 60 * u - 180 * u**2 + 120 * u**3


def virtual_transfer_goal_position(record: G0Record, nominal_orientation_B: np.ndarray) -> np.ndarray:
    """Query-side provisional target: frozen proxy plus G0 nominal +Y offset.

    Crucially this does not read ``record.gt_position_B`` or
    ``record.gt_orientation_B``.  Its upstream proxy provenance is still
    indirect future-endpoint information and is reported as such in the audit.
    """
    orientation = np.asarray(nominal_orientation_B, float)
    if record.phase != "transfer" or orientation.shape != (101, 3, 3):
        raise ValueError("G2A virtual target requires a 101-sample transfer trajectory")
    return np.asarray(record.mouth_proxy_B, float) + OFFSET_M * orientation[-1, :, 1]


def constrain_position_endpoint(nominal_position_B: np.ndarray, goal_position_B: np.ndarray,
                                start_phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the prescribed quintic translation correction and return (path, delta)."""
    if float(start_phase) not in START_CANDIDATES:
        raise ValueError(f"start_phase must be one of {START_CANDIDATES}")
    nominal = np.asarray(nominal_position_B, float)
    goal = np.asarray(goal_position_B, float)
    if nominal.shape != (101, 3) or goal.shape != (3,):
        raise ValueError("expected nominal [101,3] and goal [3]")
    correction = goal - nominal[-1]
    blend = np.zeros(len(GRID))
    active = GRID >= start_phase
    u = (GRID[active] - start_phase) / (1.0 - start_phase)
    blend[active] = quintic_smoothstep(u)
    delta = blend[:, None] * correction
    constrained = nominal + delta
    constrained[-1] = goal  # exact floating-point endpoint contract
    return constrained, delta


def constrain_orientation_endpoint(nominal_orientation_B: np.ndarray, start_phase: float,
                                   goal_orientation_B: np.ndarray | None = None) -> np.ndarray:
    """Future opt-in SO(3) endpoint adaptation; primary G2A calls this with None."""
    nominal = np.asarray(nominal_orientation_B, float)
    if nominal.shape != (101, 3, 3) or not _proper_so3(nominal).all():
        raise ValueError("nominal orientation must be 101 proper SO(3) matrices")
    if goal_orientation_B is None:
        return nominal.copy()
    if not _proper_so3(np.asarray(goal_orientation_B, float)[None]).all():
        raise ValueError("goal orientation must be proper SO(3)")
    delta = Rotation.from_matrix(nominal[-1].T @ goal_orientation_B).as_rotvec()
    blend = np.zeros(len(GRID)); active = GRID >= start_phase
    blend[active] = quintic_smoothstep((GRID[active] - start_phase) / (1. - start_phase))
    increments = Rotation.from_rotvec(blend[:, None] * delta).as_matrix()
    return np.einsum("nij,njk->nik", nominal, increments)


def _bite_balanced_rms(rows: list[tuple[str, float]]) -> float:
    frame = pd.DataFrame(rows, columns=["parent_bite_id", "rms"])
    return float(frame.groupby("parent_bite_id", sort=True).rms.mean().mean())


def select_adaptation_start(outer_training: list[G0Record], alpha: float,
                            candidates: tuple[float, ...] = START_CANDIDATES,
                            trace: dict | None = None) -> tuple[float, list[dict]]:
    """Nested leave-one-training-take-out selection, never reading outer test data."""
    transfer = [r for r in outer_training if r.phase == "transfer"]
    takes = sorted({r.take for r in transfer})
    if len(takes) < 2:
        raise ValueError("G2A selection needs at least two outer-training takes")
    scores = []
    for start in candidates:
        errors: list[tuple[str, float]] = []
        for held in takes:
            inner_train = [r for r in transfer if r.take != held]
            validation = [r for r in transfer if r.take == held]
            for query in validation:
                nominal, orientation, _ = promp_generate(query, inner_train, alpha)
                goal = virtual_transfer_goal_position(query, orientation)
                constrained, _ = constrain_position_endpoint(nominal, goal, start)
                rms = float(np.sqrt(np.mean(np.sum((constrained - query.gt_position_B) ** 2, axis=1))))
                errors.append((query.parent_bite_id, rms))
        scores.append({"adaptation_start_phase": float(start),
                       "inner_bite_balanced_trajectory_position_rms_m": _bite_balanced_rms(errors),
                       "inner_validation_takes": takes})
    # Negative start makes later phases win exact ties.
    chosen = min(scores, key=lambda x: (x["inner_bite_balanced_trajectory_position_rms_m"], -x["adaptation_start_phase"]))
    if trace is not None:
        trace["adaptation_selection"] = scores
        trace["selection_training_takes"] = takes
    return float(chosen["adaptation_start_phase"]), scores


def _g0_arrays(g0_output: Path, record_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = g0_output / "generated" / "contextual_promp" / f"{record_id}.npz"
    with np.load(path, allow_pickle=False) as z:
        return (np.asarray(z["generated_tool_position_B"], float),
                np.asarray(z["generated_tool_orientation_B"], float), np.asarray(z["time_s"], float))


def _metrics(record: G0Record, nominal: np.ndarray, constrained: np.ndarray,
             orientation: np.ndarray, goal: np.ndarray, start: float) -> dict:
    error_nom = np.linalg.norm(nominal - record.gt_position_B, axis=1)
    error_con = np.linalg.norm(constrained - record.gt_position_B, axis=1)
    rot_error = Rotation.from_matrix(np.einsum("nij,njk->nik", orientation.transpose(0, 2, 1), record.gt_orientation_B)).magnitude()
    deform = np.linalg.norm(constrained - nominal, axis=1)
    nominal_path = float(np.linalg.norm(np.diff(nominal, axis=0), axis=1).sum())
    constrained_path = float(np.linalg.norm(np.diff(constrained, axis=0), axis=1).sum())
    deriv = lambda p: np.gradient(p, GRID, axis=0, edge_order=2)
    c1, n1 = deriv(constrained), deriv(nominal)
    c2, n2 = deriv(c1), deriv(n1)
    c3, n3 = deriv(c2), deriv(n2)
    bins = {"early": GRID < .6, "middle": (GRID >= .6) & (GRID < .8), "late": GRID >= .8,
            "last20": GRID >= .8, "last10": GRID >= .9}
    out = {"record_id": record.record_id, "parent_bite_id": record.parent_bite_id, "take": record.take,
           "phase": "transfer", "outer_fold": record.take, "generation_status": "SUCCESS",
           "adaptation_start_phase": start,
           "virtual_endpoint_error_m": float(np.linalg.norm(constrained[-1] - goal)),
           "nominal_measured_endpoint_error_m": float(error_nom[-1]),
           "constrained_measured_endpoint_error_m": float(error_con[-1]),
           "nominal_rms_position_error_m": float(np.sqrt(np.mean(error_nom**2))),
           "constrained_rms_position_error_m": float(np.sqrt(np.mean(error_con**2))),
           "nominal_mean_position_error_m": float(error_nom.mean()), "constrained_mean_position_error_m": float(error_con.mean()),
           "nominal_p95_position_error_m": float(np.percentile(error_nom, 95)), "constrained_p95_position_error_m": float(np.percentile(error_con, 95)),
           "nominal_max_position_error_m": float(error_nom.max()), "constrained_max_position_error_m": float(error_con.max()),
           "rms_orientation_error_rad": float(np.sqrt(np.mean(rot_error**2))), "final_orientation_error_rad": float(rot_error[-1]),
           "nominal_path_length_m": nominal_path, "constrained_path_length_m": constrained_path,
           "nominal_path_length_absolute_error_m": abs(nominal_path - np.linalg.norm(np.diff(record.gt_position_B, axis=0), axis=1).sum()),
           "constrained_path_length_absolute_error_m": abs(constrained_path - np.linalg.norm(np.diff(record.gt_position_B, axis=0), axis=1).sum()),
           "adaptation_rms_m": float(np.sqrt(np.mean(deform**2))), "adaptation_max_m": float(deform.max()),
           "adaptation_first_nonzero_phase": float(GRID[np.flatnonzero(deform > 1e-14)[0]]) if (deform > 1e-14).any() else np.nan,
           "adaptation_at_80pct_m": float(deform[80]), "adaptation_at_90pct_m": float(deform[90]), "adaptation_at_100pct_m": float(deform[100]),
           "constrained_first_derivative_rms": float(np.sqrt(np.mean(np.sum(c1*c1, axis=1)))), "nominal_first_derivative_rms": float(np.sqrt(np.mean(np.sum(n1*n1, axis=1)))),
           "constrained_second_derivative_rms": float(np.sqrt(np.mean(np.sum(c2*c2, axis=1)))), "nominal_second_derivative_rms": float(np.sqrt(np.mean(np.sum(n2*n2, axis=1)))),
           "constrained_third_derivative_rms": float(np.sqrt(np.mean(np.sum(c3*c3, axis=1)))), "nominal_third_derivative_rms": float(np.sqrt(np.mean(np.sum(n3*n3, axis=1)))),}
    nearest = int(np.argmin(np.abs(GRID - start)))
    for label, indexes in {"near_adaptation_start": np.arange(max(0, nearest-1), min(101, nearest+2)),
                           "near_endpoint": np.arange(98, 101)}.items():
        for derivative_name, constrained_d, nominal_d in (("first", c1, n1), ("second", c2, n2), ("third", c3, n3)):
            out[f"{label}_{derivative_name}_derivative_change_max"] = float(np.max(np.linalg.norm(constrained_d[indexes] - nominal_d[indexes], axis=1)))
    for name, mask in bins.items():
        out[f"nominal_{name}_rms_position_error_m"] = float(np.sqrt(np.mean(error_nom[mask]**2)))
        out[f"constrained_{name}_rms_position_error_m"] = float(np.sqrt(np.mean(error_con[mask]**2)))
    return out


def paired_bootstrap_g2a(metrics: pd.DataFrame, seed: int = BOOTSTRAP_SEED,
                         resamples: int = BOOTSTRAP_RESAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = ("rms_position_error_m", "measured_endpoint_error_m", "last20_rms_position_error_m", "path_length_absolute_error_m")
    rows = []
    for name in columns:
        pair = metrics.groupby("parent_bite_id")[[f"nominal_{name}", f"constrained_{name}"]].mean()
        difference = (pair.iloc[:, 1] - pair.iloc[:, 0]).to_numpy()
        draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
        samples = difference[draws].mean(axis=1)
        rows.append({"metric": name, "comparison": "constrained_minus_nominal", "convention": "negative favors constrained",
                     "mean_difference_m": float(difference.mean()), "ci95_low_m": float(np.percentile(samples, 2.5)),
                     "ci95_high_m": float(np.percentile(samples, 97.5)), "n_parent_bites": len(difference),
                     "seed": seed, "resamples": resamples})
    return pd.DataFrame(rows)


def _representative(metrics: pd.DataFrame) -> dict:
    ordered = metrics.sort_values("record_id").copy()
    def item(frame: pd.DataFrame) -> dict:
        r = frame.iloc[0]
        return {k: (float(r[k]) if isinstance(r[k], np.floating) else r[k]) for k in ("record_id", "parent_bite_id", "take", "adaptation_rms_m", "nominal_last20_rms_position_error_m", "constrained_last20_rms_position_error_m")}
    median = ordered.adaptation_rms_m.median()
    return {"minimal_correction": item(ordered.sort_values(["adaptation_max_m", "record_id"])),
            "typical_correction": item(ordered.assign(d=(ordered.adaptation_rms_m-median).abs()).sort_values(["d", "record_id"])),
            "large_correction": item(ordered.sort_values(["adaptation_max_m", "record_id"], ascending=[False, True])),
            "improvement_example": item(ordered.assign(d=ordered.constrained_last20_rms_position_error_m-ordered.nominal_last20_rms_position_error_m).sort_values(["d", "record_id"])),
            "shape_limited_example": item(ordered.assign(d=ordered.constrained_early_rms_position_error_m + ordered.constrained_middle_rms_position_error_m).sort_values(["d", "record_id"], ascending=[False, True]))}


def _plots(output: Path, metrics: pd.DataFrame, examples: dict, generated: Path) -> list[str]:
    """Small deterministic aggregate plots; no values are used for model selection."""
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    specs = (("endpoint_accuracy.png", "nominal_measured_endpoint_error_m", "constrained_measured_endpoint_error_m", "measured endpoint error (m)"),
             ("trajectory_rms.png", "nominal_rms_position_error_m", "constrained_rms_position_error_m", "trajectory position RMS (m)"),
             ("final_approach_error.png", "nominal_last20_rms_position_error_m", "constrained_last20_rms_position_error_m", "final 20% position RMS (m)"),
             ("adaptation_magnitude.png", "adaptation_rms_m", "adaptation_max_m", "adaptation magnitude (m)"))
    names = []
    for name, left, right, ylabel in specs:
        values = [float(metrics[left].mean()), float(metrics[right].mean())]
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        labels = ["RMS adaptation", "Maximum adaptation"] if name == "adaptation_magnitude.png" else ["G0 nominal", "G2A constrained"]
        ax.bar(labels, values, color=["#6c8ebf", "#e69138"])
        ax.set_ylabel(ylabel); ax.set_title(name.removesuffix(".png").replace("_", " "))
        fig.tight_layout(); fig.savefig(plots / name, dpi=150); plt.close(fig); names.append(f"plots/{name}")
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, example in examples.items():
        with np.load(generated / f"{example['record_id']}.npz", allow_pickle=False) as z:
            n, c = z["nominal_position_B"], z["constrained_position_B"]
        ax.plot(n[:, 0], n[:, 1], "--", alpha=.55, label=f"{label} nominal")
        ax.plot(c[:, 0], c[:, 1], alpha=.9, label=f"{label} constrained")
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("B x (m)"); ax.set_ylabel("B y (m)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(plots / "example_paths.png", dpi=150); plt.close(fig)
    names.append("plots/example_paths.png")
    return names


def process_generator_g2a(root: Path, output_dir: Path | None = None) -> dict:
    """Run G2A from frozen Phase 1.5/G0 outputs; writes only ``output_dir``."""
    root = Path(root)
    cfg = GeneratorG2AConfig()
    phase1, g0 = root / cfg.phase1_output_path, root / cfg.g0_output_path
    output = Path(output_dir) if output_dir is not None else root / cfg.output_path
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "generated" / "contextual_promp_endpoint_constrained"
    generated.mkdir(parents=True, exist_ok=True)
    records, phase1_manifest, exclusions = load_phase15_records(phase1)
    transfers = [r for r in records if r.phase == "transfer"]
    g0_summary = json.loads((g0 / "summary.json").read_text())
    alphas = g0_summary["generators"]["contextual_promp"]["selected_alphas"]
    rows, selections, artifacts = [], [], []
    for held in sorted({r.take for r in transfers}):
        outer_train = [r for r in transfers if r.take != held]
        alpha = float(alphas[f"{held}:transfer"])
        start, score = select_adaptation_start(outer_train, alpha)
        selections.append({"outer_fold": held, "g0_selected_alpha": alpha, "selected_adaptation_start_phase": start,
                           "candidate_scores_json": json.dumps(score, sort_keys=True), "outer_training_takes": json.dumps(sorted({r.take for r in outer_train}))})
        for query in (r for r in transfers if r.take == held):
            nominal, orientation, _ = promp_generate(query, outer_train, alpha)
            frozen_position, frozen_orientation, frozen_time_s = _g0_arrays(g0, query.record_id)
            if not (np.allclose(nominal, frozen_position, atol=1e-12, rtol=0) and np.allclose(orientation, frozen_orientation, atol=1e-12, rtol=0)):
                raise ValueError(f"G0 parity failed for {query.record_id}")
            goal = virtual_transfer_goal_position(query, orientation)
            constrained, delta = constrain_position_endpoint(nominal, goal, start)
            constrained_orientation = constrain_orientation_endpoint(orientation, start, None)
            if not (_proper_so3(constrained_orientation).all() and np.array_equal(constrained_orientation, frozen_orientation)):
                raise ValueError(f"invalid or altered nominal orientation for {query.record_id}")
            rows.append(_metrics(query, nominal, constrained, orientation, goal, start))
            path = generated / f"{query.record_id}.npz"
            np.savez_compressed(path, record_id=np.array(query.record_id), parent_bite_id=np.array(query.parent_bite_id), take=np.array(query.take), phase=np.array("transfer"), outer_fold=np.array(held), nominal_position_B=nominal, nominal_orientation_B=orientation, constrained_position_B=constrained, constrained_orientation_B=constrained_orientation, virtual_goal_position_B=goal, adaptation_start_phase=np.array(start), adaptation_vector_B=delta[-1], normalized_phase=GRID, time_s=frozen_time_s, physical_tool_calibration_validated=np.array(False), physical_mouth_pose_validated=np.array(False), virtual_endpoint_constraint=np.array(True), upstream_proxy_indirect_future_endpoint_information=np.array(True))
            artifacts.append({"record_id": query.record_id, "path": str(path), "sha256": _sha256(path)})
    metrics = pd.DataFrame(rows).sort_values("record_id")
    metrics.to_csv(output / "record_metrics.csv", index=False)
    bite = metrics.groupby(["parent_bite_id", "take"], as_index=False).mean(numeric_only=True)
    bite.to_csv(output / "bite_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "fold_selection.csv", index=False)
    bootstrap = paired_bootstrap_g2a(metrics)
    bootstrap.to_csv(output / "pairwise_bootstrap.csv", index=False)
    examples = _representative(metrics)
    (output / "representative_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True))
    plot_paths = _plots(output, metrics, examples, generated)
    recorded_relation_residuals = [float(np.linalg.norm(r.mouth_proxy_B + OFFSET_M * r.gt_orientation_B[-1, :, 1] - r.gt_position_B[-1])) for r in transfers]
    audit = {"upstream_field": "Phase1.5 mouth_target_position (aliased mouth_proxy)", "upstream_coordinate_frame": "Motive Global millimetres", "phase1_coordinate_frame": "B/world metres", "contains_position_only": True,
             "upstream_construction": "Hand_final + 0.080 m * ForkMinusY_final", "fork_rigid_body_convention": "Hand + 0.100 m * ForkPlusY",
             "deterministic_relation": "mouth_proxy ~= fork_final - 0.180 m * fork_final_R[:,1]", "virtual_goal_construction": "mouth_proxy_B + 0.180 m * G0_nominal_final_R_B[:,1]",
             "recorded_relation_residual_mean_m": float(np.mean(recorded_relation_residuals)),
             "recorded_relation_residual_max_m": float(np.max(recorded_relation_residuals)),
             "recorded_relation_uses_measured_orientation_for_audit_only": True,
             "physical_tool_calibration_validated": False, "physical_mouth_pose_validated": False, "virtual_endpoint_constraint": True,
             "indirect_future_endpoint_information": True, "claim": "offline provisional endpoint verification only; no independent mouth sensing or leakage-free deployment claim"}
    (output / "endpoint_semantics_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    summary = {"schema_version": "generator-g2a-v1", "config": asdict(cfg), "records": len(metrics), "takes": sorted({r.take for r in transfers}), "endpoint_semantics_audit": audit,
               "selected_adaptation_starts": {r["outer_fold"]: r["selected_adaptation_start_phase"] for r in selections},
               "endpoint_max_error_m": float(metrics.virtual_endpoint_error_m.max()), "metrics_bite_balanced": bite.mean(numeric_only=True).to_dict(),
               "paired_parent_bite_bootstrap": bootstrap.to_dict("records"), "representative_examples": examples,
               "withdrawal_endpoint_optimization_deferred": True, "scope": {"robot_ik_started": False, "stronglocal_started": False, "exact_sew_started": False, "physical_calibration_started": False},
               "source": {"phase1_manifest_sha256": _sha256(phase1 / "manifest.json"), "g0_manifest_sha256": _sha256(g0 / "manifest.json")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    manifest = {"schema_version": "generator-g2a-v1", "summary": "summary.json", "endpoint_semantics_audit": "endpoint_semantics_audit.json", "fold_selection": "fold_selection.csv", "record_metrics": "record_metrics.csv", "bite_metrics": "bite_metrics.csv", "pairwise_bootstrap": "pairwise_bootstrap.csv", "representative_examples": "representative_examples.json", "plots": plot_paths, "generated_artifacts": artifacts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return summary
