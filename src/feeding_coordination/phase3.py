"""Phase 3: held-out-session human coordination information hierarchy.

This module deliberately consumes only immutable Phase 1/1.5 human records.
It has no dependency on Phase 2, robot feasibility, or mouth-target fields.
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
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from .phase1 import _revision

MODELS = ("M1_pose", "M2_pose_velocity", "M3_strong_local", "M4_history", "M5_future", "M6_full_plan")
HISTORY_OFFSETS_S = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class Phase3Config:
    phase1_output_path: str = "outputs/phase1"
    output_path: str = "outputs/phase3"
    alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1., 10., 100.)
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260915
    spline_basis_functions: int = 12

    @classmethod
    def load(cls, path: Path) -> "Phase3Config":
        raw = json.loads(Path(path).read_text())
        raw["alphas"] = tuple(raw.get("alphas", cls.alphas))
        cfg = cls(**raw)
        if cfg.spline_basis_functions != 12: raise ValueError("Phase 3 freezes 12 cubic B-spline bases")
        if not cfg.alphas or any(a <= 0 for a in cfg.alphas): raise ValueError("alphas must be positive")
        return cfg


def _runs(mask: np.ndarray, motive_frame: np.ndarray | None = None) -> list[np.ndarray]:
    """Contiguous index runs; adjacency is original segment-row adjacency."""
    idx = np.flatnonzero(mask)
    if not len(idx): return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    if motive_frame is not None: cuts = np.unique(np.r_[cuts, np.flatnonzero(np.diff(np.asarray(motive_frame)[idx]) != 1) + 1])
    return list(np.split(idx, cuts))


def _rotvec(matrices: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrices).as_rotvec()


def _basis(s: np.ndarray, n_basis: int = 12) -> np.ndarray:
    degree = 3
    knots = np.r_[np.zeros(degree + 1), np.linspace(0, 1, n_basis - degree + 1)[1:-1], np.ones(degree + 1)]
    return BSpline.design_matrix(np.clip(s, 0, 1), knots, degree).toarray()


def _finite_pose(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    return np.isfinite(position).all(1) & np.isfinite(orientation).all((1, 2))


def _segment_features(z: np.lib.npyio.NpzFile, record: dict, cfg: Phase3Config) -> tuple[list[dict], dict]:
    """Construct all model rows for one transfer/withdrawal segment using U alone."""
    p = np.asarray(z["tool_position"], float); R = np.asarray(z["tool_orientation"], float)
    time = np.asarray(z["time"], float); psi = np.asarray(z["psi_unwrapped"], float); motive = np.asarray(z["motive_frame"], int)
    tool = np.asarray(z["tool_pose_valid"], bool) & _finite_pose(p, R)
    psi_ok = np.asarray(z["psi_valid"], bool) & np.isfinite(psi)
    target = tool & psi_ok  # sole Phase-3 target eligibility
    if not target.any(): return [], {"max_relative_rotation_rad": np.nan, "psi_runs": 0}
    first = np.flatnonzero(tool)[0]; p0, R0 = p[first], R[first]
    p_rel = (R0.T @ (p - p0).T).T
    R_rel = np.einsum("ij,njk->nik", R0.T, R)
    x_u = np.full((len(p), 6), np.nan); x_u[tool] = np.c_[p_rel[tool], _rotvec(R_rel[tool])]
    s = (time - time[0]) / (time[-1] - time[0]) if time[-1] > time[0] else np.zeros(len(time))
    # Causal velocity: only immediate, measured predecessor; missing predecessor is zero.
    vel = np.zeros((len(p), 6)); vel_ok = np.zeros(len(p), bool)
    for i in range(1, len(p)):
        dt = time[i] - time[i - 1]
        if tool[i] and tool[i - 1] and np.asarray(z["motive_frame"], int)[i] == np.asarray(z["motive_frame"], int)[i - 1] + 1 and dt > 0:
            vel[i, :3] = (p_rel[i] - p_rel[i - 1]) / dt
            # Body angular increment is invariant to a common B-frame transform.
            vel[i, 3:] = _rotvec((R[i - 1].T @ R[i])[None])[0] / dt
            vel_ok[i] = True
    plate_valid = np.asarray(z["plate_position_valid"], bool) & np.isfinite(z["plate_position"]).all(1)
    plate_rel = np.median((R0.T @ (np.asarray(z["plate_position"], float)[plate_valid] - p0).T).T, axis=0) if plate_valid.any() else np.zeros(3)
    plate_mask = float(plate_valid.any())
    # Full-plan coefficients use valid U observations, never psi or mouth target.
    plan_coeff = np.linalg.lstsq(_basis(s[tool], cfg.spline_basis_functions), x_u[tool], rcond=None)[0].ravel()
    max_angle = float(np.max(np.linalg.norm(x_u[tool, 3:], axis=1)))
    phase = str(record["phase"]); phase_onehot = np.array([phase == "transfer", phase == "withdrawal"], float)
    rows: list[dict] = []
    # psi0 is deliberately defined over psi-valid runs, irrespective of missing U rows.
    psi_run_by_index = {}
    for run_number, run in enumerate(_runs(psi_ok, motive)):
        psi0 = psi[run[0]]
        for i in run: psi_run_by_index[i] = (run_number, psi0)
    valid_u_idx = np.flatnonzero(tool)
    tool_run_by_index = {i: run for run in _runs(tool, motive) for i in run}
    path_cumulative = np.zeros(len(p))
    for a, b in zip(valid_u_idx[:-1], valid_u_idx[1:]):
        path_cumulative[b] = path_cumulative[a] + (np.linalg.norm(p_rel[b] - p_rel[a]) if b == a + 1 and motive[b] == motive[a] + 1 else 0.)
    # Carry cumulative length across non-tool rows only for target rows (which are tool rows).
    for i in range(1, len(path_cumulative)):
        if not tool[i]: path_cumulative[i] = path_cumulative[i - 1]
    for i in np.flatnonzero(target):
        run_id, psi0 = psi_run_by_index[i]
        local = np.r_[x_u[i], vel[i], s[i], phase_onehot, plate_rel, plate_mask, psi0]
        history = []
        for offset in HISTORY_OFFSETS_S:
            current_run = tool_run_by_index[i]
            candidates = current_run[(current_run <= i) & (time[current_run] <= time[i] - offset + 1e-9)]
            if len(candidates): history.extend(x_u[candidates[-1]]); history.append(1.)
            else: history.extend(x_u[current_run[0]]); history.append(0.)
        future = []
        remaining = np.flatnonzero(tool & (s >= s[i]))
        for fraction in (0.25, 0.50, 1.0):
            desired = s[i] + fraction * (1. - s[i]); j = remaining[np.argmin(np.abs(s[remaining] - desired))]
            future.extend(R[i].T @ (p[j] - p[i])); future.extend(_rotvec((R[i].T @ R[j])[None])[0])
        future.append(float(path_cumulative[valid_u_idx[-1]] - path_cumulative[i]))
        rows.append({
            "record_id": record["record_id"], "parent_bite_id": record["parent_bite_id"], "take": record["source_take"],
            "demonstrator": record.get("demonstrator_id", "D1" if record["source_take"] == "trial_0015" else "D0"),
            "phase": phase, "frame": int(z["motive_frame"][i]), "segment_index": int(i), "time": float(time[i]), "s": float(s[i]),
            "psi_true": float(psi[i]), "psi0": float(psi0), "delta_psi": float(psi[i] - psi0), "psi_run_id": int(run_id),
            "M1_pose": x_u[i], "M2_pose_velocity": np.r_[x_u[i], vel[i]], "M3_strong_local": local,
            "M4_history": np.r_[local, history], "M5_future": np.r_[local, future], "M6_full_plan": np.r_[local, plan_coeff],
            "velocity_measured_predecessor": bool(vel_ok[i]),
        })
    return rows, {"max_relative_rotation_rad": max_angle, "psi_runs": len(_runs(psi_ok, motive))}


def build_dataset(phase1_dir: Path, cfg: Phase3Config) -> tuple[pd.DataFrame, dict]:
    """The sole Phase-3 data path. Reads Phase 1.5 only; Phase 2 is never opened."""
    phase1_dir = Path(phase1_dir)
    manifest = json.loads((phase1_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != "phase1-v1.5": raise ValueError("Phase 3 requires Phase 1.5 human data")
    all_rows, diagnostics = [], []
    for record in manifest["records"]:
        with np.load(phase1_dir / record["file"]) as z:
            rows, diag = _segment_features(z, record, cfg)
        all_rows.extend(rows); diagnostics.append({"record_id": record["record_id"], **diag})
    data = pd.DataFrame(all_rows)
    if data.empty: raise ValueError("no Phase-3 eligible human rows")
    return data, {"source_manifest": str((phase1_dir / "manifest.json").resolve()), "source_schema": manifest["schema_version"],
                  "records": len(manifest["records"]), "eligible_segments": int(data.record_id.nunique()), "parent_bites": int(data.parent_bite_id.nunique()), "valid_target_frames": len(data),
                  "psi_runs": int(sum(d["psi_runs"] for d in diagnostics)), "max_relative_rotation_rad": float(np.nanmax([d["max_relative_rotation_rad"] for d in diagnostics]))}


def bite_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("parent_bite_id")["parent_bite_id"].transform("size").to_numpy(float)
    return 1. / counts


def circular_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (pred - true)))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str, alpha: float, trace: dict | None = None) -> np.ndarray:
    if trace is not None:
        event = {"outer_held": trace.get("current_outer"), "scaler_takes": set(train["take"]), "fit_takes": set(train["take"])}
        trace.setdefault("fit_events", []).append(event)
    Xtr = np.vstack(train[model]); Xte = np.vstack(test[model]); y = train.delta_psi.to_numpy(float); w=bite_weights(train)
    if not (np.isfinite(Xtr).all() and np.isfinite(Xte).all() and np.isfinite(y).all()): raise ValueError(f"non-finite Phase 3 features for {model}")
    mean=np.average(Xtr,axis=0,weights=w); scale=np.sqrt(np.maximum(np.average((Xtr-mean)**2,axis=0,weights=w),0)); scale[scale < 1e-12]=1.
    Xtr=(Xtr-mean)/scale; Xte=(Xte-mean)/scale
    fit = Ridge(alpha=alpha).fit(Xtr, y, sample_weight=w)
    return test.psi0.to_numpy(float) + fit.predict(Xte)


def _metric_rows(frame: pd.DataFrame, pred: np.ndarray, model: str, grouping: str, group: str) -> dict:
    ce = circular_error(pred, frame.psi_true.to_numpy(float)); de = pred - frame.psi_true.to_numpy(float)
    # Bite-balanced primary: average bite-level squared/absolute circular error.
    bite = pd.DataFrame({"bite": frame.parent_bite_id.to_numpy(), "ce": ce, "de": de}).groupby("bite")
    b_rmse = float(np.sqrt(bite.ce.apply(lambda x: np.mean(x * x)).mean())); b_mae = float(bite.ce.apply(lambda x: np.mean(np.abs(x))).mean())
    return {"model": model, "grouping": grouping, "group": group, "n_frames": len(frame), "n_bites": int(frame.parent_bite_id.nunique()),
            "circular_rmse": b_rmse, "circular_mae": b_mae, "delta_rmse": float(np.sqrt(np.mean(de * de))),
            "frame_circular_rmse": float(np.sqrt(np.mean(ce * ce))), "frame_circular_mae": float(np.mean(np.abs(ce)))}


def choose_alpha(train: pd.DataFrame, model: str, alphas: tuple[float, ...], trace: dict | None = None) -> float:
    takes = sorted(train["take"].unique()); scores = []
    for alpha in alphas:
        valid_parts=[]
        for held in takes:
            inn_train, valid = train[train["take"] != held], train[train["take"] == held]
            pred = _fit_predict(inn_train, valid, model, alpha, trace)
            part=valid[["parent_bite_id","psi_true"]].copy(); part["pred"] = pred; valid_parts.append(part)
        pooled=pd.concat(valid_parts); ce=circular_error(pooled.pred.to_numpy(),pooled.psi_true.to_numpy())
        scores.append(float(np.sqrt(pd.DataFrame({"bite":pooled.parent_bite_id,"sq":ce*ce}).groupby("bite").sq.mean().mean())))
    return float(alphas[int(np.argmin(scores))])


def coordinate_audit(phase1_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Absolute B-frame diagnostic only; its results are never model features."""
    manifest = json.loads((Path(phase1_dir) / "manifest.json").read_text()); records = []
    for rec in manifest["records"]:
        with np.load(Path(phase1_dir) / rec["file"]) as z:
            valid = np.asarray(z["tool_pose_valid"], bool) & _finite_pose(z["tool_position"], z["tool_orientation"])
            if not valid.any(): continue
            i = np.flatnonzero(valid)[0]; pos = np.asarray(z["tool_position"], float); plate = np.asarray(z["plate_position"], float); pv = np.asarray(z["plate_position_valid"], bool) & np.isfinite(plate).all(1)
            records.append({"take":rec["source_take"], "phase":rec["phase"], "initial_position":pos[i], "initial_rotation":np.asarray(z["tool_orientation"],float)[i], "plate":np.median(plate[pv],axis=0) if pv.any() else np.full(3,np.nan)})
    rows=[]
    raw=pd.DataFrame(records)
    for keys,g in raw.groupby(["take","phase"]):
        take,phase=keys; starts=np.vstack(g.initial_position); plates=np.vstack(g.plate); rots=Rotation.from_matrix(np.stack(g.initial_rotation))
        mean_rot=rots.mean(); geodesic=(mean_rot.inv()*rots).magnitude()
        within=float(np.sqrt(np.mean(np.sum((starts-starts.mean(0))**2,axis=1))))
        plate_within=float(np.sqrt(np.nanmean(np.sum((plates-np.nanmean(plates,0))**2,axis=1))))
        rows.append({"take":take,"phase":phase,"segments":len(g),"initial_position_mean":starts.mean(0).tolist(),"initial_position_std":starts.std(0).tolist(),"plate_mean":np.nanmean(plates,0).tolist(),"plate_std":np.nanstd(plates,0).tolist(),"initial_orientation_mean_rotvec":mean_rot.as_rotvec().tolist(),"within_take_orientation_rms_rad":float(np.sqrt(np.mean(geodesic**2))),"within_take_orientation_max_rad":float(np.max(geodesic)),"within_take_segment_start_rms_m":within,"within_take_plate_rms_m":plate_within})
    # Overall per-take summaries retain the requested unstratified distributions.
    for take,g in raw.groupby("take"):
        starts=np.vstack(g.initial_position); plates=np.vstack(g.plate); rots=Rotation.from_matrix(np.stack(g.initial_rotation)); mean_rot=rots.mean(); geodesic=(mean_rot.inv()*rots).magnitude()
        within=float(np.sqrt(np.mean(np.sum((starts-starts.mean(0))**2,axis=1))))
        plate_within=float(np.sqrt(np.nanmean(np.sum((plates-np.nanmean(plates,0))**2,axis=1))))
        rows.append({"take":take,"phase":"overall","segments":len(g),"initial_position_mean":starts.mean(0).tolist(),"initial_position_std":starts.std(0).tolist(),"plate_mean":np.nanmean(plates,0).tolist(),"plate_std":np.nanstd(plates,0).tolist(),"initial_orientation_mean_rotvec":mean_rot.as_rotvec().tolist(),"within_take_orientation_rms_rad":float(np.sqrt(np.mean(geodesic**2))),"within_take_orientation_max_rad":float(np.max(geodesic)),"within_take_segment_start_rms_m":within,"within_take_plate_rms_m":plate_within})
    audit=pd.DataFrame(rows)
    spread=[]
    for phase,g in list(raw.groupby("phase"))+[("overall",raw)]:
        centers=g.groupby("take").initial_position.apply(lambda x: np.mean(np.vstack(x),0)); center=np.mean(np.vstack(centers),0)
        plate_centers=g.groupby("take").plate.apply(lambda x: np.nanmean(np.vstack(x),0)); plate_center=np.nanmean(np.vstack(plate_centers),0)
        within_values=[]
        for _, starts_for_take in g.groupby("take").initial_position:
            starts_for_take=np.vstack(starts_for_take); within_values.append(float(np.sqrt(np.mean(np.sum((starts_for_take-starts_for_take.mean(0))**2,axis=1)))))
        take_rots=[Rotation.from_matrix(np.stack(x)).mean() for _,x in g.groupby("take").initial_rotation]; grand=Rotation.concatenate(take_rots).mean(); between_angles=(grand.inv()*Rotation.concatenate(take_rots)).magnitude()
        spread.append({"phase":phase,"between_take_initial_position_rms_m":float(np.sqrt(np.mean(np.sum((np.vstack(centers)-center)**2,axis=1)))),"within_take_segment_start_rms_m":float(np.mean(within_values)),"between_take_plate_rms_m":float(np.sqrt(np.nanmean(np.sum((np.vstack(plate_centers)-plate_center)**2,axis=1)))),"between_take_orientation_rms_rad":float(np.sqrt(np.mean(between_angles**2))),"between_take_orientation_max_rad":float(np.max(between_angles)),"within_take_orientation_rms_rad":float(audit[audit.phase==phase].within_take_orientation_rms_rad.mean()) if phase != "overall" else float(audit[audit.phase=="overall"].within_take_orientation_rms_rad.mean()),"within_take_plate_rms_m":float(audit[audit.phase==phase].within_take_plate_rms_m.mean()) if phase != "overall" else float(audit[audit.phase=="overall"].within_take_plate_rms_m.mean())})
    return {"per_take_phase":audit.to_dict("records"),"spread":spread}, raw


def _bootstrap(predictions: pd.DataFrame, cfg: Phase3Config) -> list[dict]:
    rng=np.random.default_rng(cfg.bootstrap_seed); pairs=[("M4_history","M3_strong_local"),("M5_future","M4_history"),("M6_full_plan","M5_future"),("M6_full_plan","M3_strong_local")]; out=[]
    for better, baseline in pairs:
        keys=["parent_bite_id","phase","record_id","segment_index"]
        a=predictions[predictions.model==better].copy(); b=predictions[predictions.model==baseline][keys+["circular_abs_error"]]
        joined=a.merge(b,on=keys,suffixes=("_new","_base"),validate="one_to_one"); bite=joined.groupby("parent_bite_id")[["circular_abs_error_new","circular_abs_error_base"]].mean()
        diff=bite.circular_abs_error_new.to_numpy()-bite.circular_abs_error_base.to_numpy(); base=bite.circular_abs_error_base.to_numpy(); draws=rng.integers(0,len(diff),(cfg.bootstrap_resamples,len(diff))); samples=np.mean(diff[draws],axis=1); relative=100*np.mean(diff[draws],axis=1)/np.mean(base[draws],axis=1)
        out.append({"comparison":f"{better} vs {baseline}","convention":"negative means candidate improves","absolute_mae_difference_rad":float(diff.mean()),"relative_percent_change":float(100*diff.mean()/base.mean()),"absolute_ci95":np.percentile(samples,[2.5,97.5]).tolist(),"relative_percent_ci95":np.percentile(relative,[2.5,97.5]).tolist(),"n_bites":len(bite)})
    return out


def _prediction_metric_row(model: str, grouping: str, group: str, pg: pd.DataFrame) -> dict:
    bite=pg.groupby("parent_bite_id").circular_error
    ordered=pg.sort_values(["record_id", "segment_index"])
    previous=ordered.groupby("record_id").shift()
    previous_record=ordered.record_id.shift()
    adjacent=(ordered.record_id.to_numpy() == previous_record.to_numpy()) & (ordered.segment_index.to_numpy() == previous.segment_index.to_numpy() + 1) & (ordered.motive_frame.to_numpy() == previous.motive_frame.to_numpy() + 1) & (ordered.psi_run_id.to_numpy() == previous.psi_run_id.to_numpy()) & (ordered.time.to_numpy() > previous.time.to_numpy())
    if adjacent.any():
        pred_dot=np.diff(ordered.psi_pred.to_numpy()) / np.diff(ordered.time.to_numpy())
        true_dot=np.diff(ordered.psi_true.to_numpy()) / np.diff(ordered.time.to_numpy())
        dot_rmse=float(np.sqrt(np.mean((pred_dot[adjacent[1:]]-true_dot[adjacent[1:]])**2)))
    else: dot_rmse=np.nan
    return {"model":model,"grouping":grouping,"group":str(group),"n_frames":len(pg),"n_bites":pg.parent_bite_id.nunique(),"circular_rmse":float(np.sqrt(bite.apply(lambda x:np.mean(x*x)).mean())),"circular_mae":float(bite.apply(lambda x:np.mean(np.abs(x))).mean()),"delta_rmse":float(np.sqrt(np.mean(pg.delta_error**2))),"psi_dot_rmse":dot_rmse,"frame_circular_rmse":float(np.sqrt(np.mean(pg.circular_error**2))),"frame_circular_mae":float(np.mean(np.abs(pg.circular_error)))}


def _plots(output: Path, metrics: pd.DataFrame, audit_records: pd.DataFrame, data: pd.DataFrame, predictions: pd.DataFrame, examples: pd.DataFrame) -> None:
    fig,ax=plt.subplots(figsize=(8,4)); overall=metrics[metrics.grouping=="overall"]; ax.bar(overall.model,overall.circular_rmse); ax.set(ylabel="bite-balanced circular RMSE (rad)"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(output/"information_hierarchy.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));
    for take,g in audit_records.groupby("take"): ax.scatter(np.vstack(g.initial_position)[:,0],np.vstack(g.initial_position)[:,1],label=take,s=12)
    ax.set(xlabel="initial fork B-x (m)",ylabel="initial fork B-y (m)"); ax.legend(fontsize=6,ncol=2); fig.tight_layout(); fig.savefig(output/"session_coordinate_audit.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(max(1,len(examples)),2,figsize=(12,max(3,2.2*len(examples))),squeeze=False)
    for axrow,ex in zip(axes,examples.itertuples()):
        ax=axrow[0]
        subset=predictions[(predictions.record_id.isin([ex.record_a,ex.record_b])) & predictions.model.isin(["M3_strong_local","M5_future"])].sort_values("s")
        for (record,model),g in subset.groupby(["record_id","model"]): ax.plot(g.s,g.psi_pred,label=f"{record} {model}")
        truth=subset.drop_duplicates(["record_id","segment_index"])
        for record,g in truth.groupby("record_id"): ax.plot(g.s,g.psi_true,"--",alpha=.7,label=f"{record} true")
        ax.set(title=f"local d={ex.local_distance:.3g}, future d={ex.future_feature_distance:.3g}",ylabel="psi (rad)"); ax.legend(fontsize=6,ncol=2)
        path_ax=axrow[1]
        for record, start_s in ((ex.record_a,ex.s_a),(ex.record_b,ex.s_b)):
            path=data[(data.record_id==record) & (data.s>=start_s)].sort_values("s"); u=np.vstack(path.M1_pose)
            path_ax.plot(u[:,0],u[:,1],label=record)
        path_ax.set(title="actual U trajectory in each initial-tool frame",xlabel="relative x (m)",ylabel="relative y (m)"); path_ax.legend(fontsize=6)
    axes[-1,0].set(xlabel="normalized phase"); fig.tight_layout(); fig.savefig(output/"example_plan_context.png",dpi=160); plt.close(fig)


def representative_examples(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cross-bite, held-out-OOF local-similar/future-different pairs."""
    from sklearn.neighbors import NearestNeighbors
    selected=[]
    for (take, phase), g in data.groupby(["take", "phase"], sort=True):
        if g.parent_bite_id.nunique() < 2: continue
        local=np.vstack(g["M3_strong_local"]); future=np.vstack(g["M5_future"])[:,20:]
        scale=np.std(local,axis=0); scale[scale < 1e-9]=1.; standardized=local/scale
        fscale=np.std(future,axis=0); fscale[fscale < 1e-9]=1.
        count=min(48,len(g)); distances,neighbors=NearestNeighbors(n_neighbors=count).fit(standardized).kneighbors(standardized)
        candidates=[]
        for i in range(len(g)):
            for d,j in zip(distances[i,1:],neighbors[i,1:]):
                if g.iloc[i].parent_bite_id == g.iloc[j].parent_bite_id: continue
                future_distance=float(np.linalg.norm((future[i]-future[j])/fscale)); psi_difference=float(abs(np.angle(np.exp(1j*(g.iloc[i].psi_true-g.iloc[j].psi_true)))) )
                candidates.append((float(d),future_distance,psi_difference,i,j)); break
        if not candidates: continue
        local_cut=np.percentile([x[0] for x in candidates],25)
        eligible=[x for x in candidates if x[0] <= local_cut]
        d,fd,pd_,i,j=max(eligible,key=lambda x:x[1]*x[2])
        a,b=g.iloc[i],g.iloc[j]
        entry={"take":take,"phase":phase,"fold_id":take,"record_a":a.record_id,"frame_a":a.frame,"s_a":a.s,"psi_a":a.psi_true,"record_b":b.record_id,"frame_b":b.frame,"s_b":b.s,"psi_b":b.psi_true,"local_distance":d,"future_feature_distance":fd,"psi_difference_rad":pd_}
        for suffix,row in (("a",a),("b",b)):
            hit=predictions[(predictions.record_id==row.record_id)&(predictions.segment_index==row.segment_index)]
            for model in ("M3_strong_local","M5_future","M6_full_plan"):
                value=hit[hit.model==model].iloc[0]; entry[f"{suffix}_{model}_pred"]=float(value.psi_pred); entry[f"{suffix}_{model}_circular_error"]=float(value.circular_error)
        selected.append(entry)
    if not selected:
        return pd.DataFrame(columns=["take","phase","fold_id","record_a","frame_a","s_a","psi_a","record_b","frame_b","s_b","psi_b","local_distance","future_feature_distance","psi_difference_rad"])
    return pd.DataFrame(selected).sort_values(["future_feature_distance","psi_difference_rad"],ascending=False).head(5)


def process_phase3(phase1_dir: Path, output_dir: Path, cfg: Phase3Config, trace: dict | None = None) -> dict:
    phase1_dir, output_dir = Path(phase1_dir), Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); (output_dir/"predictions").mkdir(exist_ok=True)
    data, dataset = build_dataset(phase1_dir, cfg); audit, audit_rows = coordinate_audit(phase1_dir)
    prediction_rows=[]; metric_rows=[]; selected={}
    for held in sorted(data["take"].unique()):
        train, test=data[data["take"]!=held],data[data["take"]==held]
        if trace is not None: trace["current_outer"] = held
        for model in MODELS:
            alpha=choose_alpha(train,model,cfg.alphas,trace); selected[f"{held}:{model}"]=alpha
            pred=_fit_predict(train,test,model,alpha,trace); ce=circular_error(pred,test.psi_true.to_numpy(float))
            prediction_rows.extend({"record_id":r.record_id,"parent_bite_id":r.parent_bite_id,"take":r.take,"demonstrator":r.demonstrator,"phase":r.phase,"frame":r.frame,"motive_frame":r.frame,"segment_index":r.segment_index,"time":r.time,"s":r.s,"psi_run_id":r.psi_run_id,"psi0":r.psi0,"fold_id":held,"valid_target":True,"psi_true":r.psi_true,"psi_pred":float(q),"delta_pred":float(q-r.psi0),"model":model,"circular_error":float(e),"circular_abs_error":float(abs(e)),"delta_psi":r.delta_psi,"delta_error":float(q-r.psi_true)} for r,q,e in zip(test.itertuples(),pred,ce))
    predictions=pd.DataFrame(prediction_rows); predictions.to_csv(output_dir/"predictions"/"oof_predictions.csv",index=False)
    for model,g in predictions.groupby("model"):
        for grouping,column in (("overall",None),("phase","phase"),("take","take"),("demonstrator","demonstrator")):
            groups=[("overall",g)] if column is None else list(g.groupby(column))
            for name, pg in groups:
                metric_rows.append(_prediction_metric_row(model, grouping, str(name), pg))
    metrics=pd.DataFrame(metric_rows); metrics.to_csv(output_dir/"fold_metrics.csv",index=False)
    bite_metrics=predictions.groupby(["model","parent_bite_id","take","phase"],as_index=False).agg(circular_mae=("circular_abs_error","mean"),circular_rmse=("circular_error",lambda x:float(np.sqrt(np.mean(x*x)))),n_frames=("frame","size")); bite_metrics.to_csv(output_dir/"bite_metrics.csv",index=False)
    boot=_bootstrap(predictions,cfg); examples=representative_examples(data,predictions); examples.to_csv(output_dir/"representative_examples.csv",index=False); _plots(output_dir,metrics,audit_rows,data,predictions,examples)
    gate_comparisons={"Gate_A_history_over_local":["M4_history vs M3_strong_local"],"Gate_B_future_over_history":["M5_future vs M4_history"],"Gate_C_full_plan_over_future_and_local":["M6_full_plan vs M5_future","M6_full_plan vs M3_strong_local"]}
    bootstrap_by_name={x["comparison"]:x for x in boot}
    gates={gate:[{"comparison":name,"mean_improves":bootstrap_by_name[name]["absolute_mae_difference_rad"] < 0,"ci95_supports_improvement":bootstrap_by_name[name]["absolute_ci95"][1] < 0,"absolute_mae_difference_rad":bootstrap_by_name[name]["absolute_mae_difference_rad"],"absolute_ci95":bootstrap_by_name[name]["absolute_ci95"],"interpretation":"95% CI below zero supports improvement; no invented threshold"} for name in names] for gate,names in gate_comparisons.items()}
    root=Path(__file__).resolve().parents[2]; dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True,check=False).stdout.strip())
    summary={"schema_version":"phase3-v1","phase1_manifest":dataset["source_manifest"],"dataset":dataset,"coordinate_audit":audit,"feature_representation":{"relative_tool":"p_rel=R0.T@(p-p0), r_rel=log(R0.T@R)","velocity":"causal immediate measured predecessor; linear in initial frame, angular body log(R_prev.T@R)/dt","history_offsets_s":HISTORY_OFFSETS_S,"future":"three U-only samples at 25/50/100% remaining phase plus path length","full_plan":"12-basis cubic B-spline coefficients of six U-only relative channels","excluded_inputs":["mouth_target_position","Phase 2 status","session/demonstrator/recipient identifiers"]},"outer_folds":sorted(data["take"].unique()),"selected_alphas":selected,"bootstrap":boot,"gates":gates,"overall":metrics[metrics.grouping=="overall"].to_dict("records"),"per_phase":metrics[metrics.grouping=="phase"].to_dict("records"),"per_take":metrics[metrics.grouping=="take"].to_dict("records"),"per_demonstrator":metrics[metrics.grouping=="demonstrator"].to_dict("records"),"provenance":{"code_revision":_revision(root),"code_worktree_dirty":dirty,"config_hash":hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()}}
    summary["representative_examples"] = examples.to_dict("records")
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); (output_dir/"manifest.json").write_text(json.dumps({"schema_version":"phase3-v1","summary":"summary.json","predictions":"predictions/oof_predictions.csv","fold_metrics":"fold_metrics.csv","bite_metrics":"bite_metrics.csv","examples":"representative_examples.csv","config":asdict(cfg),"source_phase1_manifest_sha256":hashlib.sha256((phase1_dir/"manifest.json").read_bytes()).hexdigest()},indent=2,sort_keys=True))
    return summary
