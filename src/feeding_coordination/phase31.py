"""Phase 3.1: nested local-ablation audit, using Phase 1.5 human data only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase1 import _revision
from .phase3 import (Phase3Config, _fit_predict, _prediction_metric_row, build_dataset,
                     bite_weights, circular_error, process_phase3)

LOCAL_MODELS = ("L0_kinematic", "L1_temporal", "L2_temporal_psi0", "L3_temporal_plate", "L4_current_m3")
STAR_MODELS = ("StrongLocal_star", "H_star", "F_star", "P_star")
ORIGINAL_MODELS = ("M2_pose_velocity", "M3_strong_local", "M4_history", "M5_future", "M6_full_plan")


@dataclass(frozen=True)
class Phase31Config:
    phase1_output_path: str = "outputs/phase1"
    phase3_output_path: str = "outputs/phase3"
    output_path: str = "outputs/phase31"
    alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1., 10., 100.)
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260915
    spline_basis_functions: int = 12

    @classmethod
    def load(cls, path: Path) -> "Phase31Config":
        raw = json.loads(Path(path).read_text()); raw["alphas"] = tuple(raw.get("alphas", cls.alphas))
        cfg = cls(**raw)
        if cfg.spline_basis_functions != 12: raise ValueError("Phase 3.1 freezes 12 cubic B-spline bases")
        if not cfg.alphas or any(x <= 0 for x in cfg.alphas): raise ValueError("alphas must be positive")
        return cfg


def add_local_candidates(data: pd.DataFrame) -> pd.DataFrame:
    """Attach exactly the five pre-registered local feature blocks."""
    out = data.copy()
    for name, stop in (("L0_kinematic", 12), ("L1_temporal", 15), ("L2_temporal_psi0", 20),
                       ("L3_temporal_plate", 19), ("L4_current_m3", 20)):
        if name == "L2_temporal_psi0":
            out[name] = [np.r_[x[:15], x[19]] for x in out.M3_strong_local]
        elif name == "L3_temporal_plate":
            out[name] = [x[:19].copy() for x in out.M3_strong_local]
        else:
            source = "M2_pose_velocity" if name == "L0_kinematic" else "M3_strong_local"
            out[name] = [x[:stop].copy() for x in out[source]]
    return out


def _bite_rmse(frame: pd.DataFrame, pred: np.ndarray) -> float:
    ce = circular_error(pred, frame.psi_true.to_numpy(float))
    return float(np.sqrt(pd.DataFrame({"bite": frame.parent_bite_id, "sq": ce * ce}).groupby("bite").sq.mean().mean()))


def select_strong_local(train: pd.DataFrame, alphas: tuple[float, ...], trace: dict | None = None) -> tuple[str, float, pd.DataFrame]:
    """Joint inner LOTO selection.  `train` is already outer-held-out free."""
    rows = []
    for model in LOCAL_MODELS:
        for alpha in alphas:
            parts = []
            for held in sorted(train["take"].unique()):
                inner_train, valid = train[train["take"] != held], train[train["take"] == held]
                if trace is not None:
                    trace.setdefault("inner_selection_events", []).append({"outer_held": trace.get("current_outer"), "inner_held": held, "fit_takes": set(inner_train["take"]), "valid_takes": set(valid["take"]), "model": model, "alpha": alpha})
                parts.append((valid, _fit_predict(inner_train, valid, model, alpha)))
            pooled = pd.concat([x[0] for x in parts]); pred = np.concatenate([x[1] for x in parts])
            rows.append({"model": model, "alpha": float(alpha), "inner_bite_circular_rmse": _bite_rmse(pooled, pred)})
    scores = pd.DataFrame(rows).sort_values(["inner_bite_circular_rmse", "model", "alpha"], kind="stable")
    best = scores.iloc[0]
    return str(best.model), float(best.alpha), scores


def _star_features(frame: pd.DataFrame, local: str) -> pd.DataFrame:
    out = frame.copy(); base = out[local].tolist()
    out["StrongLocal_star"] = [x.copy() for x in base]
    out["H_star"] = [np.r_[b, old[20:]] for b, old in zip(base, out.M4_history)]
    out["F_star"] = [np.r_[b, old[20:]] for b, old in zip(base, out.M5_future)]
    out["P_star"] = [np.r_[b, old[20:]] for b, old in zip(base, out.M6_full_plan)]
    return out


def _choose_alpha_fixed(train: pd.DataFrame, model: str, alphas: tuple[float, ...]) -> float:
    scores=[]
    for alpha in alphas:
        pieces=[]
        for held in sorted(train["take"].unique()):
            itr, val = train[train["take"] != held], train[train["take"] == held]
            pieces.append((val, _fit_predict(itr, val, model, alpha)))
        joined=pd.concat([x[0] for x in pieces]); scores.append(_bite_rmse(joined, np.concatenate([x[1] for x in pieces])))
    return float(alphas[int(np.argmin(scores))])


def feature_shift(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    specs={"s": lambda g: np.asarray(g.s, float)[:,None], "plate_rel": lambda g: np.vstack(g.L3_temporal_plate)[:,15:18],
           "psi0": lambda g: np.asarray(g.psi0, float)[:,None]}
    rows=[]
    for held in sorted(data["take"].unique()):
        train, test=data[data["take"] != held], data[data["take"] == held]
        for name, extract in specs.items():
            a,b=extract(train),extract(test); mean=a.mean(0); std=a.std(0); safe=np.where(std < 1e-12, 1., std)
            z=np.abs((b-mean)/safe); lo,hi=a.min(0),a.max(0)
            rows.append({"take":held,"feature_group":name,"train_mean":mean.tolist(),"train_std":std.tolist(),"test_mean":b.mean(0).tolist(),"test_std":b.std(0).tolist(),"median_abs_z":float(np.median(z)),"p95_abs_z":float(np.percentile(z,95)),"max_abs_z":float(z.max()),"fraction_outside_train_range":float(np.mean((b < lo) | (b > hi)))})
    # Geometric-only plate diagnostic: no classifier or identity feature.
    centers=[]
    for take,g in data.groupby("take"):
        # Each record contributes once: plate_rel is segment-constant and frames must not weight it.
        points=np.vstack(g.groupby("record_id", sort=True).L3_temporal_plate.first())[:,15:18]
        centers.append((take,points.mean(0),float(np.sqrt(np.mean(np.sum((points-points.mean(0))**2,axis=1)))),len(points)))
    plate=[]
    for take,c,spread,n_segments in centers:
        distances={other:float(np.linalg.norm(c-c2)) for other,c2,_,_ in centers if other != take}
        plate.append({"take":take,"plate_rel_centroid":c.tolist(),"within_take_spread":spread,"n_segments":n_segments,"weighting":"one plate_rel sample per record/segment (not frame weighted)","pairwise_centroid_distances":distances,"nearest_training_centroid_distance":min(distances.values()) if distances else np.nan})
    return pd.DataFrame(rows), {"plate_rel_geometry":plate}


def psi0_distribution(data: pd.DataFrame) -> list[dict]:
    out=[]
    for (take,phase),g in data.groupby(["take","phase"]):
        x=g.psi0.to_numpy(float); out.append({"take":take,"phase":phase,"median":float(np.median(x)),"iqr":float(np.percentile(x,75)-np.percentile(x,25)),"range":[float(x.min()),float(x.max())]})
    return out


def paired_bootstrap(predictions: pd.DataFrame, cfg: Phase31Config) -> list[dict]:
    pairs=[("H_star","StrongLocal_star"),("F_star","StrongLocal_star"),("P_star","StrongLocal_star"),("F_star","H_star"),("P_star","H_star"),("P_star","F_star")]
    rng=np.random.default_rng(cfg.bootstrap_seed); result=[]; keys=["parent_bite_id","record_id","segment_index"]
    for candidate,baseline in pairs:
        a=predictions[predictions.model==candidate]; b=predictions[predictions.model==baseline][keys+["circular_error"]]
        j=a.merge(b,on=keys,suffixes=("_new","_base"),validate="one_to_one").groupby("parent_bite_id")
        mae_new=j.circular_error_new.apply(lambda x:np.mean(np.abs(x))).to_numpy(); mae_base=j.circular_error_base.apply(lambda x:np.mean(np.abs(x))).to_numpy()
        # Preserve bite MSE.  The aggregate and every bootstrap draw take the
        # square root only after averaging bite MSE, matching circular RMSE.
        mse_new=j.circular_error_new.apply(lambda x:np.mean(x*x)).to_numpy(); mse_base=j.circular_error_base.apply(lambda x:np.mean(x*x)).to_numpy()
        draw=rng.integers(0,len(mae_new),(cfg.bootstrap_resamples,len(mae_new)))
        row={"comparison":f"{candidate} vs {baseline}","convention":"negative means candidate improves","n_parent_bites":len(mae_new)}
        for name,new,old in (("mae",mae_new,mae_base),("rmse",mse_new,mse_base)):
            if name == "rmse":
                point_new,point_old=np.sqrt(new.mean()),np.sqrt(old.mean())
                samples=np.sqrt(new[draw].mean(1))-np.sqrt(old[draw].mean(1))
                relative=100*(np.sqrt(new[draw].mean(1))-np.sqrt(old[draw].mean(1)))/np.sqrt(old[draw].mean(1))
            else:
                point_new,point_old=new.mean(),old.mean()
                samples=(new-old)[draw].mean(1); relative=100*(new[draw].mean(1)-old[draw].mean(1))/old[draw].mean(1)
            diff=point_new-point_old
            row[f"{name}_difference_rad"]=float(diff); row[f"{name}_relative_percent_change"]=float(100*diff/point_old); row[f"{name}_difference_ci95"]=np.percentile(samples,[2.5,97.5]).tolist(); row[f"{name}_relative_percent_ci95"]=np.percentile(relative,[2.5,97.5]).tolist()
        result.append(row)
    return result


def _prediction_rows(frame: pd.DataFrame, pred: np.ndarray, model: str, fold: str) -> list[dict]:
    ce=circular_error(pred,frame.psi_true.to_numpy(float)); out=[]
    for r,q,e in zip(frame.itertuples(),pred,ce):
        out.append({"record_id":r.record_id,"parent_bite_id":r.parent_bite_id,"take":r.take,"phase":r.phase,"frame":r.frame,"motive_frame":r.frame,"segment_index":r.segment_index,"time":r.time,"s":r.s,"psi_run_id":r.psi_run_id,"fold_id":fold,"model":model,"psi_true":r.psi_true,"psi_pred":float(q),"circular_error":float(e),"circular_abs_error":float(abs(e)),"delta_error":float(q-r.psi_true)})
    return out


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for model,g in predictions.groupby("model"):
        for grouping,col in (("overall",None),("phase","phase"),("take","take")):
            groups=[("overall",g)] if col is None else g.groupby(col)
            for name,pg in groups: rows.append(_prediction_metric_row(model,grouping,str(name),pg))
    return pd.DataFrame(rows)


def psi0_effects(local_metrics: pd.DataFrame) -> list[dict]:
    """Positive effects mean that adding psi0 made the circular error worse."""
    out=[]
    for without,with_ in (("L1_temporal","L2_temporal_psi0"),("L3_temporal_plate","L4_current_m3")):
        a=local_metrics[local_metrics.model==without]; b=local_metrics[local_metrics.model==with_]
        joined=b.merge(a,on=["grouping","group"],suffixes=("_with","_without"),validate="one_to_one")
        for r in joined.itertuples():
            out.append({"comparison":f"{with_} minus {without}","grouping":r.grouping,"group":r.group,"convention":"positive means adding psi0 worsens error","circular_rmse_effect_rad":float(r.circular_rmse_with-r.circular_rmse_without),"circular_mae_effect_rad":float(r.circular_mae_with-r.circular_mae_without)})
    return out


def star_gates(bootstrap: list[dict]) -> dict:
    by_name={x["comparison"]:x for x in bootstrap}
    def judgment(name: str) -> dict:
        item=by_name[name]; return {"comparison":name,"metric":"circular MAE","reliably_improves":bool(item["mae_difference_ci95"][1] < 0),"mae_difference_rad":item["mae_difference_rad"],"mae_difference_ci95":item["mae_difference_ci95"]}
    return {"Gate_A_star_history_over_strong_local":[judgment("H_star vs StrongLocal_star")],
            "Gate_B_star_future_over_strong_local_or_history":[judgment("F_star vs StrongLocal_star"),judgment("F_star vs H_star")],
            "Gate_C_star_plan_over_strong_local_history_or_future":[judgment("P_star vs StrongLocal_star"),judgment("P_star vs H_star"),judgment("P_star vs F_star")]}


def _max_numeric_difference(a, b) -> float:
    if isinstance(a, dict):
        if set(a) != set(b): return np.inf
        return max([_max_numeric_difference(a[k],b[k]) for k in a] or [0.])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return np.inf
        return max([_max_numeric_difference(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a, (int,float,np.number)) and isinstance(b,(int,float,np.number)):
        if np.isnan(float(a)) and np.isnan(float(b)): return 0.
        if np.isnan(float(a)) or np.isnan(float(b)): return np.inf
        return float(abs(float(a)-float(b)))
    return 0. if a == b else np.inf


def phase3_reproduction_check(reproduced: dict, official_summary: Path, atol: float = 1e-10) -> dict:
    """Compare all requested Phase-3 summary metrics and selection results."""
    official=json.loads(official_summary.read_text())
    fields=("overall","per_phase","per_take","selected_alphas","bootstrap")
    diffs={field:_max_numeric_difference(reproduced[field],official[field]) for field in fields}
    maximum=max(diffs.values())
    return {"performed":True,"atol":atol,"maximum_absolute_difference":maximum,"per_field_maximum_absolute_difference":diffs,"matches_existing_phase3":bool(maximum <= atol)}


def _plots(output: Path, local: pd.DataFrame, shift: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig,ax=plt.subplots(figsize=(8,4)); o=local[local.grouping=="overall"]; ax.bar(o.model,o.circular_rmse); ax.tick_params(axis="x",rotation=25); ax.set_ylabel("bite-balanced circular RMSE (rad)"); fig.tight_layout(); fig.savefig(output/"local_ablation.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for name,g in shift.groupby("feature_group"): ax.plot(g["take"],g["p95_abs_z"],marker="o",label=name)
    ax.set(ylabel="p95 |standardized z|",xlabel="held-out take"); ax.legend(); fig.tight_layout(); fig.savefig(output/"feature_shift_by_take.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); o=metrics[(metrics.grouping=="overall") & metrics.model.isin(STAR_MODELS)]; ax.bar(o.model,o.circular_rmse); ax.tick_params(axis="x",rotation=25); ax.set_ylabel("bite-balanced circular RMSE (rad)"); fig.tight_layout(); fig.savefig(output/"corrected_information_hierarchy.png",dpi=160); plt.close(fig)


def process_phase31(phase1_dir: Path, output_dir: Path, cfg: Phase31Config, trace: dict | None = None) -> dict:
    """Run the Phase 3.1 experiment.  Only Phase 1.5 and Phase-3 source code are read."""
    phase1_dir,output_dir=Path(phase1_dir),Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"predictions").mkdir(exist_ok=True)
    p3cfg=Phase3Config(phase1_output_path=str(phase1_dir),alphas=cfg.alphas,bootstrap_resamples=cfg.bootstrap_resamples,bootstrap_seed=cfg.bootstrap_seed,spline_basis_functions=cfg.spline_basis_functions)
    # Reproduce current Phase 3 in a temporary directory, never touching original outputs.
    with tempfile.TemporaryDirectory(prefix="phase31_phase3_") as tmp:
        reproduced=process_phase3(phase1_dir,Path(tmp),p3cfg,trace=None)
        original_predictions=pd.read_csv(Path(tmp)/"predictions"/"oof_predictions.csv")
    root=Path(__file__).resolve().parents[2]
    phase3_path=Path(cfg.phase3_output_path); phase3_path=phase3_path if phase3_path.is_absolute() else root/phase3_path
    official=phase3_path/"summary.json"
    reproduction={"performed":True,"matches_existing_phase3":None}
    if official.exists():
        reproduction=phase3_reproduction_check(reproduced,official)
        if not reproduction["matches_existing_phase3"]: raise RuntimeError("Phase 3 reproduction does not match existing Phase-3 metrics")
    data,dataset=build_dataset(phase1_dir,p3cfg); data=add_local_candidates(data); all_rows=[]; nested=[]
    for held in sorted(data["take"].unique()):
        train,test=data[data["take"]!=held],data[data["take"]==held]
        if trace is not None: trace["current_outer"]=held
        local,alpha,scores=select_strong_local(train,cfg.alphas,trace); nested.extend([{"outer_held_take":held,**r} for r in scores.to_dict("records")])
        train_s,test_s=_star_features(train,local),_star_features(test,local)
        selected={"StrongLocal_star":alpha}
        for model in ("H_star","F_star","P_star"): selected[model]=_choose_alpha_fixed(train_s,model,cfg.alphas)
        for model,a in selected.items(): all_rows.extend(_prediction_rows(test_s,_fit_predict(train_s,test_s,model,a),model,held))
        # Original Phase-3 OOF is retained only as a transparent control.
        for model,g in original_predictions[original_predictions.model.isin(ORIGINAL_MODELS)].groupby("model"):
            all_rows.extend(g[g["take"]==held].to_dict("records"))
        nested.append({"outer_held_take":held,"model":"SELECTED","alpha":alpha,"selected_local":local,"inner_bite_circular_rmse":float(scores.iloc[0].inner_bite_circular_rmse),"star_alphas":selected})
    # Local ablations require OOF predictions in addition to selected hierarchy.
    local_rows=[]
    for held in sorted(data["take"].unique()):
        train,test=data[data["take"]!=held],data[data["take"]==held]
        for model in LOCAL_MODELS:
            a=_choose_alpha_fixed(train,model,cfg.alphas); local_rows.extend(_prediction_rows(test,_fit_predict(train,test,model,a),model,held))
    local_pred=pd.DataFrame(local_rows)
    all_rows.extend(local_rows)
    predictions=pd.DataFrame(all_rows); predictions.to_csv(output_dir/"predictions"/"oof_predictions.csv",index=False)
    metrics=_metrics(predictions); metrics.to_csv(output_dir/"fold_metrics.csv",index=False)
    local_metrics=metrics[metrics.model.isin(LOCAL_MODELS)].copy(); local_metrics.to_csv(output_dir/"local_ablation.csv",index=False)
    nested_df=pd.DataFrame(nested); nested_df.to_csv(output_dir/"nested_selection.csv",index=False)
    bite=predictions.groupby(["model","parent_bite_id","take","phase"],as_index=False).agg(circular_mae=("circular_abs_error","mean"),circular_rmse=("circular_error",lambda x:float(np.sqrt(np.mean(x*x)))),n_frames=("frame","size")); bite.to_csv(output_dir/"bite_metrics.csv",index=False)
    shift,plate=feature_shift(data); boot=paired_bootstrap(predictions,cfg); gates=star_gates(boot); effects=psi0_effects(local_metrics); _plots(output_dir,local_metrics,shift,metrics)
    dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True).stdout.strip())
    summary={"schema_version":"phase31-v1","phase3_reproduction":reproduction,"dataset":dataset,"local_candidates":{"L0":"M2","L1":"M2+s+phase","L2":"L1+psi0","L3":"L1+plate_rel+plate_mask","L4":"current M3"},"feature_shift":shift.to_dict("records"),"plate_diagnostic":plate,"psi0_by_take_phase":psi0_distribution(data),"nested_selection":nested_df.to_dict("records"),"local_ablation":{"overall":local_metrics[local_metrics.grouping=="overall"].to_dict("records"),"per_phase":local_metrics[local_metrics.grouping=="phase"].to_dict("records"),"per_take":local_metrics[local_metrics.grouping=="take"].to_dict("records")},"psi0_effects":effects,"bootstrap":boot,"gates":gates,"overall":metrics[metrics.grouping=="overall"].to_dict("records"),"per_phase":metrics[metrics.grouping=="phase"].to_dict("records"),"per_take":metrics[metrics.grouping=="take"].to_dict("records"),"psi_dot_audit":{"secondary_diagnostic":True,"ordering":"original record then segment_index","boundary_guards":["same record_id","adjacent segment_index","adjacent motive_frame","same psi_run_id","strictly increasing actual timestamp"],"timestamps":"actual recorded time; no prediction smoothing or derivative optimization"},"feature_representation":{"target":"delta_psi=psi-psi0; psi_pred=psi0+delta_pred","future_and_plan":"existing U-only Phase-3 suffixes","excluded_inputs":["Phase 2 feasibility","mouth target","session identity"]},"provenance":{"code_revision":_revision(root),"code_worktree_dirty":dirty,"config_hash":hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()}}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
