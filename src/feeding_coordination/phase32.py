"""Phase 3.2: simple baselines and frozen deployable local redundancy model.

This module deliberately reads only the immutable Phase 1.5 human records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib, json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .phase1 import _revision
from .phase3 import Phase3Config, build_dataset, bite_weights, circular_error, _prediction_metric_row

MODELS = ("R0_hold_psi0", "R1_phase_only", "R2_pose_only", "R3_pose_velocity", "StrongLocal")
STRONG_LOCAL_ORDER = ("relative_tool_position_x", "relative_tool_position_y", "relative_tool_position_z",
    "relative_tool_rotation_x", "relative_tool_rotation_y", "relative_tool_rotation_z",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "normalized_phase_s", "phase_transfer", "phase_withdrawal", "plate_relative_x", "plate_relative_y", "plate_relative_z", "plate_valid_mask")

@dataclass(frozen=True)
class Phase32Config:
    phase1_output_path: str = "outputs/phase1"
    phase31_output_path: str = "outputs/phase31"
    output_path: str = "outputs/phase32"
    alphas: tuple[float, ...] = (0.001, .01, .1, 1., 10., 100.)
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260915
    spline_basis_functions: int = 12
    @classmethod
    def load(cls, path: Path) -> "Phase32Config":
        raw=json.loads(Path(path).read_text()); raw["alphas"]=tuple(raw.get("alphas",cls.alphas)); cfg=cls(**raw)
        if not cfg.alphas or any(a <= 0 for a in cfg.alphas): raise ValueError("alphas must be positive")
        if cfg.spline_basis_functions != 12: raise ValueError("Phase 3.2 freezes the Phase-3 12-basis setting")
        return cfg

def add_baseline_features(data: pd.DataFrame) -> pd.DataFrame:
    out=data.copy()
    out["R1_phase_only"]=[np.array([r.s, r.phase == "transfer", r.phase == "withdrawal"],float) for r in out.itertuples()]
    out["R2_pose_only"]=[np.asarray(x,float).copy() for x in out.M1_pose]
    out["R3_pose_velocity"]=[np.asarray(x,float).copy() for x in out.M2_pose_velocity]
    out["StrongLocal"]=[np.asarray(x,float)[:19].copy() for x in out.M3_strong_local]
    if any(len(x) != 19 for x in out.StrongLocal): raise ValueError("frozen StrongLocal contract must have 19 features")
    return out

def _fit_components(train: pd.DataFrame, model: str, alpha: float):
    X=np.vstack(train[model]); y=train.delta_psi.to_numpy(float); w=bite_weights(train)
    mean=np.average(X,axis=0,weights=w); scale=np.sqrt(np.maximum(np.average((X-mean)**2,axis=0,weights=w),0)); scale[scale < 1e-12]=1.
    fit=Ridge(alpha=alpha).fit((X-mean)/scale,y,sample_weight=w)
    return mean,scale,fit

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str, alpha: float, trace: dict | None=None) -> np.ndarray:
    if trace is not None: trace.setdefault("fit_events",[]).append({"outer_held":trace.get("current_outer"),"fit_takes":set(train["take"]),"scaler_takes":set(train["take"]),"model":model})
    mean,scale,fit=_fit_components(train,model,alpha)
    return test.psi0.to_numpy(float)+fit.predict((np.vstack(test[model])-mean)/scale)

def r0_predict(frame: pd.DataFrame) -> np.ndarray: return frame.psi0.to_numpy(float).copy()

def _bite_rmse(frame: pd.DataFrame, pred: np.ndarray) -> float:
    e=circular_error(pred,frame.psi_true.to_numpy(float)); return float(np.sqrt(pd.DataFrame({"bite":frame.parent_bite_id,"sq":e*e}).groupby("bite").sq.mean().mean()))

def choose_alpha(train: pd.DataFrame, model: str, alphas: tuple[float,...], trace:dict|None=None) -> float:
    scores=[]
    for alpha in alphas:
        frames=[]; predictions=[]
        for held in sorted(train["take"].unique()):
            itr,val=train[train["take"] != held],train[train["take"] == held]
            if trace is not None: trace.setdefault("inner_selection_events",[]).append({"outer_held":trace.get("current_outer"),"inner_held":held,"fit_takes":set(itr["take"]),"valid_takes":set(val["take"]),"model":model,"alpha":alpha})
            frames.append(val); predictions.append(fit_predict(itr,val,model,alpha,trace))
        scores.append(_bite_rmse(pd.concat(frames),np.concatenate(predictions)))
    return float(alphas[int(np.argmin(scores))])

def _prediction_rows(frame,pred,model,fold):
    ce=circular_error(pred,frame.psi_true.to_numpy(float)); return [{"record_id":r.record_id,"parent_bite_id":r.parent_bite_id,"take":r.take,"phase":r.phase,"frame":r.frame,"motive_frame":r.frame,"segment_index":r.segment_index,"time":r.time,"s":r.s,"psi_run_id":r.psi_run_id,"fold_id":fold,"model":model,"psi_true":r.psi_true,"psi0":r.psi0,"delta_psi":r.delta_psi,"psi_pred":float(q),"delta_pred":float(q-r.psi0),"circular_error":float(e),"circular_abs_error":float(abs(e)),"delta_error":float(q-r.psi_true)} for r,q,e in zip(frame.itertuples(),pred,ce)]

def metrics(predictions):
    rows=[]
    for model,g in predictions.groupby("model"):
        for grouping,col in (("overall",None),("phase","phase"),("take","take")):
            for name,pg in ([('overall',g)] if col is None else g.groupby(col)): rows.append(_prediction_metric_row(model,grouping,str(name),pg))
    return pd.DataFrame(rows)

def paired_bootstrap(predictions,cfg):
    rng=np.random.default_rng(cfg.bootstrap_seed); out=[]; keys=["parent_bite_id","record_id","segment_index"]
    for base in MODELS[:-1]:
        a=predictions[predictions.model == "StrongLocal"]; b=predictions[predictions.model == base][keys+["circular_error"]]
        j=a.merge(b,on=keys,suffixes=("_new","_base"),validate="one_to_one").groupby("parent_bite_id")
        mae_new=j.circular_error_new.apply(lambda x:np.mean(abs(x))).to_numpy(); mae_base=j.circular_error_base.apply(lambda x:np.mean(abs(x))).to_numpy()
        mse_new=j.circular_error_new.apply(lambda x:np.mean(x*x)).to_numpy(); mse_base=j.circular_error_base.apply(lambda x:np.mean(x*x)).to_numpy(); draw=rng.integers(0,len(mae_new),(cfg.bootstrap_resamples,len(mae_new)))
        row={"comparison":f"StrongLocal vs {base}","convention":"negative means StrongLocal improves","n_parent_bites":len(mae_new)}
        for label,new,old in (("mae",mae_new,mae_base),("rmse",mse_new,mse_base)):
            if label == "mae": point_new,point_old=new.mean(),old.mean(); samples=(new[draw]-old[draw]).mean(1); rel=100*(new[draw]-old[draw]).mean(1)/old[draw].mean(1)
            else: point_new,point_old=np.sqrt(new.mean()),np.sqrt(old.mean()); samples=np.sqrt(new[draw].mean(1))-np.sqrt(old[draw].mean(1)); rel=100*samples/np.sqrt(old[draw].mean(1))
            diff=point_new-point_old; row.update({f"{label}_difference_rad":float(diff),f"{label}_relative_percent_change":float(100*diff/point_old),f"{label}_difference_ci95":np.percentile(samples,[2.5,97.5]).tolist(),f"{label}_relative_percent_ci95":np.percentile(rel,[2.5,97.5]).tolist()})
        out.append(row)
    return out

def psi_variation(data):
    rows=[]
    for (bite,phase),g in data.groupby(["parent_bite_id","phase"]):
        g=g.sort_values(["record_id","segment_index"]); psi=g.psi_true.to_numpy(float); delta=g.delta_psi.to_numpy(float); travel=0.
        a=g.iloc[:-1]; b=g.iloc[1:]; good=(a.record_id.to_numpy()==b.record_id.to_numpy()) & (np.diff(g.segment_index)==1) & (np.diff(g.frame)==1) & (a.psi_run_id.to_numpy()==b.psi_run_id.to_numpy())
        if good.any(): travel=float(np.abs(np.diff(psi)[good]).sum())
        rows.append({"parent_bite_id":bite,"phase":phase,"psi_range":float(psi.max()-psi.min()),"delta_psi_range":float(delta.max()-delta.min()),"rms_change_from_psi0":float(np.sqrt(np.mean(delta*delta))),"max_abs_change_from_psi0":float(abs(delta).max()),"total_abs_psi_travel":travel,"n_frames":len(g)})
    per=pd.DataFrame(rows); summary=[]
    quantities=("psi_range","delta_psi_range","rms_change_from_psi0","max_abs_change_from_psi0","total_abs_psi_travel")
    # Overall is parent-bite balanced even when a bite lacks an eligible phase.
    overall=per.groupby("parent_bite_id",as_index=False)[list(quantities)].mean()
    summary.append({"grouping":"overall","group":"overall","n_parent_bites":int(len(overall)),"n_bite_phase_units":int(len(per)),**{k:float(overall[k].mean()) for k in quantities}})
    for group,g in per.groupby("phase"):
        summary.append({"grouping":"phase","group":str(group),"n_parent_bites":int(g.parent_bite_id.nunique()),"n_bite_phase_units":int(len(g)),**{k:float(g[k].mean()) for k in quantities}})
    return per,pd.DataFrame(summary)

def _plots(output, metrics_df, variation_df):
    fig,ax=plt.subplots(figsize=(8,4)); x=metrics_df[metrics_df.grouping=="overall"]; ax.bar(x.model,x.circular_rmse); ax.set(ylabel="bite-balanced circular RMSE (rad)"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(output/"baseline_comparison.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); x=variation_df[variation_df.grouping=="phase"]; ax.bar(x.group,x.rms_change_from_psi0); ax.set(ylabel="bite-balanced RMS change from psi0 (rad)"); fig.tight_layout(); fig.savefig(output/"psi_variation.png",dpi=160); plt.close(fig)

def save_final_model(data, alpha, output, manifest_path):
    mean,scale,fit=_fit_components(data,"StrongLocal",alpha); X=np.vstack(data.StrongLocal)
    np.savez(output/"model.npz",mean=mean,scale=scale,coef=fit.coef_,intercept=np.asarray(fit.intercept_),alpha=np.asarray(alpha))
    delta_memory=fit.predict((X-mean)/scale); delta_replay=(X-mean)/scale @ fit.coef_+fit.intercept_
    max_delta_prediction_error=float(np.max(np.abs(delta_memory-delta_replay)))
    if max_delta_prediction_error > 1e-12: raise RuntimeError("saved linear model fails delta-prediction reproduction")
    contract={"schema_version":"phase32-final-local-ridge-v1","feature_contract_version":"strong-local-l3-v1","model":"StrongLocal","feature_order":list(STRONG_LOCAL_ORDER),"feature_count":19,"input_blocks":{"relative_tool_pose":"p_rel=R0.T@(p-p0), rotvec(log(R0.T@R))","velocity":"causal immediate measured predecessor in segment-local coordinates","phase":"normalized s plus transfer/withdrawal one-hot","plate_context":"segment-local plate-relative position plus validity mask"},"target":"delta_psi=psi_unwrapped-psi0","output":"predicted_delta_psi","absolute_reconstruction":"psi_robot=psi_robot0+predicted_delta_psi","psi_robot0":"initial redundancy offset for reconstruction, not an ML feature","alpha":float(alpha),"training_scaler":{"mean_key":"mean","scale_key":"scale"},"ridge":{"coef_key":"coef","intercept_key":"intercept"},"training_source_manifest_sha256":hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),"code_revision":_revision(Path(__file__).resolve().parents[2]),"delta_prediction_replay_max_abs_error":max_delta_prediction_error,"delta_prediction_replay_tolerance":1e-12}
    (output/"model_contract.json").write_text(json.dumps(contract,indent=2,sort_keys=True)); return contract

def phase31_secondary_control(summary_path: Path):
    """Retain only the pre-existing H* overall result and Gate A* decision."""
    raw=json.loads(Path(summary_path).read_text())
    hstar=[x for x in raw.get("overall",[]) if x.get("model")=="H_star"]
    return {"label":"secondary temporal control; not retrained in Phase 3.2","H_star_overall":hstar,
            "Gate_A_star_history_over_strong_local":raw.get("gates",{}).get("Gate_A_star_history_over_strong_local",[])}

def process_phase32(phase1_dir:Path, output_dir:Path, cfg:Phase32Config, trace:dict|None=None):
    phase1_dir,output_dir=Path(phase1_dir),Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"predictions").mkdir(exist_ok=True); (output_dir/"final_model").mkdir(exist_ok=True)
    p3cfg=Phase3Config(phase1_output_path=str(phase1_dir),alphas=cfg.alphas,spline_basis_functions=cfg.spline_basis_functions); data,dataset=build_dataset(phase1_dir,p3cfg); data=add_baseline_features(data); rows=[]; selected={}
    for held in sorted(data["take"].unique()):
        train,test=data[data["take"] != held],data[data["take"] == held]
        if trace is not None: trace["current_outer"]=held
        for model in MODELS:
            if model == "R0_hold_psi0": pred=r0_predict(test)
            else:
                alpha=choose_alpha(train,model,cfg.alphas,trace); selected.setdefault(held,{})[model]=alpha; pred=fit_predict(train,test,model,alpha,trace)
            rows.extend(_prediction_rows(test,pred,model,held))
    predictions=pd.DataFrame(rows); predictions.to_csv(output_dir/"predictions"/"oof_predictions.csv",index=False); metric=metrics(predictions); metric.to_csv(output_dir/"fold_metrics.csv",index=False)
    bite=predictions.groupby(["model","parent_bite_id","take","phase"],as_index=False).agg(circular_mae=("circular_abs_error","mean"),circular_rmse=("circular_error",lambda x:float(np.sqrt(np.mean(x*x)))),delta_rmse=("delta_error",lambda x:float(np.sqrt(np.mean(x*x)))),n_frames=("frame","size"))
    variation,variation_summary=psi_variation(data); bite=bite.merge(variation,on=["parent_bite_id","phase"],how="left",validate="many_to_one"); bite.to_csv(output_dir/"bite_metrics.csv",index=False); variation.to_csv(output_dir/"psi_variation_by_bite.csv",index=False); boot=paired_bootstrap(predictions,cfg); _plots(output_dir,metric,variation_summary)
    deployment_alpha=choose_alpha(data,"StrongLocal",cfg.alphas); contract=save_final_model(data,deployment_alpha,output_dir/"final_model",phase1_dir/"manifest.json")
    phase31=Path(cfg.phase31_output_path); phase31=phase31 if phase31.is_absolute() else Path(__file__).resolve().parents[2]/phase31; hstar=None
    if (phase31/"summary.json").exists(): hstar=phase31_secondary_control(phase31/"summary.json")
    overall={r["model"]:r for r in metric[metric.grouping=="overall"].to_dict("records")}; ratio=overall["StrongLocal"]["circular_rmse"]/overall["R0_hold_psi0"]["circular_rmse"]
    by_boot={x["comparison"]:x for x in boot}; sl_r0=by_boot["StrongLocal vs R0_hold_psi0"]; sl_r1=by_boot["StrongLocal vs R1_phase_only"]
    summary={"schema_version":"phase32-v1","dataset":dataset,"models":{"R0_hold_psi0":"no fit; predicted delta psi is exactly zero","R1_phase_only":"three features [normalized_phase_s, is_transfer, is_withdrawal]; ridge target delta psi; training folds only","R2_pose_only":"Phase-3 M1_pose","R3_pose_velocity":"Phase-3 M2_pose_velocity","StrongLocal":"frozen Phase-3.1 L3, 19 features; no feature selection"},"selected_alphas":selected,"deployment_alpha":deployment_alpha,"overall":metric[metric.grouping=="overall"].to_dict("records"),"per_phase":metric[metric.grouping=="phase"].to_dict("records"),"per_take":metric[metric.grouping=="take"].to_dict("records"),"bootstrap":boot,"human_psi_variation":{"bite_level_file":"psi_variation_by_bite.csv","bite_balanced":variation_summary.to_dict("records"),"stronglocal_over_r0_rmse_ratio":float(ratio),"stronglocal_rmse_reduction_percent":float(100*(1-ratio)),"travel_boundary_guards":["same record_id","adjacent segment_index","adjacent motive_frame","same psi_run_id"]},"scientific_conclusions":{"H1":{"criterion":"supported if StrongLocal vs R0 paired bite-balanced circular MAE bootstrap CI upper bound is below zero, interpreted with bite-balanced human variation statistics","mae_ci95":sl_r0["mae_difference_ci95"],"supported":bool(sl_r0["mae_difference_ci95"][1] < 0),"variation":variation_summary.to_dict("records")},"H2_revised":{"criterion":"local information is supported only if StrongLocal has paired bite-balanced circular MAE bootstrap CI upper bound below zero against both R0 and R1","vs_R0_mae_ci95":sl_r0["mae_difference_ci95"],"vs_R1_mae_ci95":sl_r1["mae_difference_ci95"],"supported":bool(sl_r0["mae_difference_ci95"][1] < 0 and sl_r1["mae_difference_ci95"][1] < 0)},"original_full_plan":{"conclusion":"rejected/not supported by Phase 3.1; not resumed"}},"secondary_temporal_control_from_phase31":hstar,"full_plan_hypothesis":"rejected/not supported by Phase 3.1; not resumed","circular_psi0_diagnostic":{"run":False,"reason":"L3 excludes raw psi0; no broad feature search reopened"},"final_model_contract":contract,"provenance":{"code_revision":_revision(Path(__file__).resolve().parents[2]),"code_worktree_dirty":bool(subprocess.run(["git","status","--porcelain"],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True).stdout.strip()),"config_hash":hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()}}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary
