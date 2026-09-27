"""G2B: oracle full-pose terminal constraint feasibility for transfer only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .generator_g0 import GRID, G0Record, _proper_so3, _sha256, load_phase15_records, promp_generate
from .generator_g2a import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
    constrain_position_endpoint, constrain_orientation_endpoint,
    quintic_smoothstep_first, quintic_smoothstep_second)


@dataclass(frozen=True)
class GeneratorG2BConfig:
    phase1_output_path: str = "outputs/phase1"
    g0_output_path: str = "outputs/generator_g0"
    g2a_output_path: str = "outputs/generator_g2a"
    output_path: str = "outputs/generator_g2b"
    target_position_tolerance_m: float = 1e-5
    target_orientation_tolerance_rad: float = 1e-8
    endpoint_position_tolerance_m: float = 1e-12
    endpoint_orientation_tolerance_rad: float = 1e-10
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES


def oracle_terminal_pose(record: G0Record, phase1_dir: Path, phase1_manifest: dict,
                         offset_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read final Hand and fork pose from the same frozen Phase 1.5 frame."""
    entry = next(x for x in phase1_manifest["records"] if x["record_id"] == record.record_id)
    with np.load(Path(phase1_dir) / entry["file"], allow_pickle=False) as z:
        valid = np.asarray(z["tool_pose_valid"], bool)
        pos, ori, hand = (np.asarray(z[x], float) for x in ("tool_position", "tool_orientation", "hand_xyz"))
    # G2B is specifically the recorded terminal frame, not a fallback final
    # valid frame.  A missing terminal tool pose is a semantic failure.
    if not valid[-1]:
        raise ValueError(f"{record.record_id}: transfer terminal frame lacks a valid tracked fork pose")
    idx = len(valid) - 1
    if not (np.isfinite(hand[idx]).all() and np.isfinite(pos[idx]).all() and _proper_so3(ori[idx:idx + 1]).all()):
        raise ValueError(f"{record.record_id}: invalid final hand/fork oracle frame")
    if not np.isfinite(offset_m) or offset_m <= 0:
        raise ValueError("invalid frozen Phase 1.5 legacy_hand_offset_m")
    goal_R = ori[idx]
    goal_p = hand[idx] + offset_m * goal_R[:, 1]
    residual_p = float(np.linalg.norm(goal_p - pos[idx]))
    residual_p_gt = float(np.linalg.norm(goal_p - record.gt_position_B[-1]))
    residual_R = float(Rotation.from_matrix(goal_R.T @ record.gt_orientation_B[-1]).magnitude())
    # The record's GT final is its last valid tracked pose resampled to s=1.
    if residual_p > 1e-5 or residual_p_gt > 1e-5 or residual_R > 1e-8:
        raise ValueError(f"{record.record_id}: hand/fork target semantic mismatch: raw={residual_p:.3e} m, GT={residual_p_gt:.3e} m, {residual_R:.3e} rad")
    return goal_p, goal_R, {"source_final_frame_index": idx, "oracle_target_reconstruction_position_residual_m": residual_p,
                              "oracle_target_reconstruction_gt_position_residual_m": residual_p_gt,
                              "oracle_target_reconstruction_orientation_residual_rad": residual_R}


def _g0_arrays(g0: Path, record_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(g0 / "generated" / "contextual_promp" / f"{record_id}.npz", allow_pickle=False) as z:
        return tuple(np.asarray(z[x], float) for x in ("generated_tool_position_B", "generated_tool_orientation_B", "time_s"))


def _rotation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.einsum("nij,njk->nik", a.transpose(0, 2, 1), b)).magnitude()


def _error_metrics(prefix: str, p: np.ndarray, R: np.ndarray, record: G0Record, goal_p: np.ndarray, goal_R: np.ndarray) -> dict:
    pe = np.linalg.norm(p - record.gt_position_B, axis=1); re = _rotation_error(R, record.gt_orientation_B)
    out = {f"{prefix}_terminal_position_error_m": float(np.linalg.norm(p[-1] - goal_p)),
           f"{prefix}_terminal_orientation_error_rad": float(_rotation_error(R[-1:], goal_R[None])[0]),
           f"{prefix}_terminal_orientation_error_deg": float(np.degrees(_rotation_error(R[-1:], goal_R[None])[0])),
           f"{prefix}_rms_position_error_m": float(np.sqrt(np.mean(pe**2))),
           f"{prefix}_mean_position_error_m": float(pe.mean()), f"{prefix}_median_position_error_m": float(np.median(pe)),
           f"{prefix}_p95_position_error_m": float(np.percentile(pe, 95)), f"{prefix}_max_position_error_m": float(pe.max()),
           f"{prefix}_rms_orientation_error_rad": float(np.sqrt(np.mean(re**2))),
           f"{prefix}_rms_orientation_error_deg": float(np.degrees(np.sqrt(np.mean(re**2))),),
           f"{prefix}_mean_orientation_error_deg": float(np.degrees(re.mean())), f"{prefix}_median_orientation_error_deg": float(np.degrees(np.median(re))),
           f"{prefix}_p95_orientation_error_deg": float(np.degrees(np.percentile(re, 95))), f"{prefix}_max_orientation_error_deg": float(np.degrees(re.max()))}
    for label, mask in {"early": GRID < .6, "middle": (GRID >= .6) & (GRID < .8), "late": GRID >= .8, "last20": GRID >= .8, "last10": GRID >= .9}.items():
        out[f"{prefix}_{label}_rms_position_error_m"] = float(np.sqrt(np.mean(pe[mask]**2)))
        out[f"{prefix}_{label}_rms_orientation_error_deg"] = float(np.degrees(np.sqrt(np.mean(re[mask]**2))))
    out[f"{prefix}_path_length_m"] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    out[f"{prefix}_path_length_absolute_error_m"] = abs(out[f"{prefix}_path_length_m"] - np.linalg.norm(np.diff(record.gt_position_B, axis=0), axis=1).sum())
    return out


def _angular_diagnostics(R: np.ndarray, start: float, nominal: np.ndarray | None = None) -> dict:
    ds = GRID[1] - GRID[0]
    omega = Rotation.from_matrix(np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])).as_rotvec() / ds
    alpha = np.diff(omega, axis=0) / ds
    near = int(np.argmin(abs(GRID[:-1] - start)))
    out = {"angular_increment_rms_rad_per_phase": float(np.sqrt(np.mean(np.sum(omega**2, axis=1)))),
            "angular_acceleration_rms_rad_per_phase2": float(np.sqrt(np.mean(np.sum(alpha**2, axis=1)))),
            "angular_increment_near_start_rad_per_phase": float(np.linalg.norm(omega[near])),
            "angular_increment_endpoint_rad_per_phase": float(np.linalg.norm(omega[-1])),
            "angular_increment_last20_max_rad_per_phase": float(np.max(np.linalg.norm(omega[80:], axis=1))),
            "angular_increment_before_start_rad_per_phase": float(np.linalg.norm(omega[max(0, near-1)])),
            "angular_increment_after_start_rad_per_phase": float(np.linalg.norm(omega[min(len(omega)-1, near+1)]))}
    if nominal is not None:
        nw = Rotation.from_matrix(np.einsum("nij,njk->nik", nominal[:-1].transpose(0, 2, 1), nominal[1:])).as_rotvec() / ds
        out["angular_increment_jump_relative_nominal_near_start_rad_per_phase"] = float(np.max(np.linalg.norm(omega[max(0, near-1):min(len(omega), near+2)] - nw[max(0, near-1):min(len(nw), near+2)], axis=1)))
        out["angular_increment_difference_relative_nominal_endpoint_max_rad_per_phase"] = float(np.max(np.linalg.norm(omega[-3:] - nw[-3:], axis=1)))
    return out


def _bootstrap(metrics: pd.DataFrame, cfg: GeneratorG2BConfig) -> pd.DataFrame:
    rows=[]; rng=np.random.default_rng(cfg.bootstrap_seed)
    comparisons=(("full_pose_minus_nominal", "full", "nominal"), ("full_pose_minus_position_only", "full", "position_only"))
    names=("rms_position_error_m", "rms_orientation_error_rad", "last20_rms_position_error_m", "last20_rms_orientation_error_deg", "path_length_absolute_error_m")
    for comparison, left, right in comparisons:
        for name in names:
            pair=metrics.groupby("parent_bite_id")[[f"{left}_{name}", f"{right}_{name}"]].mean(); diff=(pair.iloc[:,0]-pair.iloc[:,1]).to_numpy()
            draws=rng.integers(0,len(diff),(cfg.bootstrap_resamples,len(diff))); samples=diff[draws].mean(1)
            rows.append({"comparison":comparison,"metric":name,"convention":"negative favors full-pose", "mean_difference":float(diff.mean()),"ci95_low":float(np.percentile(samples,2.5)),"ci95_high":float(np.percentile(samples,97.5)),"n_parent_bites":len(diff),"seed":cfg.bootstrap_seed,"resamples":cfg.bootstrap_resamples})
    return pd.DataFrame(rows)


def _examples(metrics: pd.DataFrame) -> dict:
    d=metrics.sort_values("record_id").copy()
    # Rank each incompatible physical unit before combining them deterministically.
    position_rank=d.position_correction_norm_m.rank(method="first", pct=True)
    orientation_rank=d.terminal_orientation_correction_rad.rank(method="first", pct=True)
    combined=np.maximum(position_rank, orientation_rank)
    def item(row): return {x: (float(row[x]) if isinstance(row[x], np.floating) else row[x]) for x in ("record_id","parent_bite_id","take","position_correction_norm_m","terminal_orientation_correction_deg","nominal_last20_rms_orientation_error_deg","full_last20_rms_orientation_error_deg")}
    typical_distance=abs(position_rank-.5)+abs(orientation_rank-.5)
    return {"small_pose_correction":item(d.iloc[int(np.argmin(combined))]), "typical_pose_correction":item(d.iloc[int(np.argmin(typical_distance))]), "large_orientation_correction":item(d.sort_values(["terminal_orientation_correction_rad","record_id"],ascending=[False,True]).iloc[0]), "large_position_correction":item(d.sort_values(["position_correction_norm_m","record_id"],ascending=[False,True]).iloc[0]), "strong_approach_improvement":item(d.assign(x=d.full_last20_rms_orientation_error_deg-d.nominal_last20_rms_orientation_error_deg).sort_values(["x","record_id"]).iloc[0]), "shape_limited":item(d.assign(x=d.full_early_rms_position_error_m+d.full_middle_rms_position_error_m).sort_values(["x","record_id"],ascending=[False,True]).iloc[0])}


def _plots(output: Path, metrics: pd.DataFrame, examples: dict, generated: Path) -> list[str]:
    import matplotlib.pyplot as plt
    plots=output/"plots"; plots.mkdir(exist_ok=True); names=[]
    specs=(("final_approach_error.png",("nominal_last20_rms_position_error_m","full_last20_rms_position_error_m"),"last 20% position RMS (m)"),)
    for name, cols, ylabel in specs:
        fig,ax=plt.subplots(figsize=(5,3.3)); ax.bar([x.replace("_"," ") for x in cols],[metrics[x].mean() for x in cols]); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(plots/name,dpi=150); plt.close(fig); names.append(f"plots/{name}")
    fig,(axp,axr)=plt.subplots(1,2,figsize=(7,3.3))
    variants=("nominal","position_only","full")
    axp.bar(variants,[metrics[f"{v}_terminal_position_error_m"].mean() for v in variants]); axp.set_ylabel("terminal position error (m)")
    axr.bar(variants,[metrics[f"{v}_terminal_orientation_error_deg"].mean() for v in variants]); axr.set_ylabel("terminal orientation error (deg)")
    fig.tight_layout(); fig.savefig(plots/"terminal_pose_error.png",dpi=150); plt.close(fig); names.append("plots/terminal_pose_error.png")
    fig,ax=plt.subplots(figsize=(5,3.3)); phases=("early", "middle", "late"); x=np.arange(3); width=.35
    ax.bar(x-width/2,[metrics[f"nominal_{p}_rms_orientation_error_deg"].mean() for p in phases],width,label="G0 nominal")
    ax.bar(x+width/2,[metrics[f"full_{p}_rms_orientation_error_deg"].mean() for p in phases],width,label="oracle full-pose")
    ax.set_xticks(x, phases); ax.set_ylabel("orientation RMS (deg)"); ax.legend(); fig.tight_layout(); fig.savefig(plots/"orientation_error_by_phase.png",dpi=150); plt.close(fig); names.append("plots/orientation_error_by_phase.png")
    fig,(axp,axr)=plt.subplots(1,2,figsize=(7,3.3))
    position_mean=float(metrics.position_correction_norm_m.mean())
    orientation_mean=float(metrics.terminal_orientation_correction_deg.mean())
    axp.bar([0],[position_mean],width=.35); axp.set_xlim(-.7,.7); axp.set_ylim(0,position_mean*1.25); axp.set_xticks([0],["position"]); axp.set_ylabel("terminal correction (m)")
    axr.bar([0],[orientation_mean],width=.35); axr.set_xlim(-.7,.7); axr.set_ylim(0,orientation_mean*1.25); axr.set_xticks([0],["orientation"]); axr.set_ylabel("terminal correction (deg)")
    fig.tight_layout(); fig.savefig(plots/"correction_magnitude.png",dpi=150); plt.close(fig); names.append("plots/correction_magnitude.png")
    fig,(axi,axa)=plt.subplots(1,2,figsize=(7,3.3)); axi.bar(["nominal","full"],[metrics.nominal_angular_increment_rms_rad_per_phase.mean(),metrics.full_angular_increment_rms_rad_per_phase.mean()]); axi.set_ylabel("rad / normalized phase"); axi.set_title("increment RMS")
    axa.bar(["nominal","full"],[metrics.nominal_angular_acceleration_rms_rad_per_phase2.mean(),metrics.full_angular_acceleration_rms_rad_per_phase2.mean()]); axa.set_ylabel("rad / normalized phase²"); axa.set_title("acceleration-style RMS"); fig.tight_layout(); fig.savefig(plots/"smoothness_diagnostics.png",dpi=150); plt.close(fig); names.append("plots/smoothness_diagnostics.png")
    fig,ax=plt.subplots(figsize=(5,4));
    for label,e in examples.items():
        with np.load(generated/"oracle_full_pose_constrained"/f"{e['record_id']}.npz",allow_pickle=False) as z: n=z["nominal_position_B"]; c=z["full_pose_constrained_position_B"]
        ax.plot(n[:,0],n[:,1],"--",alpha=.45,label=f"{label} nominal"); ax.plot(c[:,0],c[:,1],alpha=.8,label=f"{label} full")
    ax.set_aspect("equal",adjustable="box"); ax.legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(plots/"example_paths.png",dpi=150); plt.close(fig); names.append("plots/example_paths.png")
    return names


def process_generator_g2b(root: Path, output_dir: Path | None = None) -> dict:
    """Run only the held-out-terminal-pose oracle feasibility experiment."""
    root=Path(root); cfg=GeneratorG2BConfig(); phase1=root/cfg.phase1_output_path; g0=root/cfg.g0_output_path; g2a=root/cfg.g2a_output_path; output=Path(output_dir) if output_dir else root/cfg.output_path
    output.mkdir(parents=True,exist_ok=True); generated=output/"generated"; (generated/"oracle_position_constrained").mkdir(parents=True,exist_ok=True); (generated/"oracle_full_pose_constrained").mkdir(parents=True,exist_ok=True)
    records, manifest, _=load_phase15_records(phase1); transfers=[r for r in records if r.phase=="transfer"]
    if not transfers: raise ValueError("no valid transfer records")
    offset_m=float(manifest["config"]["legacy_hand_offset_m"])
    g0_summary=json.loads((g0/"summary.json").read_text()); alphas=g0_summary["generators"]["contextual_promp"]["selected_alphas"]
    starts=pd.read_csv(g2a/"fold_selection.csv").set_index("outer_fold")["selected_adaptation_start_phase"].to_dict()
    rows=[]; artifacts=[]; audits=[]
    for held in sorted({r.take for r in transfers}):
        train=[r for r in transfers if r.take!=held]; alpha=float(alphas[f"{held}:transfer"]); start=float(starts[held])
        for r in sorted((x for x in transfers if x.take==held),key=lambda x:x.record_id):
            p,R,_=promp_generate(r,train,alpha); fp,fR,time=_g0_arrays(g0,r.record_id)
            if not(np.allclose(p,fp,atol=1e-12,rtol=0) and np.allclose(R,fR,atol=1e-12,rtol=0)): raise ValueError(f"G0 parity failed for {r.record_id}")
            goal_p,goal_R,audit=oracle_terminal_pose(r,phase1,manifest,offset_m); audits.append(audit)
            pp,_=constrain_position_endpoint(p,goal_p,start); fullR=constrain_orientation_endpoint(R,start,goal_R)
            if not(_proper_so3(fullR).all() and np.array_equal(R[GRID<start],fullR[GRID<start])): raise ValueError(f"SO(3)/pre-start contract failed for {r.record_id}")
            if np.linalg.norm(pp[-1]-goal_p)>cfg.endpoint_position_tolerance_m or _rotation_error(fullR[-1:],goal_R[None])[0]>cfg.endpoint_orientation_tolerance_rad: raise ValueError(f"endpoint contract failed for {r.record_id}")
            delta=goal_p-p[-1]; rotvec=Rotation.from_matrix(R[-1].T@goal_R).as_rotvec(); deform_p=np.linalg.norm(pp-p,axis=1); deform_R=_rotation_error(R,fullR)
            row={"record_id":r.record_id,"parent_bite_id":r.parent_bite_id,"take":r.take,"phase":"transfer","outer_fold":held,"adaptation_start_phase":start,"position_correction_norm_m":float(np.linalg.norm(delta)),"terminal_orientation_correction_rad":float(np.linalg.norm(rotvec)),"terminal_orientation_correction_deg":float(np.degrees(np.linalg.norm(rotvec))),"full_position_deformation_rms_m":float(np.sqrt(np.mean(deform_p**2))),"full_position_deformation_max_m":float(deform_p.max()),"full_orientation_deformation_rms_rad":float(np.sqrt(np.mean(deform_R**2))),"full_orientation_deformation_max_rad":float(deform_R.max()),"unchanged_fraction_within_pre_start":float(np.mean(((deform_p<1e-14)&(deform_R<1e-12))[GRID<start])),"overall_unchanged_fraction":float(np.mean((deform_p<1e-14)&(deform_R<1e-12)))}
            for phase in (.8,.9,1.):
                angle=float(np.linalg.norm(rotvec) * (0.0 if phase < start else 10*((phase-start)/(1-start))**3-15*((phase-start)/(1-start))**4+6*((phase-start)/(1-start))**5))
                row[f"orientation_correction_at_{int(phase*100)}pct_rad"] = angle
                row[f"orientation_correction_at_{int(phase*100)}pct_deg"] = float(np.degrees(angle))
            for prefix, q, Q in (("nominal",p,R),("position_only",pp,R),("full",pp,fullR)): row.update(_error_metrics(prefix,q,Q,r,goal_p,goal_R))
            row.update({f"nominal_{k}":v for k,v in _angular_diagnostics(R,start).items()}); row.update({f"full_{k}":v for k,v in _angular_diagnostics(fullR,start,R).items()}); row.update(audit); rows.append(row)
            arrays=dict(record_id=np.array(r.record_id),parent_bite_id=np.array(r.parent_bite_id),take=np.array(r.take),phase=np.array("transfer"),outer_fold=np.array(held),normalized_phase=GRID,time_s=time,nominal_position_B=p,nominal_orientation_B=R,oracle_hand_goal_position_B=goal_p-offset_m*goal_R[:,1],oracle_fork_goal_position_B=goal_p,oracle_fork_goal_orientation_B=goal_R,position_only_constrained_position_B=pp,position_only_constrained_orientation_B=R,full_pose_constrained_position_B=pp,full_pose_constrained_orientation_B=fullR,adaptation_start_phase=np.array(start),position_correction_vector_B=delta,terminal_orientation_correction_rotvec=rotvec,terminal_orientation_correction_rad=np.array(np.linalg.norm(rotvec)),terminal_orientation_correction_deg=np.array(np.degrees(np.linalg.norm(rotvec))),oracle_terminal_pose=np.array(True),uses_heldout_terminal_information=np.array(True),deployment_generalization_test=np.array(False),algorithmic_feasibility_only=np.array(True),physical_tool_calibration_validated=np.array(False),physical_mouth_pose_validated=np.array(False),mouth_proxy_used=np.array(False),mouth_offset_used=np.array(False),frozen_g0_nominal_context_contains_mouth_proxy=np.array(True))
            for variant in ("oracle_position_constrained","oracle_full_pose_constrained"):
                path=generated/variant/f"{r.record_id}.npz"; np.savez_compressed(path,**arrays); artifacts.append({"variant":variant,"record_id":r.record_id,"path":str(path),"sha256":_sha256(path)})
    metrics=pd.DataFrame(rows).sort_values("record_id"); metrics.to_csv(output/"record_metrics.csv",index=False); metrics.groupby(["parent_bite_id","take"],as_index=False).mean(numeric_only=True).to_csv(output/"bite_metrics.csv",index=False); per=metrics.groupby("take",as_index=False).mean(numeric_only=True); per.to_csv(output/"per_take_metrics.csv",index=False)
    boot=_bootstrap(metrics,cfg); boot.to_csv(output/"pairwise_bootstrap.csv",index=False); examples=_examples(metrics); (output/"representative_examples.json").write_text(json.dumps(examples,indent=2,sort_keys=True)); plots=_plots(output,metrics,examples,generated)
    audit={"exact_source_columns":["hand_xyz","tool_position","tool_orientation","tool_pose_valid"],"units":"B/world metres","frame_convention":"tracked fork rigid body: R_B_U; Hand_XYZ in B/world","fork_local_offset_vector_U_m":[0.0,-offset_m,0.0],"offset_m_source":"Phase 1.5 manifest config.legacy_hand_offset_m","axis_sign_source":"phase15.py reconstructed_hand = tool_position - offset * tool_orientation[:,1]","relation":"p_H = p_U - offset_m * R_U[:,1]; p_U = p_H + offset_m * R_U[:,1]","source_final_frame":"terminal recorded transfer frame (required valid)","position_reconstruction_residual_m":{"mean":float(metrics.oracle_target_reconstruction_position_residual_m.mean()),"max":float(metrics.oracle_target_reconstruction_position_residual_m.max())},"position_reconstruction_against_g0_resampled_gt_residual_m":{"mean":float(metrics.oracle_target_reconstruction_gt_position_residual_m.mean()),"max":float(metrics.oracle_target_reconstruction_gt_position_residual_m.max())},"orientation_reconstruction_residual_rad":{"mean":float(metrics.oracle_target_reconstruction_orientation_residual_rad.mean()),"max":float(metrics.oracle_target_reconstruction_orientation_residual_rad.max())},"oracle_terminal_pose":True,"uses_heldout_terminal_information":True,"deployment_generalization_test":False,"physical_tool_calibration_validated":False,"physical_mouth_pose_validated":False,"mouth_proxy_used":False,"mouth_offset_used":False,"frozen_g0_nominal_context_contains_mouth_proxy":True,"algorithmic_feasibility_only":True}
    (output/"endpoint_semantics_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True))
    summary={"schema_version":"generator-g2b-v1","config":asdict(cfg),"records":len(metrics),"takes":sorted(metrics["take"].unique()),"endpoint_semantics_audit":audit,"metrics_bite_balanced":metrics.groupby("parent_bite_id").mean(numeric_only=True).mean(numeric_only=True).to_dict(),"paired_parent_bite_bootstrap":boot.to_dict("records"),"representative_examples":examples,"scope":{"oracle_terminal_pose":True,"uses_heldout_terminal_information":True,"deployment_generalization_test":False,"mouth_proxy_used":False,"mouth_offset_used":False,"frozen_g0_nominal_context_contains_mouth_proxy":True,"stronglocal_started":False,"exact_sew_started":False,"robot_ik_started":False,"physical_calibration_started":False,"retiming_started":False}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=float)); outmanifest={"schema_version":"generator-g2b-v1","summary":"summary.json","endpoint_semantics_audit":"endpoint_semantics_audit.json","record_metrics":"record_metrics.csv","bite_metrics":"bite_metrics.csv","per_take_metrics":"per_take_metrics.csv","pairwise_bootstrap":"pairwise_bootstrap.csv","representative_examples":"representative_examples.json","plots":plots,"generated_artifacts":artifacts}; (output/"manifest.json").write_text(json.dumps(outmanifest,indent=2,sort_keys=True)); return summary
