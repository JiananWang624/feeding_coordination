"""Phase 2: fixed-base replay of Phase 1.5 fork trajectories through Exact-SEW."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .adapters.exact_sew import ExactSewTrajectoryAdapter
from .phase1 import _revision

INPUT_NOT_SOLVED = "INPUT_NOT_SOLVED"
SUCCESS_EXACT = "SUCCESS_EXACT"


@dataclass(frozen=True)
class Phase2Config:
    phase1_output_path: str
    output_path: str
    virtual_P_to_UR: dict
    mounting: dict

    @classmethod
    def load(cls, path: Path) -> "Phase2Config":
        value = cls(**json.loads(path.read_text()))
        virtual = value.virtual_P_to_UR
        if not np.array_equal(np.asarray(virtual.get("translation_m"), float), np.zeros(3)) or not np.array_equal(np.asarray(virtual.get("rotation"), float), np.eye(3)):
            raise ValueError("Phase 2 virtual P_to_UR must be exactly identity")
        if value.mounting.get("name") != "Rx(+90deg)" or not np.array_equal(np.asarray(value.mounting.get("robot_world_offset_m"), float), np.array([0., .15, .2])):
            raise ValueError("Phase 2 mounting must use frozen Rx(+90deg) and default offset")
        return value


def _finite_so3(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, float)
    finite = np.isfinite(rotation).all(axis=(1, 2)); out=np.zeros(len(rotation),bool)
    gram = np.swapaxes(rotation[finite], 1, 2) @ rotation[finite]
    out[finite] = (np.isclose(gram, np.eye(3), atol=1e-10, rtol=0.0).all(axis=(1, 2))
                   & np.isclose(np.linalg.det(rotation[finite]), 1., atol=1e-10, rtol=0.0))
    return out


def base_transform(points_B: np.ndarray, R_B_from_base: np.ndarray, p_B_of_base: np.ndarray) -> np.ndarray:
    """Express B-frame row vectors in the fixed native Gen3 base."""
    return (np.asarray(points_B, float) - p_B_of_base) @ R_B_from_base


def base_rotations(rotations_B: np.ndarray, R_B_from_base: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", R_B_from_base.T, np.asarray(rotations_B, float))


def valid_runs(valid: np.ndarray, motive_frame: np.ndarray) -> list[np.ndarray]:
    valid = np.asarray(valid, bool); frames = np.asarray(motive_frame, int); runs=[]; start=None
    for i, ok in enumerate(valid):
        if ok and (start is None): start=i
        if start is not None and (not ok or (i > start and frames[i] - frames[i-1] != 1)):
            end = i - 1
            runs.append(np.arange(start, end + 1)); start = i if ok else None
    if start is not None: runs.append(np.arange(start, len(valid)))
    return runs


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return {x: None for x in ("mean","median","p95","max")}
    return {"mean":float(values.mean()),"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":float(values.max())}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else str(item))


def process_phase2(phase1_dir: Path, output_dir: Path, cfg: Phase2Config,
                   mounting_factory: Callable[[np.ndarray], tuple[Any, Any]] | None = None,
                   adapter_factory: Callable[[], Any] = ExactSewTrajectoryAdapter) -> dict:
    """Replay all Phase-1 records, retaining every source row and failure."""
    phase1_dir=Path(phase1_dir).resolve(); output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True); results_dir=output_dir/"results"; results_dir.mkdir(exist_ok=True)
    manifest1=json.loads((phase1_dir/"manifest.json").read_text())
    if mounting_factory is None:
        from sew_mimic.mounting import load_humanoid_mounted_gen3
        mounting_factory=load_humanoid_mounted_gen3
    from sew_mimic.exact.residuals import robot_exact_sew_residuals
    from sew_mimic.common import ExactSewTarget, joint_limit_margin
    from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
    from sew_mimic.kinematics import gen3_kinematics
    records=manifest1["records"]
    # This local-base robot is intentionally independent of each world mounting.
    evaluation_robot=gen3_kinematics()
    evaluation_geometry=Gen3StereoSewGeometry.from_robot(evaluation_robot)
    evaluation_stereo=StereoSew(project_stereo_sew_reference())
    # One anchor and one frozen mounted world pose per take, from observed shoulder data only.
    mountings={}
    for take in sorted({r["source_take"] for r in records}):
        candidates=[]
        for record in records:
            if record["source_take"] != take: continue
            data=np.load(phase1_dir/record["file"])
            mask=data["shoulder_valid_observation"].astype(bool) & np.isfinite(data["shoulder_xyz"]).all(1)
            if mask.any(): candidates.append((int(data["motive_frame"][np.flatnonzero(mask)[0]]), data["shoulder_xyz"][np.flatnonzero(mask)[0]].copy()))
        if not candidates: raise ValueError(f"{take}: no valid observed shoulder anchor")
        anchor_frame, anchor=min(candidates, key=lambda x:x[0]); robot, data=mounting_factory(anchor)
        base_id=int(robot.frame_body_ids[0]); R=np.asarray(data.xmat[base_id],float).reshape(3,3).copy(); p=np.asarray(data.xpos[base_id],float).copy()
        mountings[take]={"anchor_B":anchor,"anchor_motive_frame":anchor_frame,"robot":robot,"R":R,"p":p}
    all_rows=[]; take_rows={}; phase_rows={}; psi_errors=[]
    for record in records:
        z=np.load(phase1_dir/record["file"]); n=len(z["motive_frame"]); take=record["source_take"]; mount=mountings[take]; R,p,robot=mount["R"],mount["p"],mount["robot"]
        tool_p=np.asarray(z["tool_position"],float); tool_R=np.asarray(z["tool_orientation"],float); psi=np.asarray(z["psi_wrapped"],float)
        target_p=np.full((n,3),np.nan); target_R=np.full((n,3,3),np.nan); target_psi=np.full(n,np.nan)
        finite_p=np.isfinite(tool_p).all(1); finite_r=_finite_so3(tool_R); finite_psi=np.isfinite(psi)
        target_p[finite_p]=base_transform(tool_p[finite_p],R,p); target_R[finite_r]=base_rotations(tool_R[finite_r],R); target_psi[finite_psi]=psi[finite_psi]
        oracle=np.asarray(z["tool_pose_valid"],bool)&np.asarray(z["psi_valid"],bool)&finite_p&finite_r&finite_psi
        status=np.full(n,INPUT_NOT_SOLVED,dtype="<U32"); q=np.full((n,7),np.nan); actual_p=np.full((n,3),np.nan); actual_R=np.full((n,3,3),np.nan); actual_psi=np.full(n,np.nan); pos_err=np.full(n,np.nan); rot_err=np.full(n,np.nan); sew_err=np.full(n,np.nan); margin=np.full(n,np.nan); solve_time=np.full(n,np.nan); branch=[""]*n; raw_branch=[""]*n; message=[""]*n; diag_json=["{}"]*n; run_id=np.full(n,-1,int)
        # Independent native Stereo-SEW diagnostic; it never affects stored psi or solving.
        native=StereoSew(project_stereo_sew_reference())
        sew_points=np.stack((z["shoulder_xyz"],z["elbow_xyz"],z["wrist_xyz"]),axis=1); finite_arm=np.isfinite(sew_points).all((1,2))
        for i in np.flatnonzero(oracle & finite_arm):
            measured_native=native.forward(*base_transform(sew_points[i],R,p))
            psi_errors.append(abs(float(np.arctan2(np.sin(measured_native-psi[i]),np.cos(measured_native-psi[i])))))
        for rid, indices in enumerate(valid_runs(oracle,z["motive_frame"])):
            run_id[indices]=rid; started=perf_counter(); solved=adapter_factory().solve_trajectory(target_p[indices],target_R[indices],target_psi[indices]); elapsed=(perf_counter()-started)*1000
            if len(solved)!=len(indices): raise RuntimeError("trajectory adapter result length mismatch")
            for idx, result in zip(indices,solved):
                status[idx]=getattr(result.status,"value",str(result.status)); diagnostics=result.diagnostics; message[idx]=result.message or ""; raw_branch[idx]=diagnostics.branch_id or ""; branch[idx]=str(diagnostics.metadata.get("search_branch", "")); diag_json[idx]=_json(diagnostics.to_dict()); solve_time[idx]=diagnostics.solve_time_ms if diagnostics.solve_time_ms is not None else elapsed/len(indices)
                if status[idx] == SUCCESS_EXACT and result.q is not None:
                    q[idx]=result.q
                    residual=robot_exact_sew_residuals(result.q,ExactSewTarget(target_p[idx],target_R[idx],target_psi[idx]),evaluation_robot,evaluation_geometry,evaluation_stereo)
                    actual_p[idx]=residual.actual_position; actual_R[idx]=residual.actual_rotation; actual_psi[idx]=residual.actual_psi if residual.actual_psi is not None else np.nan; pos_err[idx]=residual.position_error_m; rot_err[idx]=residual.orientation_error_rad; sew_err[idx]=residual.sew_error_rad if residual.sew_error_rad is not None else np.nan; margin[idx]=joint_limit_margin(result.q,evaluation_robot)
        step=np.full((n,7),np.nan); max_step=np.full(n,np.nan); branch_changed=np.zeros(n,bool)
        for indices in valid_runs(status == SUCCESS_EXACT,z["motive_frame"]):
            for prev,cur in zip(indices[:-1],indices[1:]):
                if run_id[prev] != run_id[cur]: continue
                step[cur]=np.arctan2(np.sin(q[cur]-q[prev]),np.cos(q[cur]-q[prev])); max_step[cur]=np.max(np.abs(step[cur])); branch_changed[cur]=bool(branch[cur] and branch[prev] and branch[cur]!=branch[prev])
        payload={"original_index":np.arange(n),"time_s":z["time"],"motive_frame":z["motive_frame"],"take":np.full(n,take),"bite_id":np.full(n,record["bite_id"]),"phase":z["phase"],"run_id":run_id,"input_valid":oracle,"target_position_base":target_p,"target_rotation_base":target_R,"target_psi":target_psi,"status":status,"q":q,"solver_message":np.asarray(message,dtype=str),"solver_diagnostics_json":np.asarray(diag_json,dtype=str),"branch_id":np.asarray(raw_branch,dtype=str),"search_branch":np.asarray(branch,dtype=str),"actual_position_base":actual_p,"actual_rotation_base":actual_R,"actual_psi":actual_psi,"position_error_m":pos_err,"orientation_error_rad":rot_err,"psi_error_rad":sew_err,"joint_limit_margin_rad":margin,"solve_time_ms":solve_time,"wrapped_joint_step":step,"max_wrapped_joint_step_rad":max_step,"branch_changed":branch_changed}
        np.savez_compressed(results_dir/f"{record['record_id']}.npz",**payload)
        all_rows.append(payload); take_rows.setdefault(take,[]).append(payload); phase_rows.setdefault(record["phase"],[]).append(payload)
    def summarize(payloads):
        combine=lambda k:np.concatenate([x[k] for x in payloads])
        statuses=combine("status"); attempted=statuses != INPUT_NOT_SOLVED; exact=statuses == SUCCESS_EXACT
        return {"total_source_frames":int(len(statuses)),"oracle_input_valid_frames":int(attempted.sum()),"valid_runs":int(sum(np.max(x["run_id"])+1 if len(x["run_id"]) and np.max(x["run_id"])>=0 else 0 for x in payloads)),"attempted_frames":int(attempted.sum()),"status_counts":{str(k):int((statuses==k).sum()) for k in np.unique(statuses)},"SUCCESS_EXACT":int(exact.sum()),"success_percent_among_attempts":float(100*exact.sum()/attempted.sum()) if attempted.any() else None,"position_error_m":_stats(combine("position_error_m")),"orientation_error_rad":_stats(combine("orientation_error_rad")),"psi_error_rad":_stats(combine("psi_error_rad")),"max_wrapped_joint_step_rad":float(np.nanmax(combine("max_wrapped_joint_step_rad"))) if np.isfinite(combine("max_wrapped_joint_step_rad")).any() else None,"branch_changes":int(combine("branch_changed").sum()),"solve_time_ms":_stats(combine("solve_time_ms"))}
    summary={"overall":summarize(all_rows),"per_take":{k:summarize(v) for k,v in take_rows.items()},"per_phase":{k:summarize(v) for k,v in phase_rows.items()},"psi_frame_consistency_abs_circular_rad":_stats(np.asarray(psi_errors))}
    repository_root=Path(__file__).resolve().parents[2]
    lock=json.loads((repository_root/"DEPENDENCY_LOCK.json").read_text())
    dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=repository_root,text=True).strip())
    out_manifest={"schema_version":"phase2-v1","phase1_manifest":str(phase1_dir/"manifest.json"),"phase1_manifest_sha256":hashlib.sha256((phase1_dir/"manifest.json").read_bytes()).hexdigest(),"phase1_schema_version":manifest1["schema_version"],"config":asdict(cfg),"config_hash":hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest(),"code_revision":_revision(repository_root),"code_worktree_dirty":dirty,"dependency_commit":lock["exact_sew"]["commit"],"virtual_tool_convention":"P_to_UR exactly identity; no physical fork-tip calibration","mountings":{take:{"source_take":take,"anchor_B":m["anchor_B"].tolist(),"anchor_motive_frame":m["anchor_motive_frame"],"R_B_from_base":m["R"].tolist(),"p_B_of_base":m["p"].tolist(),"name":cfg.mounting["name"],"robot_world_offset_m":cfg.mounting["robot_world_offset_m"]} for take,m in mountings.items()},"records":[{"record_id":r["record_id"],"file":f"results/{r['record_id']}.npz","source_record":r["file"]} for r in records],"summary":summary}
    (output_dir/"manifest.json").write_text(json.dumps(out_manifest,indent=2,sort_keys=True)); (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); return out_manifest
