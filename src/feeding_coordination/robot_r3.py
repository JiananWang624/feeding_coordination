"""Robot R3: frozen initial-redundancy robustness analysis."""
from __future__ import annotations
import hashlib, json, pickle, shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from .dependency.exact_sew import load_exact_sew_dependency
from .phase2 import INPUT_NOT_SOLVED, SUCCESS_EXACT
from .robot_r0 import R_INPUT_ALIGN
from .robot_r1 import CONTINUITY_THRESHOLD_RAD, LOCAL_OFFSETS_RAD, _default_solver, clean_motion, continuity_labels, robot_smooth_run, wrap
from .robot_r2 import arm_plane_angle, jacobian_metrics

OFFSETS=("psi0_minus_025","psi0_nominal","psi0_plus_025")
OFFSET_VALUES=np.array([-.25,0.,.25])
STRATEGIES=("B0","B2","B3","StrongLocal","H_star","HumanGT","RobotSmooth")
NON_SMOOTH=STRATEGIES[:-1]
PRIMARY=(('StrongLocal','B2'),('StrongLocal','B3'),('StrongLocal','RobotSmooth'),('HumanGT','RobotSmooth'))
BOOT_METRICS=("psi_mae","elbow_mean","travel","rms_velocity","min_margin","violations")
TERMINATED="NOT_EVALUATED_AFTER_INITIAL_FAILURE"

@dataclass(frozen=True)
class RobotR3Config:
    robot_r0_output_path:str="outputs/robot_r0"
    robot_r1_output_path:str="outputs/robot_r1"
    output_path:str="outputs/robot_r3"
    offsets_rad:tuple[float,...]=(-.25,0.,.25)
    continuity_threshold_rad:float=.5
    bootstrap_seed:int=20260915
    bootstrap_resamples:int=2000
    @classmethod
    def load(cls,path:Path):
        raw=json.loads(Path(path).read_text()); raw["offsets_rad"]=tuple(raw["offsets_rad"]); value=cls(**raw)
        if value.offsets_rad!=(-.25,0.,.25): raise ValueError("Robot R3 offsets are frozen to [-0.25, 0, +0.25] rad")
        if (value.continuity_threshold_rad,value.bootstrap_seed,value.bootstrap_resamples)!=(.5,20260915,2000): raise ValueError("Robot R3 continuity/bootstrap settings differ from frozen values")
        return value

def shift_trajectory(human_psi0:float, offset:float, anchored_delta:np.ndarray)->np.ndarray:
    """Shift only the run anchor; preserve the frozen delta trajectory."""
    delta=np.asarray(anchored_delta,float)
    if delta.ndim!=1 or not len(delta) or not np.isfinite(delta).all() or not np.isclose(delta[0],0.,atol=1e-12,rtol=0.): raise ValueError("anchored delta-psi run is invalid")
    return float(wrap(human_psi0+offset))+delta

def solve_stateful_run(positions:np.ndarray,rotations:np.ndarray,psi:np.ndarray,dependency_loader:Callable=load_exact_sew_dependency)->list[Any]:
    """Fresh native stateful solver; stop immediately when its initial target fails."""
    p=np.asarray(positions,float); r=np.asarray(rotations,float); angles=np.asarray(psi,float)
    if p.shape!=(len(angles),3) or r.shape!=(len(angles),3,3) or not len(angles): raise ValueError("invalid shifted trajectory")
    dep=dependency_loader(); robot=dep.gen3_kinematics(); geometry=dep.geometry_type.from_robot(robot); stereo=dep.stereo_type(dep.project_reference()); solver=dep.solver_type(robot,geometry,stereo)
    first=solver.solve(dep.target_type(p[0],r[0],angles[0])); results=[first]
    status=getattr(first.status,"value",str(first.status))
    if status!=SUCCESS_EXACT: return results
    results.extend(solver.solve(dep.target_type(p[i],r[i],angles[i])) for i in range(1,len(angles)))
    return results

def verify_tool_identity(reference_p:np.ndarray,reference_r:np.ndarray,*candidates:tuple[np.ndarray,np.ndarray])->bool:
    return all(np.array_equal(reference_p,p) and np.array_equal(reference_r,r) for p,r in candidates)

def verify_shared_initial(payloads:dict[str,dict[str,np.ndarray]],first:int)->dict[str,Any]:
    psi=np.array([payloads[s]["strategy_psi"][first] for s in STRATEGIES]); statuses=[str(payloads[s]["solver_status"][first]) for s in STRATEGIES]
    if not np.array_equal(psi,np.full(len(psi),psi[0])): raise RuntimeError("first target psi differs across strategies")
    if len(set(statuses))!=1: raise RuntimeError("initial solver status differs across strategies")
    max_q=0.
    if statuses[0]==SUCCESS_EXACT:
        q0=payloads[STRATEGIES[0]]["q"][first]
        for s in STRATEGIES[1:]:
            max_q=max(max_q,float(np.max(np.abs(payloads[s]["q"][first]-q0))))
            if not np.allclose(payloads[s]["q"][first],q0,atol=1e-10,rtol=0.): raise RuntimeError("successful first q differs across strategies")
    return {"first_target_position_equal":True,"first_target_rotation_equal":True,"first_target_psi_equal":True,"first_status_equal":True,"max_first_q_abs_difference_rad":max_q}

def initial_classification(status:str)->str:
    return {SUCCESS_EXACT:"INITIAL_STATE_SUCCESS","JOINT_LIMIT":"INITIAL_JOINT_LIMIT","NO_VALID_BRANCH":"INITIAL_NO_VALID_BRANCH"}.get(str(status),f"INITIAL_{status}")

def _empty(n:int)->dict[str,np.ndarray]:
    return {"solver_status":np.full(n,INPUT_NOT_SOLVED,dtype=object),"q":np.full((n,7),np.nan),"strategy_psi":np.full(n,np.nan),"branch_id":np.full(n,"",dtype=object),"search_branch":np.full(n,"",dtype=object),"joint_limit_margin_rad":np.full(n,np.nan),"jp_sigma":np.full(n,np.nan),"jr_sigma":np.full(n,np.nan),"elbow":np.full((n,3),np.nan),"plane_normal":np.full((n,3),np.nan)}

def _result_fields(result:Any)->tuple[str,np.ndarray|None,str,str]:
    status=getattr(result.status,"value",str(result.status)); diag=result.diagnostics; raw=diag.to_dict(); meta=raw.get("metadata",{})
    q=None if result.q is None else np.asarray(result.q,float)
    return status,q,str(diag.branch_id or ""),str(meta.get("search_branch",""))

def _rms(x:np.ndarray)->float:
    z=np.asarray(x,float); z=np.abs(z[np.isfinite(z)]); return float(np.sqrt(np.mean(z*z))) if len(z) else np.nan

def _branch_changes(payload:dict[str,np.ndarray],raw:dict[str,np.ndarray])->int:
    ok=payload["solver_status"]==SUCCESS_EXACT; branch=np.asarray(payload["search_branch"]); total=0
    for i in range(1,len(ok)):
        edge=ok[i-1] and ok[i] and raw["run_id"][i]==raw["run_id"][i-1] and raw["motive_frame"][i]==raw["motive_frame"][i-1]+1
        if edge: total+=int(bool(branch[i-1] and branch[i] and branch[i-1]!=branch[i]))
    return total

def _motion(payload:dict[str,np.ndarray],raw:dict[str,np.ndarray],success:np.ndarray|None=None,violation:np.ndarray|None=None)->dict[str,Any]:
    own=payload["solver_status"]==SUCCESS_EXACT; ok=own if success is None else np.asarray(success,bool); vio=payload["continuity_violation"] if violation is None else np.asarray(violation,bool)
    clean=clean_motion(payload["q"],raw["time_s"],raw["motive_frame"],raw["run_id"],ok,vio)
    margins=payload["joint_limit_margin_rad"][ok&~vio]; jp=payload["jp_sigma"][ok&~vio]; jr=payload["jr_sigma"][ok&~vio]
    return {"travel":float(np.nansum(np.abs(clean["clean_wrapped_delta_q_rad"]))),"rms_velocity":_rms(clean["clean_joint_velocity_rad_s"]),"rms_acceleration":_rms(clean["clean_joint_acceleration_rad_s2"]),"rms_jerk":_rms(clean["clean_joint_jerk_rad_s3"]),"min_margin":float(np.nanmin(margins)) if np.isfinite(margins).any() else np.nan,"min_jp_sigma":float(np.nanmin(jp)) if np.isfinite(jp).any() else np.nan,"min_jr_sigma":float(np.nanmin(jr)) if np.isfinite(jr).any() else np.nan,"clean_frames":int((ok&~vio).sum()),"clean_intervals":int(sum(max(0,len(x)-1) for x in clean["continuous_segments"])),"travel_edges":int(np.isfinite(clean["clean_wrapped_delta_q_rad"]).all(1).sum()),"velocity_frames":int(np.isfinite(clean["clean_joint_velocity_rad_s"]).all(1).sum()),"acceleration_frames":int(np.isfinite(clean["clean_joint_acceleration_rad_s2"]).all(1).sum()),"jerk_frames":int(np.isfinite(clean["clean_joint_jerk_rad_s3"]).all(1).sum())}

def _human(payload:dict[str,np.ndarray],human:dict[str,np.ndarray],also:np.ndarray|None=None)->dict[str,Any]:
    ok=(payload["solver_status"]==SUCCESS_EXACT)&(human["solver_status"]==SUCCESS_EXACT)
    if also is not None: ok&=np.asarray(also,bool)
    psi=np.abs(wrap(payload["strategy_psi"][ok]-human["strategy_psi"][ok])); dq=wrap(payload["q"][ok]-human["q"][ok]); qn=np.linalg.norm(dq,axis=1)
    elbow=np.linalg.norm(payload["elbow"][ok]-human["elbow"][ok],axis=1); plane=np.array([arm_plane_angle(a,b) for a,b in zip(payload["plane_normal"][ok],human["plane_normal"][ok])])
    def mean(x): return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    return {"common_success_frames":int(ok.sum()),"psi_mae":mean(psi),"psi_rmse":_rms(psi),"q_rms":_rms(qn),"elbow_mean":mean(elbow),"plane_mean":mean(plane)}

def _bootstrap(rows:pd.DataFrame,metric:str,seed:int=20260915,n:int=2000)->tuple[float,float,float,int,str]:
    x=rows[["parent_bite_id",metric]].copy(); x[metric]=pd.to_numeric(x[metric],errors="coerce"); v=x.groupby("parent_bite_id",sort=True)[metric].mean().to_numpy(float); v=v[np.isfinite(v)]
    if not len(v): return np.nan,np.nan,np.nan,0,"NO_FINITE_PARENT_BITE_VALUES"
    draws=np.random.default_rng(seed).choice(v,(n,len(v)),replace=True).mean(1); return float(v.mean()),float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5)),len(v),"OK"

def _json(x:Any)->Any:
    if isinstance(x,dict): return {str(k):_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x

def _scope(df:pd.DataFrame,grouping:str,group:str)->pd.DataFrame:
    if grouping=="overall": return df
    return df[df[grouping].astype(str).eq(str(group))]

def _aggregate_strategy(bites:pd.DataFrame,grouping:str,group:str)->list[dict[str,Any]]:
    d=_scope(bites,grouping,group); rows=[]; mean_metrics=("travel","rms_velocity","rms_acceleration","rms_jerk","min_margin","min_jp_sigma","min_jr_sigma")
    for (condition,offset,strategy),g in d.groupby(["condition","offset_rad","strategy"],sort=False):
        parent=g.groupby("parent_bite_id",sort=True)[list(mean_metrics)].mean(); source=int(g.source_frames.sum()); success=int(g.success_frames.sum()); initial=int(g.initial_success_runs.sum()); runs=int(g.source_runs.sum()); complete=int(g.complete_success_runs.sum()); violations=int(g.violations.sum())
        rows.append({"grouping":grouping,"group":group,"condition":condition,"offset_rad":offset,"strategy":strategy,"support_bites":int(g.parent_bite_id.nunique()),"source_runs":runs,"initial_success_runs":initial,"initial_state_success_percent":100*initial/max(1,runs),"source_frames":source,"success_frames":success,"trajectory_frame_success_percent":100*success/max(1,source),"complete_success_runs":complete,"complete_run_success_percent":100*complete/max(1,runs),"JOINT_LIMIT":int(g.JOINT_LIMIT.sum()),"NO_VALID_BRANCH":int(g.NO_VALID_BRANCH.sum()),"other_failure_frames":int(g.other_failure_frames.sum()),"violations":violations,"continuous_frame_percent":100*(success-violations)/max(1,success),"branch_changes":int(g.branch_changes.sum()),**{m:float(parent[m].mean()) for m in mean_metrics}})
    return rows

def _aggregate_human(rows:pd.DataFrame,grouping:str,group:str)->list[dict[str,Any]]:
    d=_scope(rows,grouping,group); out=[]; metrics=("psi_mae","psi_rmse","q_rms","elbow_mean","plane_mean")
    for (condition,offset,strategy),g in d.groupby(["condition","offset_rad","strategy"],sort=False):
        parent=g.groupby("parent_bite_id",sort=True)[list(metrics)].mean()
        out.append({"grouping":grouping,"group":group,"condition":condition,"offset_rad":offset,"strategy":strategy,"reference":"shifted_HumanGT","support_bites":int(g.parent_bite_id.nunique()),"support_common_success_frames":int(g.common_success_frames.sum()),**{m:float(parent[m].mean()) for m in metrics}})
    return out

def process_robot_r3(root:Path,output:Path,cfg:RobotR3Config)->dict[str,Any]:
    root=Path(root); output=Path(output); r0=root/cfg.robot_r0_output_path; r1=root/cfg.robot_r1_output_path; m0_path=r0/"manifest.json"; m1_path=r1/"manifest.json"; m0=json.loads(m0_path.read_text()); m1=json.loads(m1_path.read_text())
    if m0.get("schema_version")!="robot-r0-v1" or m1.get("schema_version")!="robot-r1-v1": raise ValueError("Robot R3 requires frozen Robot R0/R1 v1")
    h0=hashlib.sha256(m0_path.read_bytes()).hexdigest(); h1=hashlib.sha256(m1_path.read_bytes()).hexdigest(); before=(h0,h1); tmp=output.with_name(output.name+".tmp"); cache_dir=output.with_name(output.name+".cache")
    cache_meta={"schema_version":"robot-r3-cache-v1","robot_r0_manifest_sha256":h0,"robot_r1_manifest_sha256":h1,"config":asdict(cfg)}; cache_meta_path=cache_dir/"metadata.json"
    if cache_dir.exists() and (not cache_meta_path.exists() or json.loads(cache_meta_path.read_text())!=json.loads(json.dumps(cache_meta))): shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True); cache_meta_path.write_text(json.dumps(cache_meta,sort_keys=True))
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True); (tmp/"plots").mkdir()
    from sew_mimic.common import joint_limit_margin
    from sew_mimic.mounting import load_humanoid_mounted_gen3
    from sew_mimic.sew import Gen3StereoSewGeometry
    mounted={}; bite_rows=[]; human_rows=[]; pair_rows=[]; initial_rows=[]; validations=[]; smooth_solver,smooth_margin=_default_solver()
    for entry in m0["records"]:
        cache_file=cache_dir/f"{entry['record_id']}.pkl"
        if cache_file.exists():
            cached=pickle.loads(cache_file.read_bytes()); bite_rows.extend(cached["bite_rows"]); human_rows.extend(cached["human_rows"]); pair_rows.extend(cached["pair_rows"]); initial_rows.extend(cached["initial_rows"]); validations.extend(cached["validations"]); continue
        starts=(len(bite_rows),len(human_rows),len(pair_rows),len(initial_rows),len(validations))
        with np.load(r0/entry["file"],allow_pickle=False) as z: raw={k:z[k].copy() for k in z.files}
        n=len(raw["time_s"]); take=str(raw["take"][0]); record_id=str(raw["record_id"][0]); parent=str(raw["parent_bite_id"][0]); phase=str(raw["phase"][0]); valid=raw["common_source_valid"].astype(bool); run_ids=sorted(set(raw["run_id"][raw["run_id"]>=0].tolist()))
        if take not in mounted:
            anchor=np.asarray(m0["mountings"][take]["anchor_B"],float); robot,data=load_humanoid_mounted_gen3(anchor); mounted[take]=(robot,data,Gen3StereoSewGeometry.from_robot(robot))
        robot,data,geometry=mounted[take]
        for condition,offset in zip(OFFSETS,OFFSET_VALUES):
            payloads={s:_empty(n) for s in STRATEGIES}; condition_initial=[]
            for rid in run_ids:
                idx=np.flatnonzero(raw["run_id"]==rid); first=int(idx[0]); psi0=float(raw["human_psi_unwrapped"][first]); p=raw["desired_pinch_position_base"][idx]; rot=raw["desired_pinch_rotation_base"][idx]
                targets=[]
                for strategy in NON_SMOOTH:
                    delta=raw[f"{strategy}_anchored_delta_psi"][idx]; shifted=shift_trajectory(psi0,float(offset),delta); payload=payloads[strategy]; payload["strategy_psi"][idx]=shifted
                    if not np.allclose(shifted-shifted[0],delta,atol=1e-12,rtol=0.): raise RuntimeError("frozen delta-psi changed under shifted initial condition")
                    solved=solve_stateful_run(p,rot,shifted)
                    for local,result in enumerate(solved):
                        status,q,bid,sb=_result_fields(result); j=int(idx[local]); payload["solver_status"][j]=status; payload["branch_id"][j]=bid; payload["search_branch"][j]=sb
                        if status==SUCCESS_EXACT and q is not None: payload["q"][j]=q
                    if len(solved)==1 and str(payload["solver_status"][first])!=SUCCESS_EXACT: payload["solver_status"][idx[1:]]=TERMINATED
                    targets.append((p,rot))
                if not verify_tool_identity(p,rot,*targets): raise RuntimeError("tool trajectory differs across strategies")
                first_statuses=[str(payloads[s]["solver_status"][first]) for s in NON_SMOOTH]
                if len(set(first_statuses))!=1: raise RuntimeError("initial solver status differs across strategies")
                if first_statuses[0]==SUCCESS_EXACT:
                    q0=payloads[NON_SMOOTH[0]]["q"][first]
                    for s in NON_SMOOTH[1:]:
                        if not np.allclose(payloads[s]["q"][first],q0,atol=1e-10,rtol=0.): raise RuntimeError("initial q differs across strategies")
                else: q0=None
                smooth=robot_smooth_run(p,rot,float(payloads["B0"]["strategy_psi"][first]),q0,first_statuses[0],smooth_solver,smooth_margin); ps=payloads["RobotSmooth"]
                for key in ("solver_status","q","strategy_psi","branch_id","search_branch"): ps[key][idx]=smooth[key]
                validation=verify_shared_initial(payloads,first); validation|={"record_id":record_id,"run_id":int(rid),"condition":condition,"initial_status":first_statuses[0]}; validations.append(validation)
                cls=initial_classification(first_statuses[0]); condition_initial.append({"record_id":record_id,"parent_bite_id":parent,"take":take,"phase":phase,"run_id":int(rid),"condition":condition,"offset_rad":float(offset),"initial_status":first_statuses[0],"classification":cls})
            initial_rows.extend(condition_initial)
            # HumanGT geometry must exist before any strategy-to-HumanGT row is computed.
            for strategy in ("HumanGT","B0","B2","B3","StrongLocal","H_star","RobotSmooth"):
                payload=payloads[strategy]
                ok=payload["solver_status"]==SUCCESS_EXACT; labels=continuity_labels(payload["q"],ok,raw["motive_frame"],raw["run_id"]); payload.update(labels)
                for i in np.flatnonzero(ok):
                    q=payload["q"][i]; payload["joint_limit_margin_rad"][i]=joint_limit_margin(q,robot); jp,_,jr,_=jacobian_metrics(robot,data,q); payload["jp_sigma"][i]=jp; payload["jr_sigma"][i]=jr; points=geometry.sew_points(q); payload["elbow"][i]=points.elbow; payload["plane_normal"][i]=np.cross(points.wrist-points.shoulder,points.elbow-points.shoulder)
                mot=_motion(payload,raw); statuses=payload["solver_status"][valid]; source_runs=len(run_ids); initial_success=sum(str(payload["solver_status"][np.flatnonzero(raw["run_id"]==rid)[0]])==SUCCESS_EXACT for rid in run_ids); complete=sum(bool((payload["solver_status"][raw["run_id"]==rid]==SUCCESS_EXACT).all()) for rid in run_ids)
                known=np.isin(statuses,[SUCCESS_EXACT,"JOINT_LIMIT","NO_VALID_BRANCH",TERMINATED,INPUT_NOT_SOLVED]); base={"row_scope":"single_strategy","record_id":record_id,"parent_bite_id":parent,"take":take,"phase":phase,"condition":condition,"offset_rad":float(offset),"strategy":strategy,"source_runs":source_runs,"initial_success_runs":initial_success,"source_frames":int(valid.sum()),"success_frames":int((ok&valid).sum()),"complete_success_runs":complete,"JOINT_LIMIT":int((statuses=="JOINT_LIMIT").sum()),"NO_VALID_BRANCH":int((statuses=="NO_VALID_BRANCH").sum()),"other_failure_frames":int((~known).sum()),"violations":int((labels["continuity_violation"]&valid).sum()),"branch_changes":_branch_changes(payload,raw),**mot}; bite_rows.append(base)
                if strategy!="HumanGT": human_rows.append({"record_id":record_id,"parent_bite_id":parent,"take":take,"phase":phase,"condition":condition,"offset_rad":float(offset),"strategy":strategy,**_human(payload,payloads["HumanGT"])})
            for a,b in PRIMARY:
                pa,pb,ph=payloads[a],payloads[b],payloads["HumanGT"]; common=(pa["solver_status"]==SUCCESS_EXACT)&(pb["solver_status"]==SUCCESS_EXACT)&valid; union=pa["continuity_violation"]|pb["continuity_violation"]
                ma=_motion(pa,raw,common,union); mb=_motion(pb,raw,common,union); triple=common&(ph["solver_status"]==SUCCESS_EXACT); ha=_human(pa,ph,(pb["solver_status"]==SUCCESS_EXACT)&valid); hb=_human(pb,ph,(pa["solver_status"]==SUCCESS_EXACT)&valid)
                row={"row_scope":"primary_pair","record_id":record_id,"parent_bite_id":parent,"take":take,"phase":phase,"condition":condition,"offset_rad":float(offset),"comparison":f"{a}-{b}","strategy_a":a,"strategy_b":b,"pair_common_success_frames":int(common.sum()),"pair_clean_frames":int((common&~union).sum()),"pair_clean_intervals":ma["clean_intervals"],"triple_common_success_frames":int(triple.sum()),"a_violations":int((pa["continuity_violation"]&valid).sum()),"b_violations":int((pb["continuity_violation"]&valid).sum())}
                for metric in ("travel","rms_velocity","rms_acceleration","rms_jerk","min_margin","min_jp_sigma","min_jr_sigma"): row[f"a_{metric}"]=ma[metric]; row[f"b_{metric}"]=mb[metric]; row[f"difference_{metric}_a_minus_b"]=ma[metric]-mb[metric]
                for metric in ("psi_mae","elbow_mean"): row[f"a_{metric}_to_humangt"]=ha[metric]; row[f"b_{metric}_to_humangt"]=hb[metric]; row[f"difference_{metric}_a_minus_b"]=ha[metric]-hb[metric]
                row["difference_violations_a_minus_b"]=row["a_violations"]-row["b_violations"]; pair_rows.append(row)
        cached={"bite_rows":bite_rows[starts[0]:],"human_rows":human_rows[starts[1]:],"pair_rows":pair_rows[starts[2]:],"initial_rows":initial_rows[starts[3]:],"validations":validations[starts[4]:]}; cache_tmp=cache_file.with_suffix(".tmp"); cache_tmp.write_bytes(pickle.dumps(cached,protocol=pickle.HIGHEST_PROTOCOL)); cache_tmp.replace(cache_file)
    singles=pd.DataFrame(bite_rows); humans=pd.DataFrame(human_rows); pairs=pd.DataFrame(pair_rows); initials=pd.DataFrame(initial_rows)
    strategy_table=pd.DataFrame([row for grouping,group in (("overall","overall"),("phase","transfer"),("phase","withdrawal")) for row in _aggregate_strategy(singles,grouping,group)])
    human=pd.DataFrame([row for grouping,group in (("overall","overall"),("phase","transfer"),("phase","withdrawal"),*(("take",x) for x in sorted(humans["take"].unique()))) for row in _aggregate_human(humans,grouping,group)])
    init_out=[]
    for grouping,group in (("overall","overall"),("phase","transfer"),("phase","withdrawal"),*(("take",x) for x in sorted(initials["take"].unique()))):
        d=_scope(initials,grouping,group)
        for (condition,offset),g in d.groupby(["condition","offset_rad"],sort=False):
            counts=g.classification.value_counts(); init_out.append({"grouping":grouping,"group":group,"condition":condition,"offset_rad":offset,"source_runs":len(g),"initial_success_runs":int((g.classification=="INITIAL_STATE_SUCCESS").sum()),"initial_state_success_percent":100*float((g.classification=="INITIAL_STATE_SUCCESS").mean()) if len(g) else np.nan,"INITIAL_JOINT_LIMIT":int(counts.get("INITIAL_JOINT_LIMIT",0)),"INITIAL_NO_VALID_BRANCH":int(counts.get("INITIAL_NO_VALID_BRANCH",0)),"other_initial_status":int(len(g)-counts.get("INITIAL_STATE_SUCCESS",0)-counts.get("INITIAL_JOINT_LIMIT",0)-counts.get("INITIAL_NO_VALID_BRANCH",0)),"initial_status_counts_json":json.dumps({str(k):int(v) for k,v in g.initial_status.value_counts().items()},sort_keys=True)})
    initial_table=pd.DataFrame(init_out)
    take=pd.DataFrame([row|{"take":group} for group in sorted(singles["take"].unique()) for row in _aggregate_strategy(singles,"take",group)]).drop(columns=["grouping","group"])
    robustness=[]; robust_metrics=("trajectory_frame_success_percent","violations","travel","rms_velocity","min_margin","min_jp_sigma","min_jr_sigma")
    for (grouping,group,strategy_name),g in strategy_table.groupby(["grouping","group","strategy"]):
        nominal=g[g.condition.eq("psi0_nominal")].iloc[0]
        for _,row in g[~g.condition.eq("psi0_nominal")].iterrows():
            for metric in robust_metrics: robustness.append({"grouping":grouping,"group":group,"strategy":strategy_name,"condition":row.condition,"offset_rad":row.offset_rad,"metric":metric,"value":row[metric],"nominal_value":nominal[metric],"delta_from_nominal":row[metric]-nominal[metric]})
    boot=[]
    for (comp,condition),g in pairs.groupby(["comparison","condition"]):
        for scope,sub in (("overall",g),("transfer",g[g.phase.eq("transfer")]),("withdrawal",g[g.phase.eq("withdrawal")])):
            for metric in BOOT_METRICS:
                col=f"difference_{metric}_a_minus_b"; d,lo,hi,n,status=_bootstrap(sub,col,cfg.bootstrap_seed,cfg.bootstrap_resamples); boot.append({"comparison":comp,"condition":condition,"phase":scope,"metric":metric,"difference_a_minus_b":d,"ci_low":lo,"ci_high":hi,"support_bites":n,"status":status})
    per_bite=pd.concat((singles,pairs),ignore_index=True,sort=False)
    initial_table.to_csv(tmp/"initial_state_feasibility.csv",index=False); strategy_table.to_csv(tmp/"strategy_metrics.csv",index=False); human.to_csv(tmp/"human_consistency_metrics.csv",index=False); pd.DataFrame(robustness).to_csv(tmp/"robustness_deltas.csv",index=False); pd.DataFrame(boot).to_csv(tmp/"pairwise_bootstrap.csv",index=False); per_bite.to_csv(tmp/"per_bite_metrics.csv",index=False); take.to_csv(tmp/"per_take_metrics.csv",index=False)
    rank_strategies=("B2","B3","StrongLocal","H_star","HumanGT","RobotSmooth"); ranking={}
    for condition in OFFSETS:
        s=strategy_table[(strategy_table.grouping.eq("overall"))&(strategy_table.condition.eq(condition))].set_index("strategy"); h=human[(human.grouping.eq("overall"))&(human.condition.eq(condition))].set_index("strategy")
        ranking[condition]={}
        for metric,ascending in (("psi_mae",True),("elbow_mean",True),("travel",True),("rms_velocity",True),("violations",True),("min_margin",False)):
            values={x:(0. if x=="HumanGT" and metric in ("psi_mae","elbow_mean") else float((h if metric in ("psi_mae","elbow_mean") else s).loc[x,metric])) for x in rank_strategies}; ranking[condition][metric]=sorted(values,key=lambda x:(values[x] if ascending else -values[x],x))
    def choose(frame:pd.DataFrame,score:pd.Series,rule:str,minimum:bool=False):
        x=frame.copy(); x["_score"]=score; x=x[np.isfinite(x._score)].sort_values(["_score","record_id"],ascending=[minimum,True]); return {"record_id":str(x.iloc[0].record_id),"score":float(x.iloc[0]._score),"criterion":rule} if len(x) else {"status":"NO_ELIGIBLE_RECORD","criterion":rule}
    complete=singles.groupby("record_id").filter(lambda x:x.condition.nunique()==3 and bool((x.complete_success_runs==x.source_runs).all())); eligible=set(complete.record_id.unique()); stable_source=singles[singles.record_id.isin(eligible)]; stable_range=stable_source.groupby(["record_id","strategy"]).rms_velocity.agg(lambda x:x.max()-x.min()); stable_score=stable_range.groupby("record_id").max() if len(stable_range) else pd.Series(dtype=float)
    sens=singles.groupby(["record_id","condition"],as_index=False)[["success_frames","branch_changes"]].sum(); sens_score=sens.groupby("record_id").agg({"success_frames":lambda x:x.max()-x.min(),"branch_changes":lambda x:x.max()-x.min()}).sum(axis=1)
    def pair_range(comp,metric):
        x=pairs[pairs.comparison.eq(comp)].pivot_table(index="record_id",columns="condition",values=metric); return (x.max(axis=1)-x.min(axis=1)).dropna()
    stable_frame=pd.DataFrame({"record_id":stable_score.index,"score":stable_score.values}); sensitive_frame=pd.DataFrame({"record_id":sens_score.index,"score":sens_score.values}); hr=pair_range("HumanGT-RobotSmooth","difference_travel_a_minus_b").abs()+pair_range("HumanGT-RobotSmooth","difference_psi_mae_a_minus_b").abs(); sb=pair_range("StrongLocal-B2","difference_rms_velocity_a_minus_b").abs()+pair_range("StrongLocal-B2","difference_psi_mae_a_minus_b").abs()
    examples={"selection_rules":"deterministic metric criteria; ties record_id","stable":choose(stable_frame,stable_frame.score,"all conditions complete; minimize cross-condition motion range",True),"sensitive":choose(sensitive_frame,sensitive_frame.score,"maximize feasibility plus branch-change range"),"humangt_vs_robotsmooth":choose(pd.DataFrame({"record_id":hr.index,"score":hr.values}),pd.Series(hr.values),"maximize change in travel plus psi tradeoff"),"stronglocal_vs_b2":choose(pd.DataFrame({"record_id":sb.index,"score":sb.values}),pd.Series(sb.values),"minimize change in StrongLocal-B2 velocity plus psi difference",True)}
    (tmp/"representative_examples.json").write_text(json.dumps(_json(examples),indent=2))
    import matplotlib.pyplot as plt
    overall_init=initial_table[initial_table.grouping.eq("overall")]; overall_s=strategy_table[strategy_table.grouping.eq("overall")]; overall_h=human[human.grouping.eq("overall")]
    for filename,df,index,columns,value,ylabel in (("initial_state_feasibility",overall_init.assign(series="initial"),"condition","series","initial_state_success_percent","initial success (%)"),("robustness_by_psi0",overall_s,"strategy","condition","trajectory_frame_success_percent","frame success (%)"),("humangt_consistency_by_psi0",overall_h,"strategy","condition","psi_mae","psi MAE (rad)"),("motion_by_psi0",overall_s,"strategy","condition","rms_velocity","RMS velocity (rad/s)")):
        pivot=df.pivot(index=index,columns=columns,values=value); fig,ax=plt.subplots(figsize=(9,4)); pivot.plot.bar(ax=ax); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(tmp/"plots"/f"{filename}.png",dpi=150); plt.close(fig)
    validation_summary={"runs_checked":len(validations),"successful_first_frames_checked":sum(v["initial_status"]==SUCCESS_EXACT for v in validations),"first_target_position_equal":all(v["first_target_position_equal"] for v in validations),"first_target_rotation_equal":all(v["first_target_rotation_equal"] for v in validations),"first_target_psi_equal":all(v["first_target_psi_equal"] for v in validations),"first_status_equal":all(v["first_status_equal"] for v in validations),"max_first_q_abs_difference_rad":max((v["max_first_q_abs_difference_rad"] for v in validations),default=0.)}
    nominal_rank=ranking["psi0_nominal"]; ranking_changes={condition:{metric:{"changed_from_nominal":order!=nominal_rank[metric],"ordering":order} for metric,order in values.items()} for condition,values in ranking.items() if condition!="psi0_nominal"}
    summary={"schema_version":"robot-r3-v1","strategies":list(STRATEGIES),"conditions":dict(zip(OFFSETS,OFFSET_VALUES.tolist())),"scientific_scope":"redundancy-policy robustness under the current fixed simulation task geometry","common_initial_state_validation":validation_summary,"initial_state_feasibility":initial_table[initial_table.grouping.eq("overall")].to_dict("records"),"ranking_stability":ranking,"ranking_changes_from_nominal":ranking_changes,"robustness_deltas":pd.DataFrame(robustness).query("grouping == 'overall'").to_dict("records"),"physical_interpretation_limits":["P-to-U translation is virtual","base mounting is provisional","fork tip is not calibrated","human collision is not modeled"],"rows":{"initial_state_feasibility":len(initial_table),"strategy_metrics":len(strategy_table),"human_consistency":len(human),"robustness_deltas":len(robustness),"pairwise_bootstrap":len(boot),"per_bite":len(per_bite),"per_take":len(take)}}
    (tmp/"summary.json").write_text(json.dumps(_json(summary),indent=2)); after=(hashlib.sha256(m0_path.read_bytes()).hexdigest(),hashlib.sha256(m1_path.read_bytes()).hexdigest()); manifest={"schema_version":"robot-r3-v1","config":asdict(cfg),"robot_r0_manifest":str(m0_path),"robot_r0_manifest_sha256":h0,"robot_r1_manifest":str(m1_path),"robot_r1_manifest_sha256":h1,"frozen_delta_psi_reused":True,"tool_trajectory_bit_identical":True,"robot_smooth_settings":{"local_offsets_rad":LOCAL_OFFSETS_RAD.tolist(),"global_recovery_points":32,"implementation":"feeding_coordination.robot_r1.robot_smooth_run"},"read_only_inputs_verified_before_after":before==after}
    (tmp/"manifest.json").write_text(json.dumps(_json(manifest),indent=2))
    if output.exists(): shutil.rmtree(output)
    tmp.replace(output); shutil.rmtree(cache_dir); return summary
