"""Robot R2 read-only quantitative analysis of frozen Robot R1 trajectories."""
from __future__ import annotations
import hashlib, json, shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .phase2 import SUCCESS_EXACT
from .robot_r1 import clean_motion, wrap
from .robot_visualization import load_record, mount_record, set_stored_q, STRATEGIES

PRIMARY = (("StrongLocal","B0"),("StrongLocal","B2"),("StrongLocal","B3"),("StrongLocal","H_star"),("StrongLocal","RobotSmooth"),("StrongLocal","HumanGT"),("RobotSmooth","HumanGT"))

def freeze_arrays(arrays: dict[str,np.ndarray]) -> None:
    """Defend frozen R1 payloads against accidental in-process mutation."""
    for value in arrays.values(): value.setflags(write=False)

@dataclass(frozen=True)
class RobotR2Config:
    robot_r1_output_path: str="outputs/robot_r1"
    output_path: str="outputs/robot_r2"
    continuity_threshold_rad: float=.5
    bootstrap_seed: int=20260915
    bootstrap_resamples: int=2000
    @classmethod
    def load(cls,path:Path):
        x=cls(**json.loads(Path(path).read_text()))
        if (x.continuity_threshold_rad,x.bootstrap_seed,x.bootstrap_resamples)!=(.5,20260915,2000): raise ValueError("R2 frozen threshold/bootstrap differs")
        return x

def arm_plane_angle(a:np.ndarray,b:np.ndarray)->float:
    """Stable angle between frozen SEW arm-plane normals; NaN when degenerate."""
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.arccos(np.clip(np.dot(a,b)/(na*nb),-1,1))) if na>1e-12 and nb>1e-12 else np.nan

def _dist(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]  # preserve rank-deficient +inf condition numbers
    return {"count":int(len(x)),"min":float(x.min()) if len(x) else np.nan,"mean":float(x.mean()) if len(x) else np.nan,"median":float(np.median(x)) if len(x) else np.nan,"p05":float(np.percentile(x,5)) if len(x) else np.nan,"p95":float(np.percentile(x,95)) if len(x) else np.nan,"max":float(x.max()) if len(x) else np.nan,"rms":float(np.sqrt(np.mean(x*x))) if len(x) else np.nan}

def pair_clean(success_a,success_b,viol_a,viol_b,frames,runs):
    """Pair C: common success, then R1's exact split-before-violation semantics."""
    success=np.asarray(success_a,bool)&np.asarray(success_b,bool)
    violation=np.asarray(viol_a,bool)|np.asarray(viol_b,bool)
    return success, violation, clean_motion(np.zeros((len(success),7)),np.arange(len(success),dtype=float),frames,runs,success,violation)["continuous_segments"]

def jacobian_metrics(robot,data,q):
    import mujoco
    site=mujoco.mj_name2id(robot.model,mujoco.mjtObj.mjOBJ_SITE,"pinch_site")
    if site<0: raise RuntimeError("aligned pinch_site absent from frozen mounted model")
    set_stored_q(robot,data,q); jp=np.zeros((3,robot.model.nv)); jr=np.zeros((3,robot.model.nv)); mujoco.mj_jacSite(robot.model,data,jp,jr,site)
    cols=np.asarray([robot.model.jnt_dofadr[int(j)] for j in robot.joint_ids],int)
    def one(j):
        s=np.linalg.svd(j[:,cols],compute_uv=False); return float(s[-1]), (float(np.inf) if s[-1]<=np.finfo(float).eps else float(s[0]/s[-1]))
    return (*one(jp),*one(jr))

def self_collision_capability_audit():
    """Audit the frozen model once; do not turn unvalidated contacts into a metric."""
    import mujoco
    from sew_mimic.kinematics import gen3_kinematics
    robot=gen3_kinematics(); model=robot.model; data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    collision_geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]>0 and model.geom_contype[i]!=0 and model.geom_conaffinity[i]!=0]
    contacts=[]
    for contact in data.contact[:data.ncon]:
        body_a=int(model.geom_bodyid[contact.geom1]); body_b=int(model.geom_bodyid[contact.geom2])
        if body_a>0 and body_b>0:
            contacts.append({"body_a":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body_a),"body_b":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body_b),"distance_m":float(contact.dist)})
    validated=bool(collision_geoms and model.npair>0 and model.nexclude>0 and not any(item["distance_m"]<0 for item in contacts))
    return {"status":"SELF_COLLISION_CAPABILITY_VALIDATED" if validated else "SELF_COLLISION_METRIC_UNAVAILABLE_MODEL_NOT_VALIDATED","evidence":{"collision_meshes":len(collision_geoms),"explicit_contact_pairs":int(model.npair),"explicit_contact_excludes":int(model.nexclude),"zero_configuration_robot_contacts":contacts}}

def retiming_readiness_audit(root:Path):
    """Look for authoritative velocity/acceleration limits without inventing values."""
    model_dir=Path(root)/"external/exact_sew/assets/kinova_gen3"
    files=sorted(model_dir.glob("*.xml")); needles=("velrange","velocity_limit","acceleration_limit","max_velocity","max_acceleration")
    matches={str(path.relative_to(root)):[needle for needle in needles if needle in path.read_text().lower()] for path in files}
    matches={path:found for path,found in matches.items() if found}
    return {"status":"RETIMING_LIMITS_AUTHORITATIVE" if matches else "RETIMING_LIMITS_NOT_YET_FROZEN","evidence":{"scanned_model_files":[str(path.relative_to(root)) for path in files],"velocity_or_acceleration_limit_matches":matches,"position_limit_joints":[2,4,6]}}

def _branch_count(a,strategy,success,frames,runs):
    branch=a[f"{strategy}_branch_changed"] if strategy!="RobotSmooth" else None
    total=0
    for i in range(1,len(success)):
        edge=success[i-1] and success[i] and runs[i]==runs[i-1] and frames[i]==frames[i-1]+1
        if edge: total+=int(bool(branch[i]) if branch is not None else (a[f"{strategy}_search_branch"][i]!=a[f"{strategy}_search_branch"][i-1] and str(a[f"{strategy}_search_branch"][i])))
    return total

def _strategy_bite(record,strategy,geometry,robot,data):
    a=record.arrays; n=record.frames; ok=(a[f"{strategy}_evaluation_status"]==SUCCESS_EXACT)&a["common_source_valid"].astype(bool); viol=a[f"{strategy}_continuity_violation"].astype(bool)
    motion=clean_motion(a[f"{strategy}_q"],a["time_s"],a["motive_frame"],a["run_id"],ok,viol)
    def absvals(k): return np.abs(motion[k][np.isfinite(motion[k])])
    margins=a[f"{strategy}_joint_limit_margin_rad"][ok]; vals={"travel":float(np.nansum(np.abs(motion["clean_wrapped_delta_q_rad"]))),"rms_velocity":_dist(absvals("clean_joint_velocity_rad_s"))["rms"],"p95_velocity":_dist(absvals("clean_joint_velocity_rad_s"))["p95"],"rms_acceleration":_dist(absvals("clean_joint_acceleration_rad_s2"))["rms"],"p95_acceleration":_dist(absvals("clean_joint_acceleration_rad_s2"))["p95"],"rms_jerk":_dist(absvals("clean_joint_jerk_rad_s3"))["rms"],"p95_jerk":_dist(absvals("clean_joint_jerk_rad_s3"))["p95"],"max_clean_step":_dist(absvals("clean_wrapped_delta_q_rad"))["max"],"success_frames":int(ok.sum()),"source_valid_frames":int(a["common_source_valid"].sum()),"violations":int((viol&a["common_source_valid"]).sum()),"continuous_frame_percent":100*int((ok&~viol).sum())/max(1,int(ok.sum())),"branch_changes":_branch_count(a,strategy,ok,a["motive_frame"],a["run_id"])}
    vals.update({"travel_edges":int(np.isfinite(motion["clean_wrapped_delta_q_rad"]).all(1).sum()),"velocity_frames":int(np.isfinite(motion["clean_joint_velocity_rad_s"]).all(1).sum()),"acceleration_frames":int(np.isfinite(motion["clean_joint_acceleration_rad_s2"]).all(1).sum()),"jerk_frames":int(np.isfinite(motion["clean_joint_jerk_rad_s3"]).all(1).sum())})
    for k,v in _dist(margins).items(): vals[f"margin_{k}"]=v
    for t in (.01,.05,.10): vals[f"margin_below_{t:.2f}"]=float(np.mean(margins<t)) if len(margins) else np.nan
    j=[]; pts=[]; jac_by_frame=np.full((n,4),np.nan)
    for i in np.flatnonzero(ok):
        q=a[f"{strategy}_q"][i]; p=geometry.sew_points(q); normal=np.cross(p.wrist-p.shoulder,p.elbow-p.shoulder); pts.append((p.elbow,normal)); one=jacobian_metrics(robot,data,q); j.append(one); jac_by_frame[i]=one
    vals["points"]=pts; vals["ok"]=ok; vals["motion"]=motion; vals["jac_by_frame"]=jac_by_frame
    if j:
        j=np.asarray(j); 
        for name,col in (("jp_sigma",0),("jp_cond",1),("jr_sigma",2),("jr_cond",3)):
            d=_dist(j[:,col]); vals[f"{name}_min"]=d["min"];vals[f"{name}_p05"]=d["p05"];vals[f"{name}_median"]=d["median"];vals[f"{name}_p95"]=d["p95"]
    else:
        for name in ("jp_sigma","jp_cond","jr_sigma","jr_cond"):
            for suffix in ("min","p05","median","p95"): vals[f"{name}_{suffix}"]=np.nan
    return vals

def _human(a,b,record,also_ok=None):
    ar,br=record.arrays,record.arrays; ok=a["ok"]&b["ok"] if also_ok is None else a["ok"]&b["ok"]&np.asarray(also_ok,bool); ia=np.flatnonzero(ok); qa=ar["{s}_q".format(s=a["strategy"])] [ok]; qb=br["{s}_q".format(s=b["strategy"])][ok]
    psi=np.abs(wrap(ar[f"{a['strategy']}_strategy_psi"][ok]-ar[f"{b['strategy']}_strategy_psi"][ok])); dq=np.abs(wrap(qa-qb)); ql=np.linalg.norm(dq,axis=1)
    ea=[]; pa=[]
    pa_map={i:x for i,x in zip(np.flatnonzero(a["ok"]),a["points"])}; pb_map={i:x for i,x in zip(np.flatnonzero(b["ok"]),b["points"])}
    for i in ia:
        ea.append(np.linalg.norm(pa_map[i][0]-pb_map[i][0])); pa.append(arm_plane_angle(pa_map[i][1],pb_map[i][1]))
    return {"common_success_frames":int(len(ia)),"psi_rmse":_dist(psi)["rms"],"psi_mae":_dist(psi)["mean"],"psi_p95":_dist(psi)["p95"],"psi_max":_dist(psi)["max"],"q_mean":_dist(ql)["mean"],"q_rms":_dist(ql)["rms"],"q_p95":_dist(ql)["p95"],**{f"q_joint{j+1}_mae":_dist(dq[:,j])["mean"] for j in range(7)},**{f"elbow_{k}":v for k,v in _dist(ea).items() if k in ("mean","median","p95","max")},**{f"plane_{k}":v for k,v in _dist(pa).items() if k in ("mean","median","p95","max")}}

def _pair_motion(record,a,b,cache=None):
    x=record.arrays; aa=(x[f"{a}_evaluation_status"]==SUCCESS_EXACT)&x["common_source_valid"].astype(bool); bb=(x[f"{b}_evaluation_status"]==SUCCESS_EXACT)&x["common_source_valid"].astype(bool); suc,vio,segs=pair_clean(aa,bb,x[f"{a}_continuity_violation"],x[f"{b}_continuity_violation"],x["motive_frame"],x["run_id"])
    out={"pair_common_success_frames":int(suc.sum()),"pair_clean_frames":int((suc&~vio).sum()),"pair_clean_intervals":sum(max(0,len(z)-1) for z in segs)}
    for side in (a,b):
        m=clean_motion(x[f"{side}_q"],x["time_s"],x["motive_frame"],x["run_id"],suc,vio)
        if side==a:
            out.update({"pair_travel_edges":int(np.isfinite(m["clean_wrapped_delta_q_rad"]).all(1).sum()),"pair_velocity_frames":int(np.isfinite(m["clean_joint_velocity_rad_s"]).all(1).sum()),"pair_acceleration_frames":int(np.isfinite(m["clean_joint_acceleration_rad_s2"]).all(1).sum()),"pair_jerk_frames":int(np.isfinite(m["clean_joint_jerk_rad_s3"]).all(1).sum())})
        for label,key in (("travel","clean_wrapped_delta_q_rad"),("rms_velocity","clean_joint_velocity_rad_s"),("rms_acceleration","clean_joint_acceleration_rad_s2"),("rms_jerk","clean_joint_jerk_rad_s3")):
            vals=np.abs(m[key][np.isfinite(m[key])]); out[f"{side}_{label}"]=float(np.nansum(vals)) if label=="travel" else _dist(vals)["rms"]
        mar=x[f"{side}_joint_limit_margin_rad"][suc&~vio]; out[f"{side}_min_margin"]=_dist(mar)["min"]
        if cache is not None:
            jac=cache[side]["jac_by_frame"][suc&~vio]
            out[f"{side}_min_jp_sigma"]=_dist(jac[:,0])["min"]; out[f"{side}_min_jr_sigma"]=_dist(jac[:,2])["min"]
    return out

def _bootstrap(rows, metric, seed, n):
    values=rows[["parent_bite_id",metric]].copy(); values[metric]=pd.to_numeric(values[metric],errors="coerce")
    v=values.groupby("parent_bite_id",sort=True)[metric].mean().to_numpy(float); v=v[np.isfinite(v)]
    if not len(v): return (np.nan,np.nan,np.nan,0,"NO_FINITE_PARENT_BITE_VALUES")
    rng=np.random.default_rng(seed); d=rng.choice(v,(n,len(v)),replace=True).mean(1); return (float(v.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),len(v),"OK")

def _json(x):
    if isinstance(x,dict): return {k:_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x

def process_robot_r2(root:Path, output:Path, cfg:RobotR2Config):
    root=Path(root); output=Path(output); r1=root/cfg.robot_r1_output_path; manifest=json.loads((r1/"manifest.json").read_text()); before=hashlib.sha256((r1/"manifest.json").read_bytes()).hexdigest(); tmp=output.with_name(output.name+".tmp")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True); (tmp/"plots").mkdir()
    from sew_mimic.sew import Gen3StereoSewGeometry
    bite=[]; human=[]; pairs=[]; feas=[]; mounted={}
    for entry in manifest["records"]:
        rec=load_record(entry["record_id"],root)
        freeze_arrays(rec.arrays)
        take_name=str(rec.arrays["take"][0])
        if take_name not in mounted: mounted[take_name]=mount_record(rec)[:2]
        robot,data=mounted[take_name]; geo=Gen3StereoSewGeometry.from_robot(robot); base={"record_id":rec.record_id,"parent_bite_id":str(rec.arrays["parent_bite_id"][0]),"take":take_name,"phase":str(rec.arrays["phase"][0]),"source_valid_frames":int(rec.arrays["common_source_valid"].sum())}; cache={}
        for s in STRATEGIES:
            z=_strategy_bite(rec,s,geo,robot,data); z["strategy"]=s; cache[s]=z; bite.append(base|{k:v for k,v in z.items() if k not in ("points","ok","motion","jac_by_frame")}); feas.append(base|{"strategy":s,**{k:z[k] for k in ("source_valid_frames","success_frames","violations","continuous_frame_percent","branch_changes","margin_min")}})
        for s in STRATEGIES:
            if s=="HumanGT": continue
            h=_human(cache[s],cache["HumanGT"],rec); human.append(base|{"strategy":s,"reference":"HumanGT",**h})
        for a,b in PRIMARY:
            p=_pair_motion(rec,a,b,cache)
            # Fair A-vs-B human consistency: every value uses the exact same A/B/HumanGT frames.
            triple=cache[a]["ok"]&cache[b]["ok"]&cache["HumanGT"]["ok"]
            hA=_human(cache[a],cache["HumanGT"],rec,cache[b]["ok"])
            hB=_human(cache[b],cache["HumanGT"],rec,cache[a]["ok"])
            human_pair_metrics=("psi_mae","q_mean","q_rms","q_p95",*(f"q_joint{j}_mae" for j in range(1,8)),"elbow_mean","plane_mean","plane_p95")
            pair_human={f"{side}_{metric}_to_humangt":values[metric] for side,values in (("a",hA),("b",hB)) for metric in human_pair_metrics}
            pairs.append(base|{"comparison":f"{a}-{b}","strategy_a":a,"strategy_b":b,**p,"triple_common_success_frames":int(triple.sum()),**pair_human,"a_violations":cache[a]["violations"],"b_violations":cache[b]["violations"]})
    bite=pd.DataFrame(bite); human=pd.DataFrame(human); pairs=pd.DataFrame(pairs); feas=pd.DataFrame(feas)
    # wide comparison rows, explicitly preserving values that feed each bootstrap.
    for metric in ("travel","rms_velocity","rms_acceleration","rms_jerk","min_margin","min_jp_sigma","min_jr_sigma","violations"):
        if metric=="violations":
            pairs[f"a_{metric}"]=pairs["a_violations"]; pairs[f"b_{metric}"]=pairs["b_violations"]
        else:
            pairs[f"a_{metric}"]=pairs.apply(lambda r:r[f"{r.strategy_a}_{metric}"],axis=1); pairs[f"b_{metric}"]=pairs.apply(lambda r:r[f"{r.strategy_b}_{metric}"],axis=1)
        pairs[f"difference_{metric}_a_minus_b"]=pairs[f"a_{metric}"]-pairs[f"b_{metric}"]
    for metric in ("psi_mae","q_mean","q_rms","q_p95",*(f"q_joint{j}_mae" for j in range(1,8)),"elbow_mean","plane_mean","plane_p95"):
        pairs[f"difference_{metric}_a_minus_b"]=pairs[f"a_{metric}_to_humangt"]-pairs[f"b_{metric}_to_humangt"]
    bootstrap_metrics=("travel","rms_velocity","rms_acceleration","rms_jerk","min_margin","min_jp_sigma","min_jr_sigma","violations","psi_mae","elbow_mean")
    boot=[]
    for comp,g in pairs.groupby("comparison"):
        for scope,sub in [("overall",g),("transfer",g[g.phase=="transfer"]),("withdrawal",g[g.phase=="withdrawal"])]:
            for metric in (f"difference_{name}_a_minus_b" for name in bootstrap_metrics):
                d,lo,hi,n,status=_bootstrap(sub,metric,cfg.bootstrap_seed,cfg.bootstrap_resamples); boot.append({"comparison":comp,"phase":scope,"metric":metric,"difference_a_minus_b":d,"ci_low":lo,"ci_high":hi,"support_bites":n,"status":status})
    def grouped(df,cols,metrics):
        return df.groupby(cols,dropna=False)[metrics].mean().reset_index()
    # Feasibility frame/event counts are sums; posture/motion summaries below are bite-balanced means.
    feasibility=feas.groupby("strategy",as_index=False)[["source_valid_frames","success_frames","violations","branch_changes"]].sum(); feasibility["continuous_frame_percent"]=100*(feas.groupby("strategy").apply(lambda x:(x.success_frames-x.violations).sum()/max(1,x.success_frames.sum()),include_groups=False).to_numpy()); feasibility["margin_min"]=feas.groupby("strategy").margin_min.min().to_numpy(); feasibility["scope"]="overall"; feasibility["success_percent"]=100*feasibility.success_frames/feasibility.source_valid_frames
    for ph in ("transfer","withdrawal"):
        f=feas[feas.phase==ph]; z=f.groupby("strategy",as_index=False)[["source_valid_frames","success_frames","violations","branch_changes"]].sum();z["continuous_frame_percent"]=100*(f.groupby("strategy").apply(lambda x:(x.success_frames-x.violations).sum()/max(1,x.success_frames.sum()),include_groups=False).to_numpy());z["margin_min"]=f.groupby("strategy").margin_min.min().to_numpy();z["scope"]=ph;z["success_percent"]=100*z.success_frames/z.source_valid_frames;feasibility=pd.concat((feasibility,z),ignore_index=True)
    motion_cols=["travel","rms_velocity","p95_velocity","rms_acceleration","p95_acceleration","rms_jerk","p95_jerk","max_clean_step","violations","branch_changes","margin_min","margin_p05","margin_mean","margin_below_0.01","margin_below_0.05","margin_below_0.10","travel_edges","velocity_frames","acceleration_frames","jerk_frames"]
    motion=grouped(bite,["strategy"],motion_cols); motion["scope"]="overall"
    kin_cols=[c for c in bite if c.startswith("jp_") or c.startswith("jr_")];kin=grouped(bite,["strategy"],kin_cols);kin["scope"]="overall"
    human_cols=[c for c in human if c not in ("record_id","parent_bite_id","take","phase","strategy","reference")];ht=grouped(human,["strategy","reference"],human_cols);ht["scope"]="overall"
    for ph in ("transfer","withdrawal"):
        x=grouped(bite[bite.phase==ph],["strategy"],motion_cols);x["scope"]=ph;motion=pd.concat((motion,x),ignore_index=True)
        x=grouped(bite[bite.phase==ph],["strategy"],kin_cols);x["scope"]=ph;kin=pd.concat((kin,x),ignore_index=True)
        x=grouped(human[human.phase==ph],["strategy","reference"],human_cols);x["scope"]=ph;ht=pd.concat((ht,x),ignore_index=True)
    x=grouped(human,["take","strategy","reference"],human_cols);x["scope"]="take";ht=pd.concat((ht,x),ignore_index=True)
    take=grouped(bite,["take","strategy"],["travel","rms_velocity","rms_acceleration","rms_jerk","margin_min","jp_sigma_min","jr_sigma_min","violations"])
    take["support_bites"]=take.apply(lambda r:int(bite[(bite["take"]==r["take"])&(bite.strategy==r["strategy"])].parent_bite_id.nunique()),axis=1)
    take["support_success_frames"]=take.apply(lambda r:int(bite[(bite["take"]==r["take"])&(bite.strategy==r["strategy"])].success_frames.sum()),axis=1)
    # Explicit support accounting: aggregations are bite-balanced means except feasibility counts.
    for table in (motion,kin):
        table["support_bites"]=table.apply(lambda r:int(bite[(bite.strategy==r.strategy)&((bite.phase==r.scope) if r.scope in ("transfer","withdrawal") else True)].parent_bite_id.nunique()),axis=1)
        table["support_success_frames"]=table.apply(lambda r: int(feas[(feas.strategy==r.strategy)&((feas.phase==r.scope) if r.scope in ("transfer","withdrawal") else True)].success_frames.sum()),axis=1)
    for col in ("travel_edges","velocity_frames","acceleration_frames","jerk_frames"):
        motion[col]=motion.apply(lambda r:int(bite[(bite.strategy==r.strategy)&((bite.phase==r.scope) if r.scope in ("transfer","withdrawal") else True)][col].sum()),axis=1)
    ht["support_bites"]=ht.apply(lambda r:int(human[(human.strategy==r["strategy"])&((human.phase==r["scope"]) if r["scope"] in ("transfer","withdrawal") else True)&((human["take"]==r["take"]) if r["scope"]=="take" else True)].parent_bite_id.nunique()),axis=1)
    ht["support_common_success_frames"]=ht.apply(lambda r: int(human[(human.strategy==r["strategy"])&((human.phase==r["scope"]) if r["scope"] in ("transfer","withdrawal") else True)&((human["take"]==r["take"]) if r["scope"]=="take" else True)].common_success_frames.sum()),axis=1)
    ht=ht.rename(columns={"common_success_frames":"bite_mean_common_success_frames"})
    def pair_summary_rows(kind):
        rows=[]
        metrics=("travel","rms_velocity","rms_acceleration","rms_jerk","min_margin","violations") if kind=="motion" else ("min_jp_sigma","min_jr_sigma")
        for comp,g in pairs.groupby("comparison",sort=True):
            for scope,sub in (("overall",g),("transfer",g[g.phase.eq("transfer")]),("withdrawal",g[g.phase.eq("withdrawal")])):
                row={"row_scope":"pairwise_support_c","comparison":comp,"strategy_a":str(g.strategy_a.iloc[0]),"strategy_b":str(g.strategy_b.iloc[0]),"scope":scope,"support_bites":int(sub.parent_bite_id.nunique()),"pair_common_success_frames":int(sub.pair_common_success_frames.sum()),"pair_clean_frames":int(sub.pair_clean_frames.sum()),"pair_clean_intervals":int(sub.pair_clean_intervals.sum())}
                for metric in metrics:
                    row[f"a_{metric}"]=float(pd.to_numeric(sub[f"a_{metric}"],errors="coerce").mean())
                    row[f"b_{metric}"]=float(pd.to_numeric(sub[f"b_{metric}"],errors="coerce").mean())
                if kind=="motion":
                    for support in ("pair_travel_edges","pair_velocity_frames","pair_acceleration_frames","pair_jerk_frames"): row[support]=int(sub[support].sum())
                rows.append(row)
        return pd.DataFrame(rows)
    motion["row_scope"]="single_strategy"; kin["row_scope"]="single_strategy"
    motion_output=pd.concat((motion,pair_summary_rows("motion")),ignore_index=True,sort=False)
    kin_output=pd.concat((kin,pair_summary_rows("kinematic")),ignore_index=True,sort=False)
    singles=bite.copy(); singles["row_scope"]="single_strategy"; pairs["row_scope"]="primary_pair"
    feasibility.to_csv(tmp/"feasibility_metrics.csv",index=False);ht.to_csv(tmp/"human_consistency_metrics.csv",index=False);motion_output.to_csv(tmp/"motion_metrics.csv",index=False);kin_output.to_csv(tmp/"kinematic_metrics.csv",index=False);pd.DataFrame(boot).to_csv(tmp/"pairwise_bootstrap.csv",index=False);pd.concat((singles,pairs),sort=False).to_csv(tmp/"per_bite_metrics.csv",index=False);take.to_csv(tmp/"per_take_metrics.csv",index=False)
    collision=self_collision_capability_audit(); retiming=retiming_readiness_audit(root)
    (tmp/"self_collision_audit.json").write_text(json.dumps(collision,indent=2));(tmp/"retiming_readiness.json").write_text(json.dumps(retiming,indent=2))
    def chosen(frame, score, rule):
        x=frame.copy(); x["_score"]=score(x); x=x[np.isfinite(x._score)].sort_values(["_score","record_id"],ascending=[False,True])
        return {"record_id":str(x.iloc[0].record_id),"score":float(x.iloc[0]._score),"criterion":rule} if len(x) else {"status":"NO_ELIGIBLE_RECORD","criterion":rule}
    sl_b0=pairs[pairs.comparison.eq("StrongLocal-B0")].set_index("record_id"); sl_b2=pairs[pairs.comparison.eq("StrongLocal-B2")].set_index("record_id")
    common=sl_b0.index.intersection(sl_b2.index); aa=pd.DataFrame({"record_id":common,"b0":sl_b0.loc[common,"b_elbow_mean_to_humangt"].to_numpy()-sl_b0.loc[common,"a_elbow_mean_to_humangt"].to_numpy(),"b2":sl_b2.loc[common,"b_elbow_mean_to_humangt"].to_numpy()-sl_b2.loc[common,"a_elbow_mean_to_humangt"].to_numpy()})
    examples={"selection_rules":"deterministic quantitative criteria; ties record_id", "examples":[
        {"name":"A","value":chosen(aa,lambda x:np.minimum(x.b0,x.b2),"maximize positive min(B0,B2) StrongLocal elbow improvement")},
        {"name":"B","value":chosen(pairs[pairs.comparison.eq("StrongLocal-RobotSmooth")],lambda x:np.where((x.a_rms_velocity>x.b_rms_velocity)&(x.b_psi_mae_to_humangt>x.a_psi_mae_to_humangt),x.a_travel-x.b_travel,np.nan),"RobotSmooth smoother and more HumanGT-divergent; maximize StrongLocal-minus-RobotSmooth travel")},
        {"name":"C","value":chosen(pairs[pairs.comparison.eq("RobotSmooth-HumanGT")&((pairs.pair_common_success_frames/pairs.source_valid_frames)>=.95)],lambda x:x.a_elbow_mean_to_humangt,"identical frozen desired U; at least 95% pair common success; maximize RobotSmooth-vs-HumanGT elbow divergence")},
        {"name":"D","value":chosen(bite[(bite.strategy.eq("StrongLocal"))&(bite.phase.eq("withdrawal"))],lambda x:x.rms_velocity,"withdrawal maximum StrongLocal RMS velocity")}]}
    (tmp/"representative_examples.json").write_text(json.dumps(_json(examples),indent=2))
    # Compact figures, all sourced from the public summary tables.
    import matplotlib.pyplot as plt
    for name,df,y in (("human_consistency",ht[ht.scope.eq("overall")],"psi_mae"),("motion_quality",motion[motion.scope.eq("overall")],"rms_velocity"),("kinematic_quality",kin[kin.scope.eq("overall")],"jp_sigma_min"),("tradeoff_humangt_robotsmooth",motion[(motion.strategy.isin(["HumanGT","RobotSmooth"]))&(motion.scope.eq("overall"))],"travel")):
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(df.strategy.astype(str),pd.to_numeric(df[y],errors="coerce"));ax.set_ylabel(y);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(tmp/"plots"/f"{name}.png",dpi=150);plt.close(fig)
    phase_plot=feasibility[feasibility.scope.isin(["transfer","withdrawal"])].pivot(index="strategy",columns="scope",values="success_percent")
    fig,ax=plt.subplots(figsize=(9,4)); phase_plot.plot.bar(ax=ax);ax.set_ylabel("success_percent");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(tmp/"plots"/"transfer_withdrawal.png",dpi=150);plt.close(fig)
    def comp_mean(name):
        x=pairs[pairs.comparison.eq(name)]; metrics=("difference_psi_mae_a_minus_b","difference_elbow_mean_a_minus_b","difference_travel_a_minus_b","difference_rms_velocity_a_minus_b","difference_rms_acceleration_a_minus_b","difference_rms_jerk_a_minus_b","difference_min_margin_a_minus_b","difference_min_jp_sigma_a_minus_b","difference_min_jr_sigma_a_minus_b","difference_violations_a_minus_b"); balanced=x.groupby("parent_bite_id",sort=True)[list(metrics)].mean(); return {k:float(balanced[k].mean()) for k in metrics}
    tw=motion[motion.scope.isin(["transfer","withdrawal"])].pivot(index="strategy",columns="scope",values="rms_velocity")
    ratios={s:float(tw.loc[s,"withdrawal"]/tw.loc[s,"transfer"]) for s in tw.index if np.isfinite(tw.loc[s,"transfer"]) and tw.loc[s,"transfer"]!=0}
    take_abnormal=take.sort_values("rms_velocity",ascending=False).head(10)[["take","strategy","rms_velocity","travel","margin_min"]].to_dict("records")
    summary={"schema_version":"robot-r2-v1","strategies":list(STRATEGIES),"input_robot_r1_manifest_sha256":before,"support":"A=source-valid; B=pairwise SUCCESS_EXACT; C=pairwise common success split by union saved violations; human A-vs-B differences use triple A/B/HumanGT success","collision":collision,"retiming_status":retiming["status"],"scope_exclusions":["IK","smoothing","retiming","calibration optimization","new learning","human/environment clearance"],"empirical_findings":{"stronglocal_vs_b2":comp_mean("StrongLocal-B2"),"stronglocal_vs_b3":comp_mean("StrongLocal-B3"),"robotsmooth_minus_humangt":comp_mean("RobotSmooth-HumanGT"),"withdrawal_to_transfer_rms_velocity_ratio":ratios,"highest_per_take_motion_rows":take_abnormal},"rows":{"feasibility":len(feasibility),"human":len(ht),"motion":len(motion_output),"kinematic":len(kin_output),"bootstrap":len(boot),"per_bite":len(singles)+len(pairs)}}
    (tmp/"summary.json").write_text(json.dumps(_json(summary),indent=2)); (tmp/"manifest.json").write_text(json.dumps({"schema_version":"robot-r2-v1","config":asdict(cfg),"r1_manifest":str(r1/"manifest.json"),"r1_manifest_sha256":before,"robot_smooth_branch_count_provenance":"adjacent successful contiguous saved search_branch transitions","read_only_input_verified_before_after":before==hashlib.sha256((r1/"manifest.json").read_bytes()).hexdigest()},indent=2))
    if output.exists(): shutil.rmtree(output)
    tmp.replace(output); return summary
