"""Phase 1: auditable human landmarks to partial task data and Stereo-SEW."""
from __future__ import annotations
import hashlib, json, subprocess
from importlib import import_module
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

LANDMARKS = ("Shoulder", "Elbow", "Wrist", "Hand")

@dataclass(frozen=True)
class Phase1Config:
    input_path: str; output_path: str; source_frame: str; working_frame: str
    raw_fork_root: str; raw_fork_pattern: str; demonstrator_by_take: dict
    R_B_L: list; R_B_from_exact_sew_base: list; translation_B_m: list; scale_m_per_mm: float; legacy_hand_offset_m: float; max_orientation_consistency_rad: float; max_derived_hand_consistency_m: float
    max_interpolation_gap_frames: int; max_interpolation_gap_seconds: float
    near_shoulder_wrist_m: float; near_plane_sine: float; near_reference_denominator: float
    @classmethod
    def load(cls, path: Path) -> "Phase1Config":
        config=cls(**json.loads(path.read_text()))
        for name in ("R_B_L", "R_B_from_exact_sew_base"):
            rotation=np.asarray(getattr(config,name),float)
            if rotation.shape != (3,3) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-12) or not np.isclose(np.linalg.det(rotation),1.0,atol=1e-12): raise ValueError(f"{name} must be proper SO(3)")
        if not config.raw_fork_root or "{take}" not in config.raw_fork_pattern or not isinstance(config.demonstrator_by_take,dict): raise ValueError("raw fork source and demonstrator_by_take are required")
        if any(getattr(config, name) <= 0 for name in ("legacy_hand_offset_m", "max_orientation_consistency_rad", "max_derived_hand_consistency_m")): raise ValueError("Phase 1.5 offsets and consistency thresholds must be positive")
        return config

def transform_points(points, cfg):
    return np.asarray(points, float) @ np.asarray(cfg.R_B_L, float).T * cfg.scale_m_per_mm + np.asarray(cfg.translation_B_m, float)

def transform_rotations(rotations, cfg): return np.asarray(cfg.R_B_L, float) @ np.asarray(rotations, float)

def _interpolate(values, observed, time, cfg):
    values=np.asarray(values,float).copy(); raw=np.asarray(observed,bool)&np.isfinite(values).all(1); interpolated=np.zeros(len(values),bool)
    for left,right in zip(np.flatnonzero(raw)[:-1],np.flatnonzero(raw)[1:]):
        gap=right-left-1
        if gap and gap<=cfg.max_interpolation_gap_frames and time[right]-time[left]<=cfg.max_interpolation_gap_seconds:
            u=(time[left+1:right]-time[left])/(time[right]-time[left]); values[left+1:right]=values[left]+u[:,None]*(values[right]-values[left]); interpolated[left+1:right]=True
    return values,raw,interpolated

def _unwrap_runs(values, valid):
    out=np.full(len(values),np.nan); i=0
    while i<len(values):
        if not valid[i]: i+=1; continue
        j=i+1
        while j<len(values) and valid[j]: j+=1
        out[i:j]=np.unwrap(values[i:j]); i=j
    return out

def _position(frame, prefix): return frame[[f"{prefix}_{x}" for x in "XYZ"]].to_numpy(float)

def process_frame(frame, cfg):
    """Process one segment while retaining all source rows and failure masks."""
    frame=frame.reset_index(drop=True); time=frame.event_time_s.to_numpy(float); landmarks={}; raw={}; interpolated={}
    for name in LANDMARKS:
        landmarks[name],raw[name],interpolated[name]=_interpolate(transform_points(_position(frame,name),cfg),frame[f"{name.lower()}_valid"].to_numpy(bool),time,cfg)
    euler=frame[["Wrist_Rx","Wrist_Ry","Wrist_Rz"]].to_numpy(float); orientation_valid=frame.wrist_rotation_valid.to_numpy(bool)&np.isfinite(euler).all(1)
    orientation=np.full((len(frame),3,3),np.nan); orientation[orientation_valid]=transform_rotations(Rotation.from_euler("xyz",euler[orientation_valid],degrees=True).as_matrix(),cfg)
    shoulder,elbow,wrist=(landmarks[x] for x in ("Shoulder","Elbow","Wrist")); observed=raw["Shoulder"]&raw["Elbow"]&raw["Wrist"]
    sw=wrist-shoulder; sw_length=np.linalg.norm(sw,axis=1); plane=np.cross(sw,elbow-shoulder); plane_length=np.linalg.norm(plane,axis=1)
    # Import only the frozen SEW module; its robot/solver modules require MuJoCo
    # but are irrelevant to this human-measurement representation.
    sew=import_module("sew_mimic.sew"); base_reference=sew.project_stereo_sew_reference(); reference_rotation=np.asarray(cfg.R_B_from_exact_sew_base,float); stereo=sew.StereoSew(sew.StereoSewReference(reference_rotation@base_reference.e_t,reference_rotation@base_reference.e_r)); StereoSewSingularityError=sew.StereoSewSingularityError
    psi=np.full(len(frame),np.nan); quality=np.full(len(frame),"sew_invalid",dtype="<U32")
    for i in np.flatnonzero(observed):
        elbow_length=np.linalg.norm(elbow[i]-shoulder[i]); plane_sine=plane_length[i]/max(sw_length[i]*elbow_length,1e-30)
        denominator=np.linalg.norm(np.cross(sw[i]/sw_length[i]-stereo.reference.e_t,stereo.reference.e_r)) if sw_length[i] else 0
        if sw_length[i]<cfg.near_shoulder_wrist_m or plane_sine<cfg.near_plane_sine or denominator<cfg.near_reference_denominator: quality[i]="sew_near_singular"; continue
        try: psi[i]=stereo.forward(shoulder[i],elbow[i],wrist[i]); quality[i]="valid"
        except StereoSewSingularityError: quality[i]="sew_invalid"
    psi_valid=np.isfinite(psi); displacement=np.full_like(shoulder,np.nan); first=np.flatnonzero(raw["Shoulder"])
    if len(first): displacement=shoulder-shoulder[first[0]]
    velocity=np.full_like(shoulder,np.nan); velocity_valid=np.zeros(len(frame),bool)
    for i in range(1,len(frame)):
        if raw["Shoulder"][i-1:i+1].all() and time[i]>time[i-1]: velocity[i]=(shoulder[i]-shoulder[i-1])/(time[i]-time[i-1]); velocity_valid[i]=True
    duration=time[-1]-time[0] if len(time)>1 else 0; normalized=(time-time[0])/duration if duration>0 else np.zeros(len(time))
    result={"time":time,"source_time_s":frame.source_time_s.to_numpy(float),"motive_frame":frame.motive_frame.to_numpy(int),"event_frame_index":frame.event_frame_index.to_numpy(int),"phase":frame.event.to_numpy(dtype=str),"phase_progress_source":frame.phase_progress.to_numpy(float),"phase_normalized_time":normalized,
      "shoulder_xyz":shoulder,"elbow_xyz":elbow,"wrist_xyz":wrist,"hand_xyz":landmarks["Hand"],"plate_position":transform_points(_position(frame,"Plate"),cfg),"plate_position_valid":np.isfinite(_position(frame,"Plate")).all(1),"plate_orientation":np.full((len(frame),3,3),np.nan),"plate_orientation_valid":np.zeros(len(frame),bool),"mouth_target_position":transform_points(frame[["target_x","target_y","target_z"]],cfg),"mouth_target_position_observed":np.isfinite(frame[["target_x","target_y","target_z"]].to_numpy(float)).all(1),"mouth_target_trusted":np.zeros(len(frame),bool),
      "hand_orientation":orientation,"hand_orientation_raw_euler_deg":euler,"orientation_valid":orientation_valid,"orientation_invalid":~orientation_valid,"tool_position":np.full((len(frame),3),np.nan),"tool_orientation":np.full((len(frame),3,3),np.nan),"tool_pose_valid":np.zeros(len(frame),bool),"tool_pose_status":np.full(len(frame),"unmatched"),"tool_pose_source":np.full(len(frame),"optitrack_fork_rigid_body"),"tool_frame_semantics":np.full(len(frame),"tracked_fork_rigid_body_frame"),"tool_match_method":np.full(len(frame),"exact_frame_identifier"),"tool_tip_calibrated":np.zeros(len(frame),bool),"tool_tip_calibration_status":np.full(len(frame),"missing_fork_tip_calibration"),"missing_calibration":np.zeros(len(frame),bool),"calibration_status":np.full(len(frame),"fork_rigid_body_pending_association"),"psi_wrapped":psi,"psi_unwrapped":_unwrap_runs(psi,psi_valid),"psi_valid":psi_valid,"psi_near_singular":quality=="sew_near_singular","psi_invalid":~psi_valid,"psi_quality":quality,"shoulder_displacement":displacement,"shoulder_velocity":velocity,"shoulder_velocity_valid":velocity_valid,"reach_ratio":np.divide(sw_length,np.linalg.norm(elbow-shoulder,axis=1)+np.linalg.norm(wrist-elbow,axis=1),out=np.full(len(frame),np.nan),where=(np.linalg.norm(elbow-shoulder,axis=1)+np.linalg.norm(wrist-elbow,axis=1))>0),"shoulder_wrist_direction":np.divide(sw,sw_length[:,None],out=np.full_like(sw,np.nan),where=sw_length[:,None]>0),"arm_plane_normal":np.divide(plane,plane_length[:,None],out=np.full_like(plane,np.nan),where=plane_length[:,None]>0),"processing_status":np.where(frame.is_complete.to_numpy(bool),"processed","source_incomplete")}
    for name in LANDMARKS:
        key=name.lower(); result[f"{key}_valid_observation"]=raw[name]; result[f"{key}_interpolated"]=interpolated[name]; result[f"{key}_long_missing"]=~(raw[name]|interpolated[name])
    return result

def _revision(root):
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    except (OSError,subprocess.CalledProcessError): return None

def process_csv(input_path, output_dir, cfg):
    input_path=Path(input_path).resolve(); output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True); demos=output_dir/"demos"; demos.mkdir(exist_ok=True)
    data=pd.read_csv(input_path); source_hash=hashlib.sha256(input_path.read_bytes()).hexdigest(); records=[]; lengths=[]; shoulder=[]; aggregate={}
    for trajectory_id,group in data.groupby("trajectory_id",sort=True):
        result=process_frame(group,cfg); filename=f"{trajectory_id}.npz"; np.savez_compressed(demos/filename,**result); lengths.append(len(group)); shoulder.extend(np.linalg.norm(result["shoulder_displacement"],axis=1)[result["shoulder_valid_observation"]])
        for key in ("psi_valid","psi_invalid","psi_near_singular","tool_pose_valid","orientation_invalid",*[f"{x.lower()}_{suffix}" for x in LANDMARKS for suffix in ("valid_observation","interpolated","long_missing")]):
            aggregate[key]=aggregate.get(key,0)+int(np.count_nonzero(result[key]))
        records.append({"record_id":trajectory_id,"source_take":group.iloc[0]["take"],"bite_id":int(group.iloc[0]["bite_id"]),"parent_bite_id":f"{group.iloc[0]['take']}_bite_{int(group.iloc[0]['bite_id']):03d}","session_id":None,"demonstrator_id":None,"recipient_id":None,"calibration_id":None,"calibration_status":"missing_tool_calibration","source_file":str(input_path),"source_hash":source_hash,"phase":group.iloc[0]["event"],"samples":len(group),"processing_status_counts":{str(x):int(np.count_nonzero(result["processing_status"]==x)) for x in np.unique(result["processing_status"])},"psi_quality_counts":{str(x):int(np.count_nonzero(result["psi_quality"]==x)) for x in np.unique(result["psi_quality"])},"psi_valid_samples":int(result["psi_valid"].sum()),"tool_pose_valid_samples":0,"file":f"demos/{filename}"})
    config=asdict(cfg); psi_valid=aggregate["psi_valid"]; total=len(data); dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=Path(__file__).resolve().parents[2],text=True).strip())
    summary={"rows":total,"demonstrations":len(records),"processed_demonstrations":len(records),"failed_demonstrations":0,"parent_bites":int(data[["take","bite_id"]].drop_duplicates().shape[0]),"takes":int(data["take"].nunique()),"demonstrator_count":None,"session_count":None,"recipient_count":None,"psi_valid_samples":psi_valid,"psi_invalid_samples":aggregate["psi_invalid"],"psi_near_singular_samples":aggregate["psi_near_singular"],"psi_valid_percent":100*psi_valid/total,"tool_pose_valid_samples":0,"tool_pose_valid_percent":0.0,"calibration_available_demonstrations":0,"calibration_missing_demonstrations":len(records),"source_complete_rows":int((data.is_complete==True).sum()),"source_incomplete_rows":int((data.is_complete==False).sum()),"orientation_invalid_samples":aggregate["orientation_invalid"],"landmark_masks":{key:aggregate[key] for key in aggregate if any(key.startswith(x.lower()) for x in LANDMARKS)},"length_frames":{"min":int(min(lengths)),"median":float(np.median(lengths)),"mean":float(np.mean(lengths)),"max":int(max(lengths))},"shoulder_displacement_m":{"max":float(np.max(shoulder)),"median":float(np.median(shoulder)),"mean":float(np.mean(shoulder)),"p95":float(np.percentile(shoulder,95))}}
    manifest={"schema_version":"phase1-v1","source_file":str(input_path),"source_hash":source_hash,"config":config,"config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"code_revision":_revision(Path(__file__).resolve().parents[2]),"code_worktree_dirty":dirty,"coordinate_convention":{"source":cfg.source_frame,"working":cfg.working_frame,"point_transform":"p_B=scale*R_B_L*p_L+t_B"},"stereo_reference":{"source_frame":"frozen Exact-SEW native Gen3 base","working_frame":cfg.working_frame,"transform":"R_B_from_exact_sew_base"},"orientation_convention":"upstream quaternion-derived extrinsic xyz degrees; scipy from_euler('xyz', degrees=True)","records":records,"summary":summary}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)); return manifest
