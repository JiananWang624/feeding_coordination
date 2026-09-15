import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from feeding_coordination.phase1 import Phase1Config, process_frame, process_csv, transform_points

CFG=Phase1Config.load(__import__('pathlib').Path('configs/phase1.json'))
def rows(n=4):
 d={'event_time_s':np.arange(n)/100.,'source_time_s':np.arange(n)/100.,'motive_frame':np.arange(n),'event_frame_index':np.arange(n),'phase_progress':np.linspace(0,1,n),'event':['transfer']*n,'is_complete':[True]*n,'Wrist_Rx':[10]*n,'Wrist_Ry':[20]*n,'Wrist_Rz':[30]*n,'wrist_rotation_valid':[True]*n}
 for name,base in [('Shoulder',(0,0,0)),('Elbow',(0,1,0)),('Wrist',(1,1,0)),('Hand',(1,1,1))]:
  for j,c in enumerate('XYZ'): d[f'{name}_{c}']=[base[j]+i*.1 for i in range(n)]
  d[f'{name.lower()}_valid']=[True]*n
 for name in ('Plate',):
  for c in 'XYZ': d[f'{name}_{c}']=[0]*n
 for name in ('target',):
  for c in 'xyz': d[f'{name}_{c}']=[0]*n
 return pd.DataFrame(d)
def test_transform_roundtrip():
 x=np.array([[3.,4.,5.]]); y=transform_points(x,CFG); assert np.allclose((y-np.array(CFG.translation_B_m))@np.array(CFG.R_B_L)/CFG.scale_m_per_mm,x)
def test_euler_extrinsic_xyz():
 out=process_frame(rows(),CFG); expected=np.array(CFG.R_B_L)@Rotation.from_euler('z',30,degrees=True).as_matrix()@Rotation.from_euler('y',20,degrees=True).as_matrix()@Rotation.from_euler('x',10,degrees=True).as_matrix(); assert np.allclose(out['hand_orientation'][0],expected)
def test_pending_fork_association_and_interpolation():
 d=rows(); d.loc[1,['Wrist_X','Wrist_Y','Wrist_Z']]=np.nan; d.loc[1,'wrist_valid']=False; out=process_frame(d,CFG); assert out['wrist_interpolated'][1] and not out['tool_pose_valid'].any(); assert set(out['calibration_status'])=={'fork_rigid_body_pending_association'}; assert not out['tool_tip_calibrated'].any()
def test_long_gap_is_explicit_and_shoulder_motion_retained():
 d=rows(6); d.loc[1:4,['Wrist_X','Wrist_Y','Wrist_Z']]=np.nan; d.loc[1:4,'wrist_valid']=False; out=process_frame(d,CFG); assert out['wrist_long_missing'][1:5].all(); assert np.linalg.norm(out['shoulder_displacement'][-1])>0
def test_psi_translation_wrap_and_singular():
 d=rows(); a=process_frame(d,CFG)['psi_wrapped'];
 for name in ('Shoulder','Elbow','Wrist'):
  d[[f'{name}_{c}' for c in 'XYZ']]+=10
 b=process_frame(d,CFG)['psi_wrapped']; assert np.allclose(a,b,equal_nan=True)
 d=rows(); d[['Elbow_X','Elbow_Y','Elbow_Z']]=d[['Shoulder_X','Shoulder_Y','Shoulder_Z']].to_numpy(); out=process_frame(d,CFG); assert not out['psi_valid'].any()
def test_unwrap_preserves_invalid_breaks():
 from feeding_coordination.phase1 import _unwrap_runs
 values=np.array([3.0,-3.0,np.nan,-3.0,3.0]); valid=np.isfinite(values); lifted=_unwrap_runs(values,valid); assert np.isnan(lifted[2]); assert np.allclose(np.angle(np.exp(1j*lifted[valid])),values[valid])
def test_frozen_csv_stereo_parity_after_reference_frame_rotation():
 from pathlib import Path
 from sew_mimic.pipeline import prepare_trajectory
 from sew_mimic.exact import human_arm_to_exact_sew_target
 root=Path('external/exact_sew'); source=root/'data'/'test.csv'; d=pd.read_csv(source).iloc[:8].copy()
 d['trajectory_id']='trial_0012_bite_001_transfer'; d['take']='trial_0012'; d['source_time_s']=np.arange(len(d))*.01; d['event_time_s']=np.arange(len(d))*.01; d['phase_progress']=np.linspace(0,1,len(d)); d['wrist_rotation_valid']=True; d['hand_valid']=True; d['wrist_valid']=True; d['elbow_valid']=True; d['shoulder_valid']=True; d['is_complete']=True
 prepared=prepare_trajectory(source,max_frames=8); actual=process_frame(d,CFG)['psi_wrapped']; expected=np.array([human_arm_to_exact_sew_target(x.target,prepared.stereo).psi for x in prepared.frames]); assert np.allclose(actual,expected,atol=1e-10)
def test_deterministic(tmp_path):
 d=rows(); d['trajectory_id']='x'; d['take']='trial';d['bite_id']=1; path=tmp_path/'x.csv'; d.to_csv(path,index=False); m1=process_csv(path,tmp_path/'a',CFG); m2=process_csv(path,tmp_path/'b',CFG); assert m1==m2
 a=np.load(tmp_path/'a'/'demos'/'x.npz',allow_pickle=True); b=np.load(tmp_path/'b'/'demos'/'x.npz',allow_pickle=True); assert a.files==b.files
 for key in a.files:
  if np.issubdtype(a[key].dtype,np.number): assert np.array_equal(a[key],b[key],equal_nan=True)
  else: assert np.array_equal(a[key],b[key])
