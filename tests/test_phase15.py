import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from feeding_coordination.phase1 import Phase1Config, process_frame
from feeding_coordination.phase15 import associate_fork, association_and_consistency_status, load_raw_fork, trajectory_metadata
from test_phase1 import rows

CFG = Phase1Config.load(__import__('pathlib').Path('configs/phase1.json'))


def raw_rows():
    return pd.DataFrame({"host_time_s":[1.,2.],"natnet_frame_number":[10,11],"rigid_body_name":["fork","fork"],"x":[1.,2.],"y":[2.,3.],"z":[3.,4.],"qx":[0.,0.],"qy":[0.,0.],"qz":[0.,np.sin(np.pi/4)],"qw":[1.,np.cos(np.pi/4)],"tracking_valid":[1,0]}).set_index("natnet_frame_number",drop=False)


def frame():
    d=rows(3); d["motive_frame"]=[10,10,12]; d["source_time_s"]=[.1,.1,.3]; return d


def test_raw_metres_xyzw_b_rotation_and_duplicate_frame_key():
    d=frame(); out=associate_fork(d,process_frame(d,CFG),raw_rows(),CFG)
    assert np.allclose(out["tool_position"][0],np.array(CFG.R_B_L)@np.array([1.,2.,3.]))
    assert np.allclose(out["tool_position"][1],out["tool_position"][0])
    assert np.allclose(out["tool_orientation"][0],np.array(CFG.R_B_L))
    assert out["tool_match_exact"][:2].all() and (out["tool_match_frame_error"][:2] == 0).all()


def test_invalid_and_unmatched_fork_remain_nan_and_explicit():
    d=frame(); out=associate_fork(d,process_frame(d,CFG),raw_rows(),CFG)
    assert out["tool_pose_status"].tolist()==["valid","valid","unmatched"]
    assert not out["tool_pose_valid"][2] and np.isnan(out["tool_position"][2]).all()
    d.loc[0,"motive_frame"]=11; out=associate_fork(d,process_frame(d,CFG),raw_rows(),CFG)
    assert out["tool_pose_status"][0]=="tracking_invalid" and np.isnan(out["tool_orientation"][0]).all()


def test_loader_rejects_duplicate_raw_natnet_frame(tmp_path):
    d=raw_rows().reset_index(drop=True); d.loc[1,"natnet_frame_number"]=10; path=tmp_path/'raw.csv'; d.to_csv(path,index=False)
    try: load_raw_fork(path)
    except ValueError as error: assert "not unique" in str(error)
    else: raise AssertionError("duplicate raw frame must be rejected")


def test_frozen_demonstrator_session_and_unknown_recipient_metadata():
    assert trajectory_metadata("trial_0012", CFG)=={"session_id":"trial_0012","demonstrator_id":"D0","recipient_id":None}
    assert trajectory_metadata("trial_0015", CFG)=={"session_id":"trial_0015","demonstrator_id":"D1","recipient_id":None}


def test_partial_association_is_reported_not_a_processing_failure():
    matching={"overall":{"exact":1,"total":2}}
    orientation={"max":CFG.max_orientation_consistency_rad / 2}
    hand={"max":CFG.max_derived_hand_consistency_m / 2}
    assert association_and_consistency_status(matching,orientation,hand,CFG)==("partial","verified")


def test_compact_orientation_and_hand_relation_consistency():
    d=rows(1); d["motive_frame"]=[10]
    raw=raw_rows().iloc[:1].copy(); out=associate_fork(d,process_frame(d,CFG),raw,CFG)
    out["hand_orientation"][0]=out["tool_orientation"][0]
    out["hand_xyz"][0]=out["tool_position"][0]-CFG.legacy_hand_offset_m*out["tool_orientation"][0,:,1]
    out=associate_fork(d,out,raw,CFG)
    assert out["tool_orientation_consistency_rad"][0] < 1e-12
    assert out["tool_derived_hand_consistency_m"][0] < 1e-12
