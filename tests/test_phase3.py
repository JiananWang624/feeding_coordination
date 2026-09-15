import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feeding_coordination.phase3 import Phase3Config, _segment_features, build_dataset, process_phase3


def _record(take, phase="transfer"):
    return {"record_id":f"{take}_{phase}","file":f"demos/{take}_{phase}.npz","source_take":take,"parent_bite_id":f"{take}_bite_001","phase":phase,"demonstrator_id":"D1" if take=="trial_0015" else "D0"}


def _arrays(n=52, transform=None, tool_valid=None):
    t=np.arange(n)/100.; p=np.c_[t,.1*t,np.zeros(n)]; R=Rotation.from_rotvec(np.c_[np.zeros(n),np.zeros(n),t]).as_matrix()
    if transform is not None:
        Q,a=transform; p=(Q@p.T).T+a; R=np.einsum("ij,njk->nik",Q,R)
    plate=np.tile([.2,.3,.4],(n,1))
    if transform is not None: plate=(Q@plate.T).T+a
    return {"time":t,"motive_frame":np.arange(n),"tool_position":p,"tool_orientation":R,"tool_pose_valid":np.ones(n,bool) if tool_valid is None else tool_valid,
      "psi_valid":np.ones(n,bool),"psi_unwrapped":.2*t,"plate_position":plate,"plate_position_valid":np.ones(n,bool),
      "mouth_target_position":np.tile([99.,99.,99.],(n,1))}


def _phase1(tmp_path, takes=("trial_0012","trial_0013","trial_0014")):
    root=tmp_path/"phase1"; (root/"demos").mkdir(parents=True); recs=[]
    for take in takes:
        rec=_record(take); np.savez_compressed(root/rec["file"],**_arrays()); recs.append(rec)
    (root/"manifest.json").write_text(json.dumps({"schema_version":"phase1-v1.5","records":recs}))
    return root


def test_relative_tool_representation_is_common_se3_invariant():
    rec=_record("trial_0012"); cfg=Phase3Config(bootstrap_resamples=2)
    rows,_=_segment_features(_MemoryNpz(_arrays()),rec,cfg)
    Q=Rotation.from_euler("xyz",[.4,-.2,.3]).as_matrix(); transformed,_=_segment_features(_MemoryNpz(_arrays(transform=(Q,np.array([3.,-2.,1.])))),rec,cfg)
    for a,b in zip(rows,transformed):
        for model in ("M1_pose","M2_pose_velocity","M3_strong_local","M4_history","M5_future","M6_full_plan"): assert np.allclose(a[model],b[model])


class _MemoryNpz(dict):
    files=()
    def __getitem__(self,k): return super().__getitem__(k)


def test_parent_bites_never_cross_outer_train_test(tmp_path):
    data,_=build_dataset(_phase1(tmp_path),Phase3Config())
    for held in data["take"].unique():
        assert not set(data[data["take"]==held].parent_bite_id) & set(data[data["take"]!=held].parent_bite_id)


def test_outer_held_take_never_enters_scaler_or_inner_selection(tmp_path):
    root=_phase1(tmp_path); trace={}; out=tmp_path/"out"
    process_phase3(root,out,Phase3Config(alphas=(.1,),bootstrap_resamples=2),trace=trace)
    assert trace["fit_events"]
    assert all(e["outer_held"] not in e["scaler_takes"] and e["outer_held"] not in e["fit_takes"] for e in trace["fit_events"])


def test_history_is_past_only_with_start_padding():
    rows,_=_segment_features(_MemoryNpz(_arrays()),_record("trial_0012"),Phase3Config())
    first=rows[0]; assert np.allclose(first["M4_history"][20:26],first["M1_pose"]) and first["M4_history"][26] == 0
    later=rows[30]
    # At 0.30s, 0.10s history is measured state at 0.20s, not a future state.
    assert np.allclose(later["M4_history"][20:26],rows[20]["M1_pose"]) and later["M4_history"][26] == 1
    gap=_arrays(); gap["motive_frame"][20:] += 3
    gap_rows,_=_segment_features(_MemoryNpz(gap),_record("trial_0012"),Phase3Config())
    # The post-gap first row pads from its own tool run, never from pre-gap U.
    assert gap_rows[20]["M4_history"][26] == 0 and np.allclose(gap_rows[20]["M4_history"][20:26],gap_rows[20]["M1_pose"])


def test_future_and_full_plan_are_u_only_not_psi_or_mouth_target():
    rec=_record("trial_0012"); cfg=Phase3Config(); a=_arrays(); b=_arrays(); b["psi_unwrapped"] = np.linspace(-5,7,len(b["time"])); b["mouth_target_position"][:] = -1234
    ra,_=_segment_features(_MemoryNpz(a),rec,cfg); rb,_=_segment_features(_MemoryNpz(b),rec,cfg)
    for x,y in zip(ra,rb): assert np.allclose(x["M5_future"][20:],y["M5_future"][20:]) and np.allclose(x["M6_full_plan"][20:],y["M6_full_plan"][20:])


def test_phase2_status_is_not_used_for_phase3_eligibility(tmp_path):
    root=_phase1(tmp_path,("trial_0012",)); path=root/"demos/trial_0012_transfer.npz"; values=_arrays(); values["phase2_status"]=np.array(["NO_VALID_BRANCH"]*52)
    np.savez_compressed(path,**values)
    data,diag=build_dataset(root,Phase3Config())
    assert len(data)==52 and diag["valid_target_frames"]==52
