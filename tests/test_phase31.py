import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feeding_coordination.phase3 import Phase3Config, _segment_features, build_dataset
from feeding_coordination.phase31 import (LOCAL_MODELS, Phase31Config, _star_features,
                                           add_local_candidates, paired_bootstrap, select_strong_local,
                                           star_gates)


class _MemoryNpz(dict):
    files=()
    def __getitem__(self,k): return super().__getitem__(k)


def _record(take):
    return {"record_id":f"{take}_transfer","file":f"demos/{take}.npz","source_take":take,"parent_bite_id":f"{take}_bite","phase":"transfer"}


def _arrays(n=52):
    t=np.arange(n)/100; p=np.c_[t,.1*t,np.zeros(n)]; R=Rotation.from_rotvec(np.c_[np.zeros(n),np.zeros(n),t]).as_matrix()
    return {"time":t,"motive_frame":np.arange(n),"tool_position":p,"tool_orientation":R,"tool_pose_valid":np.ones(n,bool),"psi_valid":np.ones(n,bool),"psi_unwrapped":.2*t,"plate_position":np.tile([.2,.3,.4],(n,1)),"plate_position_valid":np.ones(n,bool),"mouth_target_position":np.full((n,3),123.)}


def _phase1(tmp_path, n=4):
    root=tmp_path/"phase1"; (root/"demos").mkdir(parents=True); recs=[]
    for i in range(n):
        rec=_record(f"trial_{i:04d}"); np.savez_compressed(root/rec["file"],**_arrays()); recs.append(rec)
    (root/"manifest.json").write_text(json.dumps({"schema_version":"phase1-v1.5","records":recs})); return root


def test_l0_to_l4_have_exact_pre_registered_blocks():
    rows,_=_segment_features(_MemoryNpz(_arrays()),_record("trial_0000"),Phase3Config())
    data=add_local_candidates(__import__("pandas").DataFrame(rows)); x=data.iloc[0]
    assert tuple(LOCAL_MODELS)==("L0_kinematic","L1_temporal","L2_temporal_psi0","L3_temporal_plate","L4_current_m3")
    assert [len(x[m]) for m in LOCAL_MODELS]==[12,15,16,19,20]
    assert np.allclose(x.L0_kinematic,x.M2_pose_velocity)
    assert np.allclose(x.L1_temporal,x.M3_strong_local[:15])
    assert np.allclose(x.L2_temporal_psi0,np.r_[x.M3_strong_local[:15],x.M3_strong_local[19]])
    assert np.allclose(x.L3_temporal_plate,x.M3_strong_local[:19]) and np.allclose(x.L4_current_m3,x.M3_strong_local)


def test_outer_held_take_cannot_enter_local_selection(tmp_path):
    data=add_local_candidates(build_dataset(_phase1(tmp_path),Phase3Config())[0]); trace={"current_outer":"trial_0000"}
    select_strong_local(data[data["take"]!="trial_0000"],(.1,),trace)
    assert trace["inner_selection_events"]
    assert all(e["outer_held"] not in e["fit_takes"] and e["outer_held"] not in e["valid_takes"] for e in trace["inner_selection_events"])


def test_inner_validation_jointly_scores_every_structure_and_alpha(tmp_path):
    data=add_local_candidates(build_dataset(_phase1(tmp_path),Phase3Config())[0]); train=data[data["take"]!="trial_0000"]
    model,alpha,scores=select_strong_local(train,(.01,.1),{})
    assert model in LOCAL_MODELS and alpha in (.01,.1)
    assert len(scores)==len(LOCAL_MODELS)*2 and set(scores.model)==set(LOCAL_MODELS) and set(scores.alpha)=={.01,.1}


def test_hfp_reuse_the_same_selected_base_and_only_extend_suffix():
    rows,_=_segment_features(_MemoryNpz(_arrays()),_record("trial_0000"),Phase3Config()); data=add_local_candidates(__import__("pandas").DataFrame(rows))
    result=_star_features(data,"L2_temporal_psi0")
    base=result.iloc[0].L2_temporal_psi0
    for model,old in (("H_star","M4_history"),("F_star","M5_future"),("P_star","M6_full_plan")):
        assert np.allclose(result.iloc[0][model][:len(base)],base)
        assert np.allclose(result.iloc[0][model][len(base):],result.iloc[0][old][20:])


def test_phase2_feasibility_is_never_a_phase31_input(tmp_path):
    root=_phase1(tmp_path); path=next((root/"demos").glob("*.npz")); values=_arrays(); values["phase2_status"]=np.array(["NO_VALID_BRANCH"]*52); np.savez_compressed(path,**values)
    data,_=build_dataset(root,Phase3Config()); assert len(data)==4*52
    text=Path("src/feeding_coordination/phase31.py").read_text(); assert "phase2" not in text.lower().replace("phase 2","")


def test_future_and_plan_extensions_remain_u_only_when_base_changes():
    rec=_record("trial_0000"); a,b=_arrays(),_arrays(); b["psi_unwrapped"]=np.linspace(-4,4,52); b["mouth_target_position"][:]=-99
    da=add_local_candidates(__import__("pandas").DataFrame(_segment_features(_MemoryNpz(a),rec,Phase3Config())[0])); db=add_local_candidates(__import__("pandas").DataFrame(_segment_features(_MemoryNpz(b),rec,Phase3Config())[0]))
    aa,bb=_star_features(da,"L1_temporal"),_star_features(db,"L1_temporal")
    for x,y in zip(aa.itertuples(),bb.itertuples()):
        assert np.allclose(x.F_star[15:],y.F_star[15:]) and np.allclose(x.P_star[15:],y.P_star[15:])


def test_bootstrap_rmse_uses_bite_mse_then_global_square_root():
    rows=[]
    for bite, candidate, baseline in (("a",3.,2.),("b",1.,2.)):
        for model,error in (("H_star",candidate),("StrongLocal_star",baseline),("F_star",candidate),("P_star",candidate)):
            rows.append({"model":model,"parent_bite_id":bite,"record_id":bite,"segment_index":0,"circular_error":error})
    result=paired_bootstrap(__import__("pandas").DataFrame(rows),Phase31Config(bootstrap_resamples=4,bootstrap_seed=1))[0]
    assert np.isclose(result["rmse_difference_rad"],np.sqrt((9+1)/2)-2)
    assert not np.isclose(result["rmse_difference_rad"],((3+1)/2-2))


def test_star_gates_are_machine_readable_and_use_ci_upper_bound():
    comparisons=[("H_star vs StrongLocal_star",[-.2,-.01]),("F_star vs StrongLocal_star",[-.1,.1]),("P_star vs StrongLocal_star",[-.2,-.1]),("F_star vs H_star",[-.2,-.01]),("P_star vs H_star",[-.2,-.01]),("P_star vs F_star",[0,.2])]
    boot=[{"comparison":name,"mae_difference_rad":-.1,"mae_difference_ci95":ci} for name,ci in comparisons]
    gates=star_gates(boot)
    assert gates["Gate_A_star_history_over_strong_local"][0]["reliably_improves"]
    assert not gates["Gate_B_star_future_over_strong_local_or_history"][0]["reliably_improves"]
    assert gates["Gate_B_star_future_over_strong_local_or_history"][1]["reliably_improves"]
    assert gates["Gate_C_star_plan_over_strong_local_history_or_future"][0]["reliably_improves"]
    assert gates["Gate_C_star_plan_over_strong_local_history_or_future"][1]["reliably_improves"]
