import json
from pathlib import Path

import numpy as np

from feeding_coordination.phase3 import Phase3Config, build_dataset
from feeding_coordination.phase32 import (Phase32Config, STRONG_LOCAL_ORDER, add_baseline_features,
    choose_alpha, fit_predict, phase31_secondary_control, process_phase32, psi_variation,
    r0_predict, save_final_model)


def _arrays(n=20):
    t=np.arange(n)/100; R=np.tile(np.eye(3),(n,1,1))
    return {"time":t,"motive_frame":np.arange(n),"tool_position":np.c_[t,t*0,t*0],"tool_orientation":R,
      "tool_pose_valid":np.ones(n,bool),"psi_valid":np.ones(n,bool),"psi_unwrapped":.2*t,
      "plate_position":np.tile([.2,.3,.4],(n,1)),"plate_position_valid":np.ones(n,bool)}

def _data(tmp_path, n=4):
    root=tmp_path/"phase1"; (root/"demos").mkdir(parents=True); records=[]
    for i in range(n):
        take=f"trial_{i:04d}"; file=f"demos/{take}.npz"; np.savez_compressed(root/file,**_arrays())
        records.append({"record_id":take,"file":file,"source_take":take,"parent_bite_id":f"bite_{i}","phase":"transfer"})
    (root/"manifest.json").write_text(json.dumps({"schema_version":"phase1-v1.5","records":records}))
    return add_baseline_features(build_dataset(root,Phase3Config())[0]),root

def test_r0_exactly_reproduces_psi0(tmp_path):
    data,_=_data(tmp_path); assert np.array_equal(r0_predict(data),data.psi0.to_numpy(float))

def test_r1_training_folds_only(tmp_path):
    data,_=_data(tmp_path); held="trial_0000"; trace={"current_outer":held}; choose_alpha(data[data["take"] != held],"R1_phase_only",(.1,),trace)
    assert trace["inner_selection_events"] and all(held not in e["fit_takes"] and held not in e["valid_takes"] for e in trace["inner_selection_events"])

def test_final_l3_feature_ordering_matches_contract(tmp_path):
    data,root=_data(tmp_path); output=tmp_path/"model"; output.mkdir(); c=save_final_model(data,.1,output,root/"manifest.json")
    assert c["feature_order"] == list(STRONG_LOCAL_ORDER) and c["feature_count"] == 19 and all(len(x)==19 for x in data.StrongLocal)
    assert "psi0" not in c["feature_order"] and c["delta_prediction_replay_max_abs_error"] <= 1e-12

def test_outer_held_take_never_affects_alpha_selection(tmp_path):
    data,_=_data(tmp_path); held="trial_0001"; trace={"current_outer":held}; choose_alpha(data[data["take"] != held],"StrongLocal",(.1,),trace)
    assert all(held not in e["fit_takes"] and held not in e["valid_takes"] for e in trace["inner_selection_events"])

def test_saved_model_reproduces_in_memory_predictions(tmp_path):
    data,root=_data(tmp_path); output=tmp_path/"model"; output.mkdir(); save_final_model(data,.1,output,root/"manifest.json")
    z=np.load(output/"model.npz"); X=np.vstack(data.StrongLocal); replay=(X-z["mean"])/z["scale"] @ z["coef"]+z["intercept"]
    pred=fit_predict(data,data,"StrongLocal",.1)-data.psi0.to_numpy(float); assert np.max(np.abs(replay-pred)) <= 1e-12

def test_robot_phase2_information_is_never_used(tmp_path):
    data,root=_data(tmp_path); path=next((root/"demos").glob("*.npz")); values=_arrays(); values["phase2_status"]=np.array(["NO_VALID_BRANCH"]*20); np.savez_compressed(path,**values)
    again=add_baseline_features(build_dataset(root,Phase3Config())[0]); assert len(again)==len(data)
    assert "phase2" not in Path("src/feeding_coordination/phase32.py").read_text().lower().replace("phase 2","")

def test_phase31_secondary_control_filters_to_hstar_and_gate_a(tmp_path):
    path=tmp_path/"summary.json"; path.write_text(json.dumps({"overall":[{"model":"H_star"},{"model":"StrongLocal_star"}],"gates":{"Gate_A_star_history_over_strong_local":[{"reliably_improves":False}],"other":[1]}}))
    result=phase31_secondary_control(path)
    assert result["H_star_overall"] == [{"model":"H_star"}] and "other" not in result and result["Gate_A_star_history_over_strong_local"]

def test_variation_overall_is_parent_bite_balanced_when_a_phase_is_missing():
    import pandas as pd
    rows=[]
    for bite,phase,delta in (("a","transfer",1.),("a","withdrawal",1.),("b","withdrawal",3.)):
        for index,value in enumerate((0.,delta)):
            rows.append({"parent_bite_id":bite,"phase":phase,"record_id":f"{bite}_{phase}",
                         "segment_index":index,"frame":index,"psi_run_id":0,
                         "psi_true":value,"delta_psi":value})
    _,summary=psi_variation(pd.DataFrame(rows)); overall=summary[summary.grouping=="overall"].iloc[0]
    # Bite a contributes 1/sqrt(2); bite b contributes 3/sqrt(2).
    assert np.isclose(overall.rms_change_from_psi0,np.sqrt(2.))
    assert overall.n_parent_bites == 2 and overall.n_bite_phase_units == 3
