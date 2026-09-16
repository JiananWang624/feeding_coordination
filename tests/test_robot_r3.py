import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from feeding_coordination import robot_r3 as r
from feeding_coordination.robot_r1 import LOCAL_OFFSETS_RAD

def test_exactly_three_offsets(): assert r.OFFSETS==("psi0_minus_025","psi0_nominal","psi0_plus_025") and np.array_equal(r.OFFSET_VALUES,[-.25,0.,.25])
def test_delta_psi_unchanged_across_conditions():
    d=np.array([0.,.1,-.2]); shifted=[r.shift_trajectory(.3,o,d) for o in r.OFFSET_VALUES]
    assert all(np.allclose(x-x[0],d,atol=1e-12,rtol=0.) for x in shifted)
def test_every_strategy_shares_first_target():
    p={s:{"strategy_psi":np.array([.2]),"solver_status":np.array(["JOINT_LIMIT"]),"q":np.full((1,7),np.nan)} for s in r.STRATEGIES}
    assert r.verify_shared_initial(p,0)["first_target_psi_equal"]
def test_initial_status_is_strategy_independent():
    p={s:{"strategy_psi":np.array([.2]),"solver_status":np.array(["JOINT_LIMIT"]),"q":np.full((1,7),np.nan)} for s in r.STRATEGIES}; p["B2"]["solver_status"][0]="NO_VALID_BRANCH"
    with pytest.raises(RuntimeError,match="status"): r.verify_shared_initial(p,0)
def test_no_external_q0_is_injected(): assert "q0" not in inspect.signature(r.solve_stateful_run).parameters
def test_tool_trajectory_is_bit_identical():
    p=np.arange(6.).reshape(2,3); q=np.repeat(np.eye(3)[None],2,axis=0); assert r.verify_tool_identity(p,q,(p.copy(),q.copy()))
def test_robot_smooth_settings_unchanged(): assert np.array_equal(r.LOCAL_OFFSETS_RAD,LOCAL_OFFSETS_RAD) and "robot_smooth_run" in inspect.getsource(r.process_robot_r3)
def test_continuity_threshold_exactly_point_five(): assert r.CONTINUITY_THRESHOLD_RAD==.5
def test_humangt_shift_preserves_original_delta():
    d=np.array([0.,.25,.4]); x=r.shift_trajectory(-.3,.25,d); assert np.allclose(x-x[0],d)
    assert 'for strategy in ("HumanGT","B0"' in inspect.getsource(r.process_robot_r3)
def test_bootstrap_resamples_parent_bites():
    d=pd.DataFrame({"parent_bite_id":["a","a","b"],"m":[1.,3.,10.]}); mean,_,_,units,status=r._bootstrap(d,"m"); assert (mean,units,status)==(6.,2,"OK")
def test_no_calibration_retiming_or_collision_implementation():
    source=inspect.getsource(r); assert "self_collision_capability_audit(" not in source and "retiming_readiness_audit(" not in source and "process_phase3" not in source
