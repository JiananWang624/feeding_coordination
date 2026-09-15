import inspect
import numpy as np
import pandas as pd
from feeding_coordination import robot_r2 as r

def test_r2_never_calls_exact_sew(): assert "solve_exact_sew" not in inspect.getsource(r)
def test_r1_stored_q_read_only():
    q=np.zeros((1,7)); r.freeze_arrays({"q":q})
    try: q[0,0]=1
    except ValueError: return
    assert False, "frozen q accepted a write"
def test_frozen_sew_geometry_used(): assert "sew_points" in inspect.getsource(r._strategy_bite)
def test_humangt_pairwise_common_success():
    assert r.pair_clean([1,1],[1,0],[0,0],[0,0],[1,2],[0,0])[0].tolist()==[True,False]
def test_derivatives_pairwise_clean_intervals():
    _,_,s=r.pair_clean([1,1,1],[1,1,1],[0,0,0],[0,1,0],[1,2,3],[0,0,0]); assert [x.tolist() for x in s]==[[0],[1,2]]
def test_q_differences_are_wrapped(): assert np.isclose(abs(r.wrap(3.13-(-3.13))),.0231853,atol=1e-4)
def test_arm_plane_angle_stable(): assert np.isclose(r.arm_plane_angle(np.array([1,0,0]),np.array([1,0,0])),0)
def test_jacobian_aligned_pinch_site(): assert "pinch_site" in inspect.getsource(r.jacobian_metrics)
def test_jacobians_separate(): assert "mj_jacSite" in inspect.getsource(r.jacobian_metrics) and "one(jp)" in inspect.getsource(r.jacobian_metrics)
def test_bootstrap_resamples_parent_bites():
    rows=pd.DataFrame({"parent_bite_id":["a","a","b"],"metric":[1.,3.,10.]})
    difference,_,_,units,status=r._bootstrap(rows,"metric",20260915,2000)
    assert units==2 and status=="OK" and difference==6.
def test_collision_only_validated_capability():
    audit=r.self_collision_capability_audit()
    assert audit["status"]=="SELF_COLLISION_METRIC_UNAVAILABLE_MODEL_NOT_VALIDATED"
    assert audit["evidence"]["collision_meshes"]>0 and audit["evidence"]["explicit_contact_pairs"]==0
def test_no_physical_human_clearance_metric():
    audit=r.retiming_readiness_audit(__import__("pathlib").Path(__file__).parents[1])
    assert audit["status"]=="RETIMING_LIMITS_NOT_YET_FROZEN"
    assert "human/environment clearance" in inspect.getsource(r.process_robot_r2)
