from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

from feeding_coordination.phase2 import SUCCESS_EXACT
from feeding_coordination.robot_r1 import (CONTINUITY_THRESHOLD_RAD, LOCAL_OFFSETS_RAD, clean_motion,
    PRIMARY_PAIRS, _pair_clean_metrics, comparison_pairs, continuity_labels, paired_bootstrap,
    robot_smooth_run)


class Result:
    def __init__(self, q=None, status=SUCCESS_EXACT):
        self.q = q; self.status = SimpleNamespace(value=status)
        self.diagnostics = SimpleNamespace(branch_id="b", solve_time_ms=1., to_dict=lambda: {"metadata": {}})


def inputs(n=2): return np.zeros((n,3)), np.repeat(np.eye(3)[None],n,axis=0)
def ok_solver(calls, predicate=lambda psi: True):
    def solve(p,r,psi,qprev):
        calls.append((p.copy(),r.copy(),psi,qprev.copy()))
        return Result(qprev + np.array([psi,0,0,0,0,0,0]) if predicate(psi) else None,
                      SUCCESS_EXACT if predicate(psi) else "JOINT_LIMIT")
    return solve


def test_01_threshold_is_exactly_half_rad(): assert CONTINUITY_THRESHOLD_RAD == .5

def test_02_derivatives_do_not_cross_violation():
    q=np.zeros((4,7)); q[1,0]=.1; q[2,0]=1.; q[3,0]=1.1
    x=clean_motion(q,np.arange(4.),np.arange(4),np.zeros(4,int),np.ones(4,bool),np.array([0,0,1,0],bool))
    assert np.isnan(x["clean_joint_velocity_rad_s"][2]).all() and np.isfinite(x["clean_joint_velocity_rad_s"][3]).all()

def test_03_all_candidates_receive_same_previous_q():
    calls=[]; p,r=inputs(); robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,ok_solver(calls),lambda q:1.)
    assert len(calls)==9 and all(np.array_equal(c[3],np.zeros(7)) for c in calls)

def test_04_robot_smooth_only_passes_current_pose_and_previous_state():
    calls=[]; p,r=inputs(); p[1]=[3,4,5]; robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,ok_solver(calls),lambda q:1.)
    assert all(np.array_equal(c[0],p[1]) and np.array_equal(c[1],r[1]) for c in calls)

def test_05_first_target_is_r0_canonical_state():
    p,r=inputs(); result=robot_smooth_run(p,r,.2,np.arange(7.),SUCCESS_EXACT,ok_solver([]),lambda q:1.)
    assert result["strategy_psi"][0] == .2 and np.array_equal(result["q"][0],np.arange(7.))

def test_06_local_order_is_fixed_and_deterministic():
    calls=[]; p,r=inputs(); result=robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,ok_solver(calls),lambda q:1.)
    # All nine are submitted; deterministic lexicographic tie-breaking selects offset zero.
    assert len(calls) == len(LOCAL_OFFSETS_RAD) and result["strategy_psi"][1] == 0.

def test_07_global_recovery_uses_32_uniform_candidates():
    calls=[]; p,r=inputs(); solver=ok_solver(calls,lambda psi: psi < -3.0)
    result=robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,solver,lambda q:1.)
    assert result["search_mode"][1] == "GLOBAL_RECOVERY" and result["candidates_evaluated"][1] == 41
    assert len(calls) == 41
    assert np.allclose(sorted(call[2] for call in calls[9:]), np.linspace(-np.pi,np.pi,32,endpoint=False))

def test_08_selection_is_lexicographic_margin_then_psi():
    calls=[]; p,r=inputs();
    result=robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,ok_solver(calls,lambda psi: psi != 0.),lambda q: 2. if q[0] > 0 else 1.)
    assert np.isclose(result["strategy_psi"][1],.05)

def test_09_failed_global_stays_explicit_and_terminates_run():
    p,r=inputs(3); result=robot_smooth_run(p,r,0.,np.zeros(7),SUCCESS_EXACT,ok_solver([],lambda psi:False),lambda q:1.)
    assert result["solver_status"][1] == "INPUT_NOT_SOLVED" and result["evaluation_status"][1] == "NO_FEASIBLE_PSI" and result["selection_status"][2] == "NOT_EVALUATED_AFTER_FAILURE"

def test_10_pairwise_clean_support_excludes_violation_edge():
    qa=np.zeros((3,7));qa[1,0]=.6;qa[2,0]=.7
    qb=np.zeros((3,7));qb[1,0]=.1;qb[2,0]=.2; success=np.ones(3,bool)
    def payload(q):
        labels=continuity_labels(q,success,np.arange(3),np.zeros(3,int))
        return {"q":q,"evaluation_status":np.full(3,SUCCESS_EXACT),"joint_limit_margin_rad":np.ones(3),"branch_id":np.full(3,"b"),**labels}
    item={"raw":{"time_s":np.arange(3.),"motive_frame":np.arange(3),"run_id":np.zeros(3,int)},"payloads":{"a":payload(qa),"b":payload(qb)}}
    ma,mb,common,intervals=_pair_clean_metrics(item,"a","b",np.ones(3,bool))
    assert common.sum()==3 and intervals==1
    assert np.isnan(ma["clean_joint_velocity_rad_s"][1]).all() and np.isfinite(ma["clean_joint_velocity_rad_s"][2]).all()
    assert np.isnan(mb["clean_joint_velocity_rad_s"][1]).all() and np.isfinite(mb["clean_joint_velocity_rad_s"][2]).all()

def test_11_bootstrap_aggregates_parent_bites_not_frames():
    values=pd.DataFrame({"parent_bite_id":["a","a","b"],"difference":[1.,3.,10.]})
    answer=paired_bootstrap(values,"difference",resamples=10)
    assert answer["units"] == 2 and answer["difference"] == 6.
    assert comparison_pairs()[:len(PRIMARY_PAIRS)] == PRIMARY_PAIRS
