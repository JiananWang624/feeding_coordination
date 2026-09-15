import numpy as np
import pytest
import json
from types import SimpleNamespace

from feeding_coordination.phase2 import Phase2Config, base_rotations, base_transform, valid_runs
from feeding_coordination.phase2 import process_phase2

R_B_FROM_BASE = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def _phase1_fixture(tmp_path):
    source=tmp_path/'phase1'; (source/'demos').mkdir(parents=True)
    from sew_mimic.sew import StereoSew, StereoSewReference, project_stereo_sew_reference
    points=np.array([[0.,0.,0.],[.2,.1,0.],[.4,.1,.2]])
    reference=project_stereo_sew_reference()
    psi=StereoSew(StereoSewReference(R_B_FROM_BASE@reference.e_t,R_B_FROM_BASE@reference.e_r)).forward(*points)
    def write(name, frames, valid):
        n=len(frames); rotations=np.repeat(np.eye(3)[None],n,axis=0); arm=np.repeat(points[None],n,axis=0)
        np.savez_compressed(source/'demos'/f'{name}.npz', motive_frame=np.asarray(frames), time=np.arange(n,dtype=float), phase=np.full(n,'transfer'), shoulder_xyz=arm[:,0], elbow_xyz=arm[:,1], wrist_xyz=arm[:,2], shoulder_valid_observation=np.ones(n,bool), tool_position=np.tile([.4,.1,.2],(n,1)), tool_orientation=rotations, tool_pose_valid=np.asarray(valid), psi_wrapped=np.full(n,psi), psi_valid=np.ones(n,bool))
    write('a',[1,2,3],[True,False,True]); write('b',[4,5],[True,True])
    rec=lambda name:{'record_id':name,'source_take':'take','bite_id':1,'phase':'transfer','file':f'demos/{name}.npz'}
    (source/'manifest.json').write_text(json.dumps({'schema_version':'phase1-v1.5','config':{'R_B_from_exact_sew_base':R_B_FROM_BASE.tolist()},'records':[rec('a'),rec('b')]}))
    return source


def _fake_runtime(calls):
    def mount(anchor):
        calls['mount'].append(anchor.copy())
        return SimpleNamespace(frame_body_ids=[0]), SimpleNamespace(xmat=R_B_FROM_BASE.reshape(1,9),xpos=np.zeros((1,3)))
    class Adapter:
        def solve_trajectory(self,p,r,psi):
            calls['runs'].append((p.copy(),r.copy(),psi.copy()))
            d=SimpleNamespace(branch_id='raw',metadata={'search_branch':'slot'},solve_time_ms=None,to_dict=lambda:{'metadata':{'search_branch':'slot'}})
            return [SimpleNamespace(status=SimpleNamespace(value='NO_VALID_BRANCH'),diagnostics=d,message='kept failure',q=None) for _ in p]
    return mount,Adapter


def test_one_fixed_mounting_across_segments(tmp_path):
    source=_phase1_fixture(tmp_path); calls={'mount':[],'runs':[]}; mount,adapter=_fake_runtime(calls)
    cfg=Phase2Config('x','y',{'translation_m':[0,0,0],'rotation':np.eye(3).tolist()},{'name':'Rx(+90deg)','robot_world_offset_m':[0,.15,.2]})
    output=process_phase2(source,tmp_path/'out',cfg,mount,adapter)
    assert len(calls['mount']) == 1
    assert output['summary']['psi_frame_consistency_abs_circular_rad']['max'] < 1e-12


def test_b_to_base_position_transform():
    assert np.allclose(base_transform([[2,4,6]],np.eye(3),np.array([1,2,3])), [[1,2,3]])


def test_virtual_identity_has_no_extra_rotation_or_translation():
    R=np.eye(3); p=np.array([.2,.3,.4])
    assert np.allclose(base_transform([p],R,np.zeros(3)),[p])
    assert np.allclose(base_rotations(R[None],R),R[None])


def test_native_rotation_transform_and_psi_agree():
    from sew_mimic.sew import StereoSew, StereoSewReference, project_stereo_sew_reference
    points_B=np.array([[0.,0.,0.],[.2,.1,0.],[.4,.1,.2]])
    reference=project_stereo_sew_reference()
    psi_B=StereoSew(StereoSewReference(R_B_FROM_BASE@reference.e_t,R_B_FROM_BASE@reference.e_r)).forward(*points_B)
    psi_base=StereoSew(reference).forward(*base_transform(points_B,R_B_FROM_BASE,np.zeros(3)))
    assert abs(np.arctan2(np.sin(psi_base-psi_B),np.cos(psi_base-psi_B))) < 1e-12
    assert np.allclose(base_rotations(np.eye(3)[None],R_B_FROM_BASE)[0],R_B_FROM_BASE.T)


def test_invalid_rows_are_not_sent_to_adapter(tmp_path):
    source=_phase1_fixture(tmp_path); calls={'mount':[],'runs':[]}; mount,adapter=_fake_runtime(calls); cfg=Phase2Config('x','y',{'translation_m':[0,0,0],'rotation':np.eye(3).tolist()},{'name':'Rx(+90deg)','robot_world_offset_m':[0,.15,.2]})
    process_phase2(source,tmp_path/'out',cfg,mount,adapter)
    assert [len(x[0]) for x in calls['runs']] == [1,1,2]


def test_gap_creates_separate_run():
    assert [x.tolist() for x in valid_runs([True,True,True],[10,11,13])] == [[0,1],[2]]


def test_solver_failure_is_preserved_and_rows_map_back(tmp_path):
    source=_phase1_fixture(tmp_path); calls={'mount':[],'runs':[]}; mount,adapter=_fake_runtime(calls); cfg=Phase2Config('x','y',{'translation_m':[0,0,0],'rotation':np.eye(3).tolist()},{'name':'Rx(+90deg)','robot_world_offset_m':[0,.15,.2]})
    process_phase2(source,tmp_path/'out',cfg,mount,adapter); z=np.load(tmp_path/'out'/'results'/'a.npz')
    assert z['status'].tolist() == ['NO_VALID_BRANCH','INPUT_NOT_SOLVED','NO_VALID_BRANCH']
    assert z['original_index'].tolist() == [0,1,2] and z['motive_frame'].tolist() == [1,2,3]


def test_config_rejects_nonidentity_virtual_tool(tmp_path):
    path=tmp_path/'bad.json'; path.write_text('{"phase1_output_path":"a","output_path":"b","virtual_P_to_UR":{"translation_m":[0,0,0],"rotation":[[1,0,0],[0,1,0],[0,0,-1]]},"mounting":{"name":"Rx(+90deg)","robot_world_offset_m":[0,0.15,0.2]}}')
    with pytest.raises(ValueError,match="identity"):
        Phase2Config.load(path)
