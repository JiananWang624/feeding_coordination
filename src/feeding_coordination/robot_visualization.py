"""Read-only MuJoCo replay of the saved Robot R1 trajectories.

This module deliberately has no solver imports.  Its only robot computation is
forward kinematics of a stored configuration for display/parity checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import functools
from typing import Any

import numpy as np
import pandas as pd

from .robot_r0 import R_INPUT_ALIGN, STRATEGIES as R0_STRATEGIES, _rotation_error

STRATEGIES = (*R0_STRATEGIES, "RobotSmooth")
SUCCESS = "SUCCESS_EXACT"


@dataclass(frozen=True)
class ReplayRecord:
    record_id: str
    arrays: dict[str, np.ndarray]
    mounting: dict[str, Any]
    events: pd.DataFrame

    @property
    def frames(self) -> int: return len(self.arrays["motive_frame"])

    def strategy(self, name: str) -> dict[str, np.ndarray]:
        if name not in STRATEGIES: raise ValueError(f"unsupported strategy: {name}")
        prefix = f"{name}_"
        values = {key[len(prefix):]: value for key, value in self.arrays.items() if key.startswith(prefix)}
        if not values: raise ValueError(f"{self.record_id}: strategy {name} is absent")
        return values


def _root(root: Path | str | None) -> Path:
    return Path(root or Path(__file__).resolve().parents[2]).resolve()


def _load_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    r1_path = root / "outputs/robot_r1/manifest.json"; r1 = json.loads(r1_path.read_text())
    r0_path = Path(r1["robot_r0_manifest"])
    if hashlib.sha256(r0_path.read_bytes()).hexdigest() != r1["robot_r0_manifest_sha256"]:
        raise ValueError("Robot R0 manifest hash does not match Robot R1 provenance")
    return r1, json.loads(r0_path.read_text())


def _validate_record_arrays(record_id: str, arrays: dict[str, np.ndarray]) -> None:
    common_shapes = {
        "take": None, "parent_bite_id": None, "record_id": None, "phase": None,
        "original_frame": None, "motive_frame": None, "time_s": None, "run_id": None,
        "desired_U_position_base": (3,), "desired_U_rotation_base": (3, 3),
    }
    strategy_shapes = {
        "q": (7,), "solver_status": None, "strategy_psi": None,
        "continuity_violation": None, "branch_id": None, "search_branch": None,
        "actual_virtual_U_position_base": (3,), "actual_virtual_U_rotation_base": (3, 3),
    }
    if "motive_frame" not in arrays or arrays["motive_frame"].ndim != 1:
        raise ValueError(f"{record_id}: missing one-dimensional motive_frame")
    frames = len(arrays["motive_frame"])
    for key, tail in common_shapes.items():
        if key not in arrays or arrays[key].shape != (frames,) + (() if tail is None else tail):
            raise ValueError(f"{record_id}: invalid or missing R1 field {key}")
    identities = np.unique(arrays["record_id"].astype(str))
    if not np.array_equal(identities, np.asarray([record_id])):
        raise ValueError(f"{record_id}: NPZ record_id identity mismatch: {identities.tolist()}")
    for strategy in STRATEGIES:
        for suffix, tail in strategy_shapes.items():
            key = f"{strategy}_{suffix}"
            if key not in arrays or arrays[key].shape != (frames,) + (() if tail is None else tail):
                raise ValueError(f"{record_id}: invalid or missing R1 field {key}")


def load_record(record_id: str, root: Path | str | None = None) -> ReplayRecord:
    root = _root(root); r1, r0 = _load_manifest(root)
    entry = next((x for x in r1["records"] if x["record_id"] == record_id), None)
    if entry is None: raise ValueError(f"unknown Robot R1 record: {record_id}")
    with np.load(root / "outputs/robot_r1" / entry["file"], allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    _validate_record_arrays(record_id, arrays)
    take = str(arrays["take"][0]); mounting = r0["mountings"].get(take)
    if mounting is None: raise ValueError(f"{record_id}: no saved R0 mounting for take {take}")
    events = pd.read_csv(root / "outputs/robot_r1/continuity_events.csv")
    events = events.loc[events.record_id.astype(str) == record_id].copy()
    return ReplayRecord(record_id, arrays, mounting, events)


def list_records(root: Path | str | None = None, strategy: str = "B0") -> pd.DataFrame:
    root = _root(root); r1, _ = _load_manifest(root)
    rows = []
    for entry in r1["records"]:
        record = load_record(entry["record_id"], root); payload = record.strategy(strategy)
        rows.append({"record_id": record.record_id, "take": str(record.arrays["take"][0]),
                     "bite": str(record.arrays["parent_bite_id"][0]), "phase": str(record.arrays["phase"][0]),
                     "frame_count": record.frames, "successful_frames": int(np.sum(payload["solver_status"] == SUCCESS)),
                     "continuity_violations": int(np.sum(payload["continuity_violation"]))})
    return pd.DataFrame(rows)


def mount_record(record: ReplayRecord):
    """Recreate the frozen R0 take mounting and validate all saved pose facts."""
    from sew_mimic.mounting import load_humanoid_mounted_gen3
    anchor = np.asarray(record.mounting["anchor_B"], float)
    robot, data = load_humanoid_mounted_gen3(anchor, record.mounting["robot_world_offset_m"])
    base_id = int(robot.frame_body_ids[0]); joint_id = int(robot.joint_ids[0])
    observed = {"R_B_from_base": np.asarray(data.xmat[base_id], float).reshape(3, 3),
                "p_B_of_base": np.asarray(data.xpos[base_id], float), "joint1_B": np.asarray(data.xanchor[joint_id], float)}
    expected = {"R_B_from_base": np.asarray(record.mounting["R_B_from_base"], float),
                "p_B_of_base": np.asarray(record.mounting["p_B_of_base"], float), "joint1_B": anchor + np.asarray(record.mounting["robot_world_offset_m"], float)}
    for name in expected:
        if not np.allclose(observed[name], expected[name], atol=1e-12, rtol=0.):
            raise RuntimeError(f"{record.record_id}: reconstructed mounting {name} differs from R0 metadata")
    return robot, data, observed


def set_stored_q(robot: Any, data: Any, q: np.ndarray) -> None:
    import mujoco
    q = np.asarray(q, float)
    if q.shape != (7,) or not np.isfinite(q).all(): raise ValueError("stored q must be a finite 7-vector")
    for index, joint_id in enumerate(robot.joint_ids): data.qpos[int(robot.model.jnt_qposadr[int(joint_id)])] = q[index]
    mujoco.mj_forward(robot.model, data)


def display_q(record: ReplayRecord, strategy: str, index: int, neutral: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Return saved q or a viewer-only within-run hold; never changes success."""
    payload = record.strategy(strategy); q = payload["q"]; success = payload["solver_status"] == SUCCESS
    if success[index]: return q[index].copy(), False
    run = record.arrays["run_id"][index]
    previous = np.flatnonzero((record.arrays["run_id"][:index] == run) & success[:index])
    return (q[previous[-1]].copy() if len(previous) else np.zeros(7) if neutral is None else neutral.copy()), True


def actual_virtual_u(robot: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sew_mimic.common.evaluation import gen3_end_effector_pose
    position, rotation = gen3_end_effector_pose(q, robot)
    # gen3_end_effector_pose is the frozen authoritative base-frame FK API.
    # The mounted MuJoCo data is advanced above for the visual scene; R0/R1
    # persisted this base-frame pose, not a world-frame duplicate.
    return np.asarray(position, float), np.asarray(rotation, float) @ R_INPUT_ALIGN.T


def parity_check(record: ReplayRecord, strategies: tuple[str, ...] = ("B0", "StrongLocal", "HumanGT", "RobotSmooth")) -> dict[str, int]:
    robot, data, _ = mount_record(record); checked = 0
    for strategy in strategies:
        payload = record.strategy(strategy); good = np.flatnonzero(payload["solver_status"] == SUCCESS)
        for index in good[np.linspace(0, len(good) - 1, min(3, len(good)), dtype=int)] if len(good) else []:
            set_stored_q(robot, data, payload["q"][index]); p, r = actual_virtual_u(robot, payload["q"][index])
            if not (np.allclose(p, payload["actual_virtual_U_position_base"][index], atol=1e-8, rtol=0.) and
                    _rotation_error(r, payload["actual_virtual_U_rotation_base"][index]) <= 1e-8):
                raise RuntimeError(f"{record.record_id}/{strategy}/{index}: FK parity mismatch")
            checked += 1
    return {"frames_checked": checked}


def event_at(root: Path | str | None, index: int) -> dict[str, Any]:
    events = pd.read_csv(_root(root) / "outputs/robot_r1/continuity_events.csv")
    if index < 0 or index >= len(events): raise IndexError(f"continuity event index must be in [0,{len(events)-1}]")
    return events.iloc[index].to_dict()


def dry_run(record_id: str, strategy: str, root: Path | str | None = None) -> dict[str, Any]:
    root = _root(root); record = load_record(record_id, root); record.strategy(strategy)
    _phase1_arrays(record, root)
    _, _, mounting = mount_record(record)
    parity = parity_check(record)
    return {"record_id": record_id, "strategy": strategy, "frames": record.frames,
            "take": str(record.arrays["take"][0]), "phase": str(record.arrays["phase"][0]),
            "frame_identity_valid": True,
            "mounting": {name: value.tolist() for name, value in mounting.items()}, **parity}


@functools.lru_cache(maxsize=512)
def _phase1_cached(record_id: str, root_text: str) -> dict[str, np.ndarray]:
    """Load only the matching measured landmarks used for visual reference."""
    root=Path(root_text); r1, r0 = _load_manifest(root); source = next(x["source_record"] for x in r1["records"] if x["record_id"] == record_id)
    phase1 = Path(r0["phase1_manifest"]).parent
    with np.load(phase1 / source, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ("motive_frame", "shoulder_xyz", "elbow_xyz", "wrist_xyz", "plate_position", "plate_position_valid")}

def _phase1_arrays(record: ReplayRecord, root: Path) -> dict[str, np.ndarray]:
    human = _phase1_cached(record.record_id, str(root))
    if not np.array_equal(human["motive_frame"], record.arrays["motive_frame"]):
        raise RuntimeError(f"{record.record_id}: Phase 1.5/R1 frame identity mismatch")
    return human


def _sampled_segments(
    points: np.ndarray,
    valid: np.ndarray,
    motive_frame: np.ndarray,
    run_id: np.ndarray,
    maximum_points: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bound a trace while retaining explicit gaps between source runs."""
    valid = np.asarray(valid, bool) & np.isfinite(points).all(axis=1)
    runs: list[np.ndarray] = []
    start = None
    for index, ok in enumerate(valid):
        contiguous = (
            index > 0
            and motive_frame[index] == motive_frame[index - 1] + 1
            and run_id[index] == run_id[index - 1]
        )
        if ok and (start is None or not contiguous):
            if start is not None:
                runs.append(np.arange(start, index))
            start = index
        elif not ok and start is not None:
            runs.append(np.arange(start, index))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(valid)))
    runs = [run for run in runs if len(run) >= 2]
    budget = max(2, maximum_points)
    if len(runs) > budget // 2:
        runs = sorted(sorted(runs, key=len, reverse=True)[:budget // 2], key=lambda run: run[0])
    if not runs:
        return []
    budget = min(budget, sum(len(run) for run in runs))
    counts = np.full(len(runs), 2, dtype=int)
    remaining = budget - int(counts.sum())
    capacities = np.asarray([len(run) - 2 for run in runs], int)
    if remaining > 0 and capacities.sum() > 0:
        shares = remaining * capacities / capacities.sum()
        extras = np.floor(shares).astype(int)
        for slot in np.argsort(-(shares - extras))[:remaining - int(extras.sum())]:
            extras[slot] += 1
        counts += extras
    sampled = [run[np.linspace(0, len(run) - 1, count, dtype=int)] for run, count in zip(runs, counts)]
    return [(points[a], points[b]) for run in sampled for a, b in zip(run[:-1], run[1:])]


def replay_frame(record: ReplayRecord, strategy: str, index: int, root: Path | str | None = None,
                 show_human: bool = True, show_tool: bool = True, trail: bool = False, robot: Any | None = None,
                 show_elbow_trail: bool = False, show_human_elbow_trail: bool = False) -> tuple[np.ndarray, bool, Any]:
    """The sole shared q/overlay path for interactive and video replay."""
    from sew_mimic.sew import Gen3StereoSewGeometry
    from sew_mimic.visualization import AxisPrimitive, LinePrimitive, OverlayFrame, SpherePrimitive
    root = _root(root); robot = robot or mount_record(record)[0]; q, held = display_q(record, strategy, index)
    base = record.mounting; R = np.asarray(base["R_B_from_base"], float); p = np.asarray(base["p_B_of_base"], float)
    human = _phase1_arrays(record, root); to_base = lambda x: (np.asarray(x, float) - p) @ R
    spheres=[]; lines=[]; axes=[]
    if show_human and all(np.isfinite(human[k][index]).all() for k in ("shoulder_xyz","elbow_xyz","wrist_xyz")):
        s,e,w=(to_base(human[k][index]) for k in ("shoulder_xyz","elbow_xyz","wrist_xyz"))
        spheres += [SpherePrimitive("human_shoulder",s,.018,(1,.3,.1,.65)),SpherePrimitive("human_elbow",e,.018,(1,.6,.1,.65)),SpherePrimitive("human_wrist",w,.018,(1,.9,.1,.65))]
        lines += [LinePrimitive("human_upper",s,e,.004,(1,.4,.1,.65)),LinePrimitive("human_forearm",e,w,.004,(1,.7,.1,.65))]
    if bool(human["plate_position_valid"][index]) and np.isfinite(human["plate_position"][index]).all(): spheres.append(SpherePrimitive("plate",to_base(human["plate_position"][index]),.035,(.8,.8,.8,.45)))
    geometry=Gen3StereoSewGeometry.from_robot(robot); pts=geometry.sew_points(q)
    for name, point in (("robot_joint1",pts.shoulder),("robot_elbow",pts.elbow),("robot_wrist",pts.wrist)):
        spheres.append(SpherePrimitive(name,point,.014,(.15,.75,1.,1.)))
    desired_p=record.arrays["desired_U_position_base"][index]; desired_R=record.arrays["desired_U_rotation_base"][index]
    actual_p, actual_R=actual_virtual_u(robot,q)
    if show_tool:
        if np.isfinite(desired_p).all() and np.isfinite(desired_R).all():
            axes.append(AxisPrimitive("desired_virtual_U",desired_p,desired_R,.07))
            spheres.append(SpherePrimitive("desired_virtual_U_origin", desired_p, .011, (.1, 1., .2, 1.)))
        axes.append(AxisPrimitive("actual_virtual_U",actual_p,actual_R,.06))
        spheres.append(SpherePrimitive("actual_virtual_U_aligned_pinch", actual_p, .008, (.1, .45, 1., 1.)))
        desired_points = record.arrays["desired_U_position_base"]
        desired_valid = np.isfinite(desired_points).all(1)
        lines += [LinePrimitive("desired_path", a, b, .002, (.2,1.,.2,.6))
                  for a, b in _sampled_segments(desired_points, desired_valid,
                                                record.arrays["motive_frame"], record.arrays["run_id"])]
    if trail:
        payload=record.strategy(strategy); actual=np.full((record.frames,3),np.nan)
        success=payload["solver_status"] == SUCCESS
        for j in np.flatnonzero(success): actual[j]=actual_virtual_u(robot,payload["q"][j])[0]
        lines += [LinePrimitive("actual_tool_trail",a,b,.002,(.1,.7,1.,.6))
                  for a,b in _sampled_segments(actual,success,record.arrays["motive_frame"],record.arrays["run_id"])]
    if show_elbow_trail:
        payload=record.strategy(strategy); elbows=np.full((record.frames,3),np.nan)
        success=(payload["solver_status"] == SUCCESS) & (np.arange(record.frames) <= index)
        for j in np.flatnonzero(success): elbows[j]=geometry.sew_points(payload["q"][j]).elbow
        lines += [LinePrimitive("robot_elbow_trail",a,b,.002,(.25,.55,1.,.6))
                  for a,b in _sampled_segments(elbows,success,record.arrays["motive_frame"],record.arrays["run_id"])]
    if show_human_elbow_trail:
        elbows=np.asarray([to_base(point) for point in human["elbow_xyz"]]); valid=np.isfinite(elbows).all(1) & (np.arange(record.frames) <= index)
        lines += [LinePrimitive("human_elbow_trail",a,b,.002,(1.,.6,.1,.5))
                  for a,b in _sampled_segments(elbows,valid,record.arrays["motive_frame"],record.arrays["run_id"])]
    if held and np.isfinite(desired_p).all():
        spheres.append(SpherePrimitive("ik_failure_desired_U",desired_p,.030,(1.,0.,1.,1.)))
    if bool(record.strategy(strategy)["continuity_violation"][index]):
        spheres.append(SpherePrimitive("continuity_violation",actual_p,.032,(1.,0.,0.,1.)))
    return q, held, OverlayFrame(tuple(spheres),tuple(lines),tuple(axes))


def append_overlay_to_scene(scene: Any, overlay: Any) -> None:
    """Append overlays to a renderer scene without deleting its model geoms."""
    from sew_mimic.visualization import render_overlay_into_scene
    start=int(scene.ngeom)
    class Tail:
        def __init__(self): self.geoms=scene.geoms[start:]; self.ngeom=0
    tail=Tail(); render_overlay_into_scene(tail,overlay); scene.ngeom=start+tail.ngeom


def configure_camera(camera: Any, record: ReplayRecord, robot: Any, data: Any) -> None:
    """A free camera covering mounted arm and the complete task path."""
    base_id=int(robot.frame_body_ids[0]); R=np.asarray(data.xmat[base_id],float).reshape(3,3); p=np.asarray(data.xpos[base_id],float)
    path=np.asarray(record.arrays["desired_U_position_base"],float); world=(R @ path[np.isfinite(path).all(1)].T).T+p if np.isfinite(path).any() else p[None]
    joints=np.asarray(data.xanchor[robot.joint_ids],float); points=np.vstack((world,joints,p[None])); lo,hi=points.min(0),points.max(0)
    camera.lookat[:]=(lo+hi)/2; camera.distance=max(.8,1.55*float(np.linalg.norm(hi-lo))); camera.azimuth=135.; camera.elevation=-25.


def replay_plan(record: ReplayRecord, strategy: str) -> list[float]:
    """Original inter-frame durations; shared by interactive and exported video."""
    times=np.asarray(record.arrays["time_s"],float); return [max(0.,float(b-a)) if np.isfinite(a*b) else 0. for a,b in zip(times[:-1],times[1:])]+[0.]


def interactive_replay(record: ReplayRecord, strategy: str, root: Path | str | None = None, speed: float = 1., show_elbow_trail: bool = False, show_tool_trail: bool = False, start_index: int = 0) -> None:
    import mujoco.viewer, time
    from sew_mimic.visualization import overlay_base_to_world, render_overlay_into_scene
    robot,data,_=mount_record(record); root=_root(root); state={"i":max(0,min(record.frames-1,start_index)),"paused":False,"strategy":strategy,"human":True,"tool":True}
    def key(code: int) -> None:
        c=chr(code).lower() if 0 <= code < 128 else ""
        if c==" ": state["paused"]=not state["paused"]
        elif c=="j": state["i"]=max(0,state["i"]-1)
        elif c=="l": state["i"]=min(record.frames-1,state["i"]+1)
        elif c=="r": state["i"]=0
        elif c=="h": state["human"]=not state["human"]
        elif c=="t": state["tool"]=not state["tool"]
        elif c=="[": state["strategy"]=STRATEGIES[(STRATEGIES.index(state["strategy"])-1)%len(STRATEGIES)]
        elif c=="]": state["strategy"]=STRATEGIES[(STRATEGIES.index(state["strategy"])+1)%len(STRATEGIES)]
        elif c=="e":
            events=record.events.loc[record.events.strategy.astype(str)==state["strategy"]]
            later=events.loc[events.current_motive_frame.to_numpy(int)>record.arrays["motive_frame"][state["i"]]]
            if len(later): state["i"]=int(np.flatnonzero(record.arrays["motive_frame"]==later.iloc[0].current_motive_frame)[0]); print(later.iloc[0].to_dict())
    with mujoco.viewer.launch_passive(robot.model,data,key_callback=key) as viewer:
        configure_camera(viewer.cam,record,robot,data); prior=(-1, "")
        while viewer.is_running():
            started=time.perf_counter(); i=state["i"]; q,held,overlay=replay_frame(record,state["strategy"],i,root,state["human"],state["tool"],state["tool"] and show_tool_trail,robot,show_elbow_trail,state["human"] and show_elbow_trail); set_stored_q(robot,data,q)
            base_id=int(robot.frame_body_ids[0]); world=overlay_base_to_world(overlay,data.xmat[base_id].reshape(3,3),data.xpos[base_id]); render_overlay_into_scene(viewer.user_scn,world)
            current=(i,state["strategy"])
            if held and current != prior:
                status=record.strategy(state["strategy"])["solver_status"][i]
                print(f"IK FAILURE — VIEWER HOLD ONLY record={record.record_id} frame={record.arrays['motive_frame'][i]} strategy={state['strategy']} solver_status={status}")
            if current != prior and bool(record.strategy(state["strategy"])["continuity_violation"][i]):
                print(record.events.loc[(record.events.strategy.astype(str)==state["strategy"]) & (record.events.current_motive_frame==record.arrays["motive_frame"][i])].to_dict("records"))
            prior=current
            viewer.sync()
            delay=(replay_plan(record,state["strategy"])[i]/speed if not state["paused"] else .02)-(time.perf_counter()-started)
            if delay>0: time.sleep(delay)
            if not state["paused"]: state["i"]=(i+1)%record.frames


def export_video(record: ReplayRecord, strategy: str, output: Path, root: Path | str | None = None, speed: float = 1., show_elbow_trail: bool = False, show_tool_trail: bool = False) -> None:
    """Raw-RGB ffmpeg export using the same replay frame and timing path as the GUI."""
    import subprocess
    import mujoco
    from sew_mimic.visualization import overlay_base_to_world
    root=_root(root); robot,data,_=mount_record(record); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    durations=np.asarray(replay_plan(record,strategy)[:-1],float)
    if not len(durations) or not np.isfinite(durations).all() or np.any(durations <= 0.) or not np.allclose(durations,durations[0],atol=1e-12,rtol=0.):
        raise ValueError("video export requires uniform positive recorded timestamps; no retiming is permitted")
    fps=float(speed/durations[0]); renderer=mujoco.Renderer(robot.model,480,640); camera=mujoco.MjvCamera(); configure_camera(camera,record,robot,data)
    command=["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r",f"{fps:.12g}","-i","-","-pix_fmt","yuv420p",str(output.resolve())]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(record.frames):
            q,_,overlay=replay_frame(record,strategy,i,root,True,True,show_tool_trail,robot,show_elbow_trail,show_elbow_trail); set_stored_q(robot,data,q); renderer.update_scene(data,camera=camera)
            base_id=int(robot.frame_body_ids[0]); append_overlay_to_scene(renderer.scene,overlay_base_to_world(overlay,data.xmat[base_id].reshape(3,3),data.xpos[base_id]))
            assert process.stdin is not None; process.stdin.write(renderer.render().tobytes())
    finally:
        if process.stdin: process.stdin.close()
        renderer.close()
    if process.wait() != 0: raise RuntimeError("ffmpeg video encoding failed")
