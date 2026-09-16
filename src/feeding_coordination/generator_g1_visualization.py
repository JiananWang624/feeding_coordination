"""Read-only MuJoCo replay for stored G1 joint trajectories.

This module intentionally imports no IK or Exact-SEW solver.  Viewer holds on
failed frames are display-only and never alter saved G1 metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .phase2 import SUCCESS_EXACT, base_transform
from .robot_r0 import _rotation_error
from .robot_visualization import actual_virtual_u, append_overlay_to_scene, set_stored_q


GENERATORS = ("retrieval", "contextual_promp", "measured_reference")
STRATEGIES = ("B0", "StrongLocal", "RobotSmooth")


@dataclass(frozen=True)
class G1ReplayRecord:
    record_id: str
    generator: str
    arrays: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    mounting: dict[str, Any]

    @property
    def frames(self) -> int:
        return 101

    def strategy(self, name: str) -> dict[str, np.ndarray]:
        if self.generator == "measured_reference" and name != "StrongLocal":
            raise ValueError("MeasuredUReference supports StrongLocal only")
        if name not in STRATEGIES:
            raise ValueError(f"unsupported G1 strategy: {name}")
        prefix = f"{name}_"
        payload = {key[len(prefix):]: value for key, value in self.arrays.items() if key.startswith(prefix)}
        if not payload:
            raise ValueError(f"{self.record_id}/{self.generator}: stored strategy {name} is absent")
        return payload


def _root(root: Path | str | None) -> Path:
    return Path(root or Path(__file__).resolve().parents[2]).resolve()


def load_g1_record(record_id: str, generator: str, root: Path | str | None = None) -> G1ReplayRecord:
    if generator not in GENERATORS:
        raise ValueError(f"unsupported G1 generator: {generator}")
    root = _root(root); output = root / "outputs" / "generator_g1"
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != "generator-g1-v1":
        raise ValueError("G1 visualization requires frozen generator-g1-v1 outputs")
    path = output / "results" / generator / f"{record_id}.npz"
    if not path.is_file(): raise ValueError(f"unknown G1 record/generator: {record_id}/{generator}")
    with np.load(path, allow_pickle=False) as z: arrays = {key: z[key].copy() for key in z.files}
    reference_path = output / "results" / "measured_reference" / f"{record_id}.npz"
    with np.load(reference_path, allow_pickle=False) as z: reference = {key: z[key].copy() for key in z.files}
    if str(arrays["record_id"]) != record_id or str(arrays["generator"]).lower() not in (generator, "measureduereference"):
        raise ValueError("G1 result identity mismatch")
    r0 = json.loads((root / "outputs" / "robot_r0" / "manifest.json").read_text())
    take = str(arrays["take"]); mounting = r0["mountings"].get(take)
    if mounting is None: raise ValueError(f"{record_id}: frozen Robot R0 mounting is missing")
    return G1ReplayRecord(record_id, generator, arrays, reference, mounting)


def mount_g1_record(record: G1ReplayRecord):
    from sew_mimic.mounting import load_humanoid_mounted_gen3
    anchor = np.asarray(record.mounting["anchor_B"], float)
    robot, data = load_humanoid_mounted_gen3(anchor, record.mounting["robot_world_offset_m"])
    base_id = int(robot.frame_body_ids[0])
    observed_R = np.asarray(data.xmat[base_id], float).reshape(3, 3)
    observed_p = np.asarray(data.xpos[base_id], float)
    if not (np.allclose(observed_R, record.mounting["R_B_from_base"], atol=1e-12, rtol=0)
            and np.allclose(observed_p, record.mounting["p_B_of_base"], atol=1e-12, rtol=0)):
        raise RuntimeError("G1 viewer mounting differs from frozen Robot R0 geometry")
    return robot, data


def display_stored_q(record: G1ReplayRecord, strategy: str, index: int) -> tuple[np.ndarray, bool]:
    payload = record.strategy(strategy); success = np.asarray(payload["solver_status"]).astype(str) == SUCCESS_EXACT
    if success[index]: return np.asarray(payload["q"][index], float).copy(), False
    previous = np.flatnonzero(success[:index])
    return (np.asarray(payload["q"][previous[-1]], float).copy() if len(previous) else np.zeros(7), True)


def viewer_parity(record: G1ReplayRecord, strategy: str, tolerance: float = 1e-8) -> dict[str, Any]:
    robot, data = mount_g1_record(record); payload = record.strategy(strategy)
    success = np.flatnonzero(np.asarray(payload["solver_status"]).astype(str) == SUCCESS_EXACT)
    selected = success[np.linspace(0, len(success) - 1, min(5, len(success)), dtype=int)] if len(success) else []
    position_max = rotation_max = 0.0
    for index in selected:
        q = np.asarray(payload["q"][index], float); set_stored_q(robot, data, q)
        position, rotation = actual_virtual_u(robot, q)
        position_max = max(position_max, float(np.linalg.norm(position - payload["actual_virtual_U_position_base"][index])))
        rotation_max = max(rotation_max, _rotation_error(rotation, payload["actual_virtual_U_rotation_base"][index]))
    if position_max > tolerance or rotation_max > tolerance:
        raise RuntimeError("G1 stored-q viewer FK parity failed")
    return {"frames_checked": len(selected), "max_position_error_m": position_max,
            "max_orientation_error_rad": rotation_max, "viewer_recomputed_q": False}


def _segments(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(points, float); good = np.isfinite(points).all(1)
    return [(points[i - 1], points[i]) for i in range(1, len(points)) if good[i - 1] and good[i]]


def replay_overlay(record: G1ReplayRecord, strategy: str, index: int, robot: Any):
    from sew_mimic.visualization import AxisPrimitive, LinePrimitive, OverlayFrame, SpherePrimitive
    q, held = display_stored_q(record, strategy, index); payload = record.strategy(strategy)
    if record.generator == "measured_reference":
        desired_p = record.arrays["measured_U_position_base"]
        desired_R = record.arrays["measured_U_rotation_base"]
    else:
        desired_p = record.arrays["generated_U_position_base"]
        desired_R = record.arrays["generated_U_rotation_base"]
    measured_p = record.reference["measured_U_position_base"]
    measured_R = record.reference["measured_U_rotation_base"]
    actual_p, actual_R = actual_virtual_u(robot, q)
    lines = [LinePrimitive("generated_U_path", a, b, .0025, (.15, 1., .25, .75)) for a, b in _segments(desired_p)]
    lines += [LinePrimitive("measured_demonstration_U_path", a, b, .002, (1., .55, .1, .65)) for a, b in _segments(measured_p)]
    axes = [AxisPrimitive("generated_U", desired_p[index], desired_R[index], .07),
            AxisPrimitive("actual_realized_generated_U", actual_p, actual_R, .055),
            AxisPrimitive("measured_demonstration_U", measured_p[index], measured_R[index], .05)]
    R = np.asarray(record.mounting["R_B_from_base"], float); p = np.asarray(record.mounting["p_B_of_base"], float)
    plate = base_transform(np.asarray(record.arrays["plate_position_B"], float)[None], R, p)[0]
    spheres = [SpherePrimitive("plate_context", plate, .035, (.8, .8, .8, .5))]
    if held: spheres.append(SpherePrimitive("stored_IK_failure_viewer_hold_only", desired_p[index], .03, (1., 0., 1., 1.)))
    return q, held, OverlayFrame(tuple(spheres), tuple(lines), tuple(axes))


def _camera(camera: Any, record: G1ReplayRecord, robot: Any, data: Any) -> None:
    base_id = int(robot.frame_body_ids[0]); R = np.asarray(data.xmat[base_id], float).reshape(3, 3); p = np.asarray(data.xpos[base_id], float)
    if record.generator == "measured_reference": base_path = record.arrays["measured_U_position_base"]
    else: base_path = record.arrays["generated_U_position_base"]
    world = (R @ np.asarray(base_path).T).T + p
    points = np.vstack([world, np.asarray(data.xanchor[robot.joint_ids], float)])
    lo, hi = points.min(0), points.max(0); camera.lookat[:] = (lo + hi) / 2
    camera.distance = max(.8, 1.55 * float(np.linalg.norm(hi - lo))); camera.azimuth = 135.; camera.elevation = -25.


def export_g1_video(record: G1ReplayRecord, strategy: str, output: Path, speed: float = 1.0) -> dict[str, Any]:
    """Export stored-q replay at provisional G0 timing; no IK and no retiming."""
    import mujoco
    from sew_mimic.visualization import overlay_base_to_world
    parity = viewer_parity(record, strategy); robot, data = mount_g1_record(record)
    time_s = np.asarray(record.arrays["time_s"], float); durations = np.diff(time_s)
    if len(durations) != 100 or np.any(durations <= 0) or not np.allclose(durations, durations[0], atol=1e-12, rtol=0):
        raise ValueError("G1 video requires uniform positive stored provisional timing")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fps = float(speed / durations[0]); renderer = mujoco.Renderer(robot.model, 480, 640)
    camera = mujoco.MjvCamera(); _camera(camera, record, robot, data)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "640x480", "-r", f"{fps:.12g}", "-i", "-", "-pix_fmt", "yuv420p", str(output.resolve())]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(record.frames):
            q, _, overlay = replay_overlay(record, strategy, index, robot); set_stored_q(robot, data, q)
            renderer.update_scene(data, camera=camera); base_id = int(robot.frame_body_ids[0])
            world = overlay_base_to_world(overlay, data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
            append_overlay_to_scene(renderer.scene, world)
            assert process.stdin is not None; process.stdin.write(renderer.render().tobytes())
    finally:
        if process.stdin: process.stdin.close()
        renderer.close()
    if process.wait() != 0: raise RuntimeError("G1 ffmpeg export failed")
    return {**parity, "record_id": record.record_id, "generator": record.generator,
            "strategy": strategy, "output": str(output.resolve()),
            "timing": "stored provisional G0 timing; not physically retimed"}


def export_representative_videos(root: Path | str | None = None) -> list[dict[str, Any]]:
    root = _root(root); output = root / "outputs" / "generator_g1"
    pipeline = pd.read_csv(output / "pipeline_metrics.csv")
    def clean(generator: str, phase: str) -> str:
        rows = pipeline.loc[(pipeline.generator == generator) & (pipeline.phase == phase)
                            & pipeline.stronglocal_complete_pipeline_success.astype(bool)]
        if rows.empty: raise RuntimeError(f"no clean {generator}/{phase} G1 example for required video")
        return str(rows.sort_values(["tool_rms_position_error_m", "record_id"]).iloc[0].record_id)
    examples = json.loads((output / "representative_examples.json").read_text())
    propagation = examples["generator_error_propagation"]
    jobs = [
        (clean("contextual_promp", "transfer"), "contextual_promp", "promp_clean_transfer.mp4"),
        (clean("contextual_promp", "withdrawal"), "contextual_promp", "promp_clean_withdrawal.mp4"),
        (clean("retrieval", "transfer"), "retrieval", "retrieval_clean_example.mp4"),
        (propagation["record_id"], propagation["generator"], "generator_error_propagation.mp4"),
    ]
    reports = [export_g1_video(load_g1_record(record_id, generator, root), "StrongLocal",
                               output / "videos" / filename) for record_id, generator, filename in jobs]
    (output / "videos" / "manifest.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
    manifest_path = output / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    manifest["visualization"] = {"videos_manifest": "videos/manifest.json",
        "videos": [f"videos/{Path(report['output']).name}" for report in reports],
        "viewer_recomputes_q": False, "timing_physically_retimed": False}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return reports
