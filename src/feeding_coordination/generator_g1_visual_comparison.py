"""Read-only G1 visual-comparison replay.

This is deliberately a presentation layer: it loads frozen G1 arrays and the
matching Phase-1.5 human ``psi_unwrapped`` samples, and only applies forward
kinematics to already stored joint coordinates.  It contains no trajectory
generation or inverse-kinematics path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import textwrap
from typing import Any

import numpy as np

from .generator_g1_visualization import (
    G1ReplayRecord, _camera, display_stored_q, load_g1_record, mount_g1_record,
)
from .phase2 import SUCCESS_EXACT, base_transform
from .robot_r0 import _rotation_error
from .robot_r1 import wrap
from .robot_r2 import arm_plane_angle
from .robot_visualization import actual_virtual_u, append_overlay_to_scene, set_stored_q


FRAME_COUNT = 101
DISPLAY_STRATEGIES = ("B0", "StrongLocal", "RobotSmooth")


@dataclass(frozen=True)
class ComparisonRecord:
    """Frozen generated/reference G1 results plus display-only human psi."""
    generated: G1ReplayRecord
    human_psi: np.ndarray

    @property
    def reference(self) -> G1ReplayRecord:
        return G1ReplayRecord(self.generated.record_id, "measured_reference",
                              self.generated.reference, self.generated.reference,
                              self.generated.mounting)

    @property
    def frames(self) -> int:
        return FRAME_COUNT


def _root(root: Path | str | None) -> Path:
    return Path(root or Path(__file__).resolve().parents[2]).resolve()


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _resampled_human_psi(root: Path, record_id: str) -> np.ndarray:
    """Display Phase-1.5 measured psi on the frozen G1 normalized grid."""
    manifest = json.loads((root / "outputs" / "phase1" / "manifest.json").read_text())
    if manifest.get("schema_version") != "phase1-v1.5":
        raise ValueError("G1V requires frozen Phase 1.5 inputs")
    matches = [x for x in manifest["records"] if x["record_id"] == record_id]
    if len(matches) != 1:
        raise ValueError(f"{record_id}: Phase 1.5 record identity is ambiguous")
    with np.load(root / "outputs" / "phase1" / matches[0]["file"], allow_pickle=False) as z:
        time = np.asarray(z["time"], float)
        psi = np.asarray(z["psi_unwrapped"], float)
        valid = np.asarray(z["psi_valid"], bool) & np.isfinite(psi) & np.isfinite(time)
    if valid.sum() < 2:
        raise ValueError(f"{record_id}: fewer than two valid measured human psi samples")
    source_time, source_psi = time[valid], psi[valid]
    if np.any(np.diff(source_time) <= 0):
        raise ValueError(f"{record_id}: Phase 1.5 psi time is not strictly increasing")
    # Match generator_g1.measured_reference exactly: preserve the source
    # segment's normalized-time endpoints even if endpoint psi is invalid.
    if time[-1] <= time[0]:
        raise ValueError(f"{record_id}: invalid Phase 1.5 segment duration")
    normalized = (source_time - time[0]) / (time[-1] - time[0])
    return np.interp(np.linspace(0.0, 1.0, FRAME_COUNT), normalized, source_psi)


def _freeze(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        value.setflags(write=False)


def load_comparison_record(record_id: str, generator: str = "contextual_promp",
                           root: Path | str | None = None) -> ComparisonRecord:
    """Load and validate a frozen generated result and its measured reference."""
    if generator not in ("contextual_promp", "retrieval"):
        raise ValueError("comparison generator must be contextual_promp or retrieval")
    base = _root(root)
    generated = load_g1_record(record_id, generator, base)
    arrays, reference = generated.arrays, generated.reference
    for label, payload, keys in (("generated", arrays, ("normalized_phase", "time_s", "generated_U_position_base", "generated_U_rotation_base")),
                                 ("measured reference", reference, ("normalized_phase", "time_s", "measured_U_position_base", "measured_U_rotation_base"))):
        if _scalar(payload["record_id"]) != record_id:
            raise ValueError(f"{record_id}: {label} G1 identity mismatch")
        for key in keys:
            if np.asarray(payload[key]).shape[0] != FRAME_COUNT:
                raise ValueError(f"{record_id}: {label} {key} is not a 101-frame array")
    if not np.array_equal(arrays["normalized_phase"], reference["normalized_phase"]):
        raise ValueError(f"{record_id}: generated/reference normalized phase differs")
    for strategy in DISPLAY_STRATEGIES:
        payload = generated.strategy(strategy)
        if np.asarray(payload["q"]).shape != (FRAME_COUNT, 7):
            raise ValueError(f"{record_id}: {strategy} q does not have shape (101,7)")
    ref_payload = G1ReplayRecord(record_id, "measured_reference", reference, reference,
                                 generated.mounting).strategy("StrongLocal")
    if np.asarray(ref_payload["q"]).shape != (FRAME_COUNT, 7):
        raise ValueError(f"{record_id}: reference StrongLocal q does not have shape (101,7)")
    human = _resampled_human_psi(base, record_id)
    _freeze(arrays); _freeze(reference); human.setflags(write=False)
    return ComparisonRecord(generated, human)


def _success(payload: dict[str, np.ndarray], index: int) -> bool:
    return str(np.asarray(payload["solver_status"])[index]) == SUCCESS_EXACT


def _normal(points: Any) -> np.ndarray:
    return np.cross(points.wrist - points.shoulder, points.elbow - points.shoulder)


def hud_state(record: ComparisonRecord, strategy: str, index: int, geometry: Any | None = None) -> dict[str, Any]:
    """Presentation values derived from frozen arrays at one sample."""
    if strategy not in DISPLAY_STRATEGIES or not 0 <= index < record.frames:
        raise ValueError("invalid G1V strategy or sample index")
    generated, reference = record.generated, record.reference
    gp, rp = generated.strategy(strategy), reference.strategy("StrongLocal")
    psi_h = float(record.human_psi[index]); psi_r = float(reference.arrays["stronglocal_psi"][index])
    psi_g = float(generated.arrays["stronglocal_psi"][index])
    result: dict[str, Any] = {
        "sample": index, "s": float(generated.arrays["normalized_phase"][index]),
        "human_measured_psi_rad": psi_h, "reference_stronglocal_psi_rad": psi_r,
        "generated_stronglocal_psi_rad": psi_g,
        "human_minus_reference_wrapped_rad": float(wrap(psi_h - psi_r)),
        "human_minus_generated_wrapped_rad": float(wrap(psi_h - psi_g)),
        "reference_minus_generated_wrapped_rad": float(wrap(psi_r - psi_g)),
        "desired_tool_position_error_m": float(np.linalg.norm(
            generated.arrays["generated_U_position_base"][index] - reference.arrays["measured_U_position_base"][index])),
        "desired_tool_orientation_error_rad": _rotation_error(
            generated.arrays["generated_U_rotation_base"][index], reference.arrays["measured_U_rotation_base"][index]),
        "generated_success": _success(gp, index), "reference_success": _success(rp, index),
        "generated_continuity_violation": bool(gp["continuity_violation"][index]),
        "reference_continuity_violation": bool(rp["continuity_violation"][index]),
    }
    if result["generated_success"] and result["reference_success"]:
        qg, qr = np.asarray(gp["q"][index], float), np.asarray(rp["q"][index], float)
        result["robot_wrapped_q_l2_rad"] = float(np.linalg.norm(wrap(qg - qr)))
        if geometry is not None:
            pg, pr = geometry.sew_points(qg), geometry.sew_points(qr)
            result["robot_elbow_distance_m"] = float(np.linalg.norm(pg.elbow - pr.elbow))
            result["robot_arm_plane_angle_rad"] = arm_plane_angle(_normal(pg), _normal(pr))
    else:
        result.update(robot_wrapped_q_l2_rad=np.nan, robot_elbow_distance_m=np.nan,
                      robot_arm_plane_angle_rad=np.nan)
    details = []
    for name, payload, ok in (("generated", gp, result["generated_success"]),
                              ("reference", rp, result["reference_success"])):
        if not ok:
            if name == "generated":
                details.append(
                    f"PIPELINE FAILURE - VIEWER HOLD ONLY: strategy={strategy}; "
                    f"frame={index}; status={payload['solver_status'][index]}"
                )
            else:
                details.append(
                    f"REFERENCE FAILURE - VIEWER HOLD ONLY: frame={index}; "
                    f"status={payload['solver_status'][index]}"
                )
        if bool(payload["continuity_violation"][index]):
            step = np.asarray(payload["wrapped_delta_q_rad"][index], float)
            joint = int(np.nanargmax(np.abs(step))) + 1 if np.isfinite(step).any() else 0
            magnitude = float(np.nanmax(np.abs(step))) if np.isfinite(step).any() else np.nan
            branch = str(payload["branch_id"][index]) if "branch_id" in payload else ""
            search = str(payload["search_branch"][index]) if "search_branch" in payload else ""
            details.append(
                f"{name}: continuity violation; max stored step joint {joint} "
                f"({magnitude:.3f} rad); branch={branch or '-'}; search_branch={search or '-'}"
            )
    result["status_label"] = "; ".join(details) if details else "stored-q replay"
    return result


def replay_frame(record: ComparisonRecord, strategy: str, index: int, robot: Any,
                 geometry: Any | None = None, show_mouth_proxy: bool = False):
    """One authoritative read-only replay frame used by interactive and video paths."""
    from sew_mimic.visualization import AxisPrimitive, LinePrimitive, OverlayFrame, SpherePrimitive
    generated, reference = record.generated, record.reference
    qg, held_g = display_stored_q(generated, strategy, index)
    qr, held_r = display_stored_q(reference, "StrongLocal", index)
    gp, rp = generated.strategy(strategy), reference.strategy("StrongLocal")
    actual_g = actual_virtual_u(robot, qg); actual_r = actual_virtual_u(robot, qr)
    desired_g = generated.arrays["generated_U_position_base"]
    desired_gr = generated.arrays["generated_U_rotation_base"]
    desired_r = reference.arrays["measured_U_position_base"]
    desired_rr = reference.arrays["measured_U_rotation_base"]
    def segments(points: np.ndarray, color: tuple[float, ...], name: str):
        return [LinePrimitive(name, points[i - 1], points[i], .002, color) for i in range(1, len(points))]
    lines = segments(desired_g, (.15, 1., .25, .65), "generated_desired_U_path")
    lines += [LinePrimitive("measured_desired_U_path", a, b, .0035, (1., .55, .1, .9))
              for a, b in zip(desired_r[:-1], desired_r[1:])]
    lines.append(LinePrimitive("current_desired_tool_error", desired_r[index], desired_g[index], .003, (1., .2, .2, .9)))
    axes = [AxisPrimitive("generated_desired_U", desired_g[index], desired_gr[index], .065),
            AxisPrimitive("measured_desired_U", desired_r[index], desired_rr[index], .06),
            AxisPrimitive("generated_actual_U_from_stored_q", actual_g[0], actual_g[1], .05),
            AxisPrimitive("reference_actual_U_from_stored_q", actual_r[0], actual_r[1], .045)]
    R, p = np.asarray(generated.mounting["R_B_from_base"], float), np.asarray(generated.mounting["p_B_of_base"], float)
    plate = base_transform(np.asarray(generated.arrays["plate_position_B"], float)[None], R, p)[0]
    spheres = [SpherePrimitive("plate_context", plate, .035, (.8, .8, .8, .5))]
    if show_mouth_proxy:
        mouth = base_transform(np.asarray(generated.arrays["mouth_proxy_position_B"], float)[None], R, p)[0]
        spheres.append(SpherePrimitive("derived_mouth_proxy_not_measured", mouth, .025, (.3, .6, 1., .55)))
    if held_g: spheres.append(SpherePrimitive("generated_failure_viewer_hold_only", desired_g[index], .028, (1., 0., 1., 1.)))
    if held_r: spheres.append(SpherePrimitive("reference_failure_viewer_hold_only", desired_r[index], .028, (1., 0., 1., 1.)))
    return {"generated_q": qg, "reference_q": qr, "generated_held": held_g, "reference_held": held_r,
            "overlay": OverlayFrame(tuple(spheres), tuple(lines), tuple(axes)),
            "hud": hud_state(record, strategy, index, geometry), "generated_payload": gp, "reference_payload": rp}


def fk_parity(record: ComparisonRecord, strategy: str = "StrongLocal", tolerance: float = 1e-8) -> dict[str, Any]:
    """Check displayed FK against saved actual-U values; never writes results."""
    robot, data = mount_g1_record(record.generated)
    maximum_p = maximum_r = 0.0; checked = 0
    for owner, payload, replay in ((record.generated, record.generated.strategy(strategy), "generated"),
                                   (record.reference, record.reference.strategy("StrongLocal"), "reference")):
        good = np.flatnonzero(np.asarray(payload["solver_status"]).astype(str) == SUCCESS_EXACT)
        for index in good[np.linspace(0, len(good) - 1, min(5, len(good)), dtype=int)] if len(good) else []:
            q = np.asarray(payload["q"][index], float); set_stored_q(robot, data, q); pos, rot = actual_virtual_u(robot, q)
            maximum_p = max(maximum_p, float(np.linalg.norm(pos - payload["actual_virtual_U_position_base"][index])))
            maximum_r = max(maximum_r, _rotation_error(rot, payload["actual_virtual_U_rotation_base"][index])); checked += 1
    if maximum_p > tolerance or maximum_r > tolerance:
        raise RuntimeError("G1V stored-q FK parity failed")
    return {"frames_checked": checked, "max_position_error_m": maximum_p,
            "max_orientation_error_rad": maximum_r, "viewer_recomputed_q": False}


def interactive_overlay(record: ComparisonRecord, strategy: str = "StrongLocal", speed: float = 1.0,
                        show_mouth_proxy: bool = False) -> None:
    """Open a read-only overlay viewer; split composition is provided by video export.

    The same :func:`replay_frame` function drives this viewer and export, so a
    displayed failure remains a hold marker in both modes.
    """
    if speed not in (.25, .5, 1., 2.):
        raise ValueError("speed must be one of 0.25, 0.5, 1, 2")
    import time
    import mujoco.viewer
    from sew_mimic.sew import Gen3StereoSewGeometry
    from sew_mimic.visualization import overlay_base_to_world, render_overlay_into_scene
    robot, data = mount_g1_record(record.generated)
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    dt = float(np.diff(np.asarray(record.generated.arrays["time_s"], float))[0]) / speed
    state = {"index": 0, "paused": False}
    def key(code: int) -> None:
        char = chr(code).lower() if 0 <= code < 128 else ""
        if char == " ": state["paused"] = not state["paused"]
        elif char == "j": state["index"] = max(0, state["index"] - 1)
        elif char == "l": state["index"] = min(record.frames - 1, state["index"] + 1)
        elif char == "r": state["index"] = 0
    last, prior = time.monotonic(), -1
    with mujoco.viewer.launch_passive(robot.model, data, key_callback=key) as viewer:
        _camera(viewer.cam, record.generated, robot, data)
        while viewer.is_running():
            now = time.monotonic()
            if not state["paused"] and now - last >= dt:
                state["index"] = (state["index"] + 1) % record.frames; last = now
            index = state["index"]
            frame = replay_frame(record, strategy, index, robot, geometry, show_mouth_proxy)
            set_stored_q(robot, data, frame["generated_q"])
            base_id = int(robot.frame_body_ids[0])
            world = overlay_base_to_world(frame["overlay"], data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
            render_overlay_into_scene(viewer.user_scn, world)
            if index != prior:
                h = frame["hud"]
                print(
                    f"sample={index:03d} s={h['s']:.2f} "
                    f"psi_human={h['human_measured_psi_rad']:.3f} "
                    f"psi_ref={h['reference_stronglocal_psi_rad']:.3f} "
                    f"psi_generated={h['generated_stronglocal_psi_rad']:.3f} "
                    f"tool_dp={h['desired_tool_position_error_m']:.4f}m "
                    f"tool_dR={h['desired_tool_orientation_error_rad']:.3f}rad "
                    f"abs_dpsi_gen_ref={abs(h['reference_minus_generated_wrapped_rad']):.3f} "
                    f"abs_dpsi_gen_human={abs(h['human_minus_generated_wrapped_rad']):.3f} "
                    f"q_L2={h['robot_wrapped_q_l2_rad']:.3f}rad "
                    f"elbow={h['robot_elbow_distance_m']:.3f}m "
                    f"plane={h['robot_arm_plane_angle_rad']:.3f}rad | {h['status_label']}"
                )
                prior = index
            viewer.sync()
            time.sleep(.01)


def export_split_video(record: ComparisonRecord, strategy: str, output: Path, speed: float = .5,
                       show_mouth_proxy: bool = False) -> dict[str, Any]:
    """Export synchronized stored-q reference/generated renders with a psi panel."""
    if speed not in (.25, .5, 1., 2.):
        raise ValueError("speed must be one of 0.25, 0.5, 1, 2")
    import matplotlib.pyplot as plt
    import mujoco
    from sew_mimic.visualization import overlay_base_to_world
    parity = fk_parity(record, strategy); robot, data = mount_g1_record(record.generated)
    from sew_mimic.sew import Gen3StereoSewGeometry
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    time_s = np.asarray(record.generated.arrays["time_s"], float); dt = np.diff(time_s)
    if len(dt) != 100 or np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-12, rtol=0):
        raise ValueError("G1V requires uniform stored G0 provisional timing")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fps = speed / float(dt[0]); renderer = mujoco.Renderer(robot.model, 360, 480)
    camera = mujoco.MjvCamera(); _camera(camera, record.generated, robot, data)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", f"{fps:.12g}", "-i", "-", "-pix_fmt", "yuv420p", str(output.resolve())]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(record.frames):
            state = replay_frame(record, strategy, index, robot, geometry, show_mouth_proxy)
            images = []
            for q in (state["reference_q"], state["generated_q"]):
                set_stored_q(robot, data, q); renderer.update_scene(data, camera=camera)
                base_id = int(robot.frame_body_ids[0]); world = overlay_base_to_world(state["overlay"], data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
                append_overlay_to_scene(renderer.scene, world); images.append(renderer.render())
            fig, ax = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100, gridspec_kw={"height_ratios": [3, 1]})
            ax[0, 0].imshow(images[0]); ax[0, 0].set_title("MeasuredUReference / StrongLocal")
            ax[0, 1].imshow(images[1]); ax[0, 1].set_title(f"{record.generated.generator} / {strategy}")
            for one in ax[0]: one.axis("off")
            curve = ax[1, 0]; s = record.generated.arrays["normalized_phase"]
            curve.plot(s, record.human_psi, label="measured human psi", color="black")
            curve.plot(s, record.reference.arrays["stronglocal_psi"], label="reference StrongLocal psi", color="orange")
            curve.plot(s, record.generated.arrays["stronglocal_psi"], label="generated StrongLocal psi", color="green")
            curve.axvline(s[index], color="red"); curve.set_xlabel("normalized phase"); curve.set_ylabel("rad"); curve.legend(fontsize=7, ncol=3)
            ax[1, 1].axis("off"); h = state["hud"]
            status = textwrap.fill(h["status_label"], width=68)
            ax[1, 1].text(0, 1, f"{record.generated.record_id}  sample {index:03d}  s={h['s']:.2f}\npsi human={h['human_measured_psi_rad']:.3f}  ref={h['reference_stronglocal_psi_rad']:.3f}  generated={h['generated_stronglocal_psi_rad']:.3f}\n|dpsi gen-ref|={abs(h['reference_minus_generated_wrapped_rad']):.3f}  |dpsi gen-human|={abs(h['human_minus_generated_wrapped_rad']):.3f}\n|delta p|={h['desired_tool_position_error_m']:.4f} m  delta R={h['desired_tool_orientation_error_rad']:.3f} rad\nq L2={h['robot_wrapped_q_l2_rad']:.3f}  elbow={h['robot_elbow_distance_m']:.3f} m  plane={h['robot_arm_plane_angle_rad']:.3f} rad\n{status}", va="top", fontsize=8, color="crimson" if "FAILURE" in h["status_label"] else "black", clip_on=True)
            fig.subplots_adjust(left=.06, right=.98, bottom=.09, top=.94, wspace=.04, hspace=.25)
            fig.canvas.draw(); frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
            assert process.stdin is not None; process.stdin.write(frame.tobytes())
    finally:
        if process.stdin: process.stdin.close()
        renderer.close()
    if process.wait() != 0: raise RuntimeError("G1V ffmpeg export failed")
    return {**parity, "record_id": record.generated.record_id, "generator": record.generated.generator,
            "strategy": strategy, "output": str(output.resolve()), "mode": "split",
            "timing": "stored provisional G0 timing; not physically retimed"}


def export_generator_pair_video(left: ComparisonRecord, right: ComparisonRecord, output: Path,
                                strategy: str = "StrongLocal", speed: float = .5,
                                show_mouth_proxy: bool = False) -> dict[str, Any]:
    """Export two saved generated pipelines for the same record in lockstep."""
    if left.generated.record_id != right.generated.record_id:
        raise ValueError("generator-pair comparison requires the same record id")
    if speed not in (.25, .5, 1., 2.):
        raise ValueError("speed must be one of 0.25, 0.5, 1, 2")
    if not np.array_equal(left.generated.arrays["time_s"], right.generated.arrays["time_s"]):
        raise ValueError("generator-pair stored timing differs")
    import matplotlib.pyplot as plt
    import mujoco
    from sew_mimic.sew import Gen3StereoSewGeometry
    from sew_mimic.visualization import overlay_base_to_world
    parity_left, parity_right = fk_parity(left, strategy), fk_parity(right, strategy)
    robot, data = mount_g1_record(left.generated); geometry = Gen3StereoSewGeometry.from_robot(robot)
    time_s = np.asarray(left.generated.arrays["time_s"], float); dt = np.diff(time_s)
    if len(dt) != 100 or np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-12, rtol=0):
        raise ValueError("G1V requires uniform stored G0 provisional timing")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(robot.model, 360, 480); camera = mujoco.MjvCamera()
    _camera(camera, left.generated, robot, data)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x720", "-r", f"{speed / float(dt[0]):.12g}", "-i", "-",
               "-pix_fmt", "yuv420p", str(output.resolve())]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(FRAME_COUNT):
            frames = [replay_frame(item, strategy, index, robot, geometry, show_mouth_proxy)
                      for item in (left, right)]
            images = []
            for item in frames:
                set_stored_q(robot, data, item["generated_q"]); renderer.update_scene(data, camera=camera)
                base_id = int(robot.frame_body_ids[0])
                world = overlay_base_to_world(item["overlay"], data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
                append_overlay_to_scene(renderer.scene, world); images.append(renderer.render())
            fig, ax = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100,
                                   gridspec_kw={"height_ratios": [3, 1]})
            ax[0, 0].imshow(images[0]); ax[0, 0].set_title(f"{left.generated.generator} / {strategy}")
            ax[0, 1].imshow(images[1]); ax[0, 1].set_title(f"{right.generated.generator} / {strategy}")
            for one in ax[0]: one.axis("off")
            s = left.generated.arrays["normalized_phase"]; curve = ax[1, 0]
            curve.plot(s, left.human_psi, label="measured human psi", color="black")
            curve.plot(s, left.generated.arrays["stronglocal_psi"], label="contextual ProMP psi", color="green")
            curve.plot(s, right.generated.arrays["stronglocal_psi"], label="retrieval psi", color="purple")
            curve.axvline(s[index], color="red"); curve.set_xlabel("normalized phase"); curve.set_ylabel("rad")
            curve.legend(fontsize=7, ncol=3)
            left_payload, right_payload = left.generated.strategy(strategy), right.generated.strategy(strategy)
            both = _success(left_payload, index) and _success(right_payload, index)
            q_l2 = float(np.linalg.norm(wrap(left_payload["q"][index] - right_payload["q"][index]))) if both else np.nan
            dp = float(np.linalg.norm(left.generated.arrays["generated_U_position_base"][index]
                                      - right.generated.arrays["generated_U_position_base"][index]))
            dR = _rotation_error(left.generated.arrays["generated_U_rotation_base"][index],
                                 right.generated.arrays["generated_U_rotation_base"][index])
            statuses = "; ".join(x["hud"]["status_label"] for x in frames
                                 if x["hud"]["status_label"] != "stored-q replay") or "stored-q replay"
            ax[1, 1].axis("off")
            status_text = textwrap.fill(statuses, width=68)
            ax[1, 1].text(0, 1, f"{left.generated.record_id}  sample {index:03d}  s={s[index]:.2f}\n"
                               f"ProMP vs retrieval: |delta p|={dp:.4f} m  delta R={dR:.3f} rad\n"
                               f"stored-q wrapped L2={q_l2:.3f} rad\n{status_text}",
                               va="top", fontsize=8, color="crimson" if "FAILURE" in statuses else "black", clip_on=True)
            fig.subplots_adjust(left=.06, right=.98, bottom=.09, top=.94, wspace=.04, hspace=.25)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
            assert process.stdin is not None; process.stdin.write(rgb.tobytes())
    finally:
        if process.stdin: process.stdin.close()
        renderer.close()
    if process.wait() != 0:
        raise RuntimeError("G1V generator-pair ffmpeg export failed")
    return {"record_id": left.generated.record_id,
            "generators": [left.generated.generator, right.generated.generator],
            "strategy": strategy, "output": str(output.resolve()), "mode": "split",
            "left_fk_parity": parity_left, "right_fk_parity": parity_right,
            "timing": "stored provisional G0 timing; not physically retimed"}
