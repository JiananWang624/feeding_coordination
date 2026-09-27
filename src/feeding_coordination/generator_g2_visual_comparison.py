"""Read-only G2 endpoint-constraint trajectory visualization.

G2A/G2B persist tool trajectories only; they deliberately do not contain robot
q or an Exact-SEW realization.  This presentation layer therefore renders the
saved tool paths in the existing MuJoCo scene and keeps the G1 measured and
reference replay panels for visual context.  No G2 solver or robot IK is run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .generator_g1_visual_comparison import (
    ComparisonRecord, FRAME_COUNT, _root, _scalar,
    load_comparison_record, replay_frame,
)
from .generator_g1_visualization import _camera, mount_g1_record
from .phase2 import base_transform
from .robot_r1 import wrap
from .robot_visualization import append_overlay_to_scene, set_stored_q

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


@dataclass(frozen=True)
class G2ComparisonRecord:
    g1: ComparisonRecord
    method: str
    constrained_position_B: np.ndarray
    constrained_orientation_B: np.ndarray
    constrained_psi: np.ndarray

    @property
    def frames(self) -> int:
        return FRAME_COUNT


def _base_path(record: ComparisonRecord, position_B: np.ndarray,
               orientation_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mounting = record.generated.mounting if isinstance(record, ComparisonRecord) else record.mounting
    R = np.asarray(mounting["R_B_from_base"], float)
    p = np.asarray(mounting["p_B_of_base"], float)
    position = (R.T @ (np.asarray(position_B, float) - p).T).T
    orientation = np.einsum("ij,njk->nik", R.T, np.asarray(orientation_B, float))
    return position, orientation


def _display_psi(record: ComparisonRecord, position_B: np.ndarray,
                 orientation_B: np.ndarray, time_s: np.ndarray,
                 root: Path | str | None = None) -> np.ndarray:
    """Apply the saved held-out StrongLocal fold to a G2 tool path for display."""
    from .generator_g1 import stronglocal_features

    generated = record.generated
    take = _scalar(generated.arrays["take"])
    model_path = _root(root) / "outputs" / "generator_g1" / "fold_models" / f"heldout_{take}" / "model.npz"
    if not model_path.is_file():
        raise ValueError(f"{record.generated.record_id}: missing saved StrongLocal fold model {take}")
    with np.load(model_path, allow_pickle=False) as z:
        mean, scale, coef = (np.asarray(z[key], float) for key in ("mean", "scale", "coef"))
        intercept = float(np.asarray(z["intercept"]).item())
    features = stronglocal_features(
        position_B, orientation_B, time_s,
        np.asarray(generated.arrays["normalized_phase"], float),
        _scalar(generated.arrays["phase"]),
        np.asarray(generated.arrays["plate_position_B"], float),
    )
    raw_delta = ((features - mean) / scale) @ coef + intercept
    psi0 = float(np.asarray(generated.arrays["psi0"]).item())
    return psi0 + raw_delta - raw_delta[0]


def load_g2_comparison_record(record_id: str, method: str = "g2b",
                              root: Path | str | None = None) -> G2ComparisonRecord:
    if method not in ("g2a", "g2b"):
        raise ValueError("G2 comparison method must be g2a or g2b")
    base = _root(root)
    g1 = load_comparison_record(record_id, "contextual_promp", base)
    if method == "g2a":
        path = base / "outputs" / "generator_g2a" / "generated" / "contextual_promp_endpoint_constrained" / f"{record_id}.npz"
        position_key, orientation_key = "constrained_position_B", "constrained_orientation_B"
    else:
        path = base / "outputs" / "generator_g2b" / "generated" / "oracle_full_pose_constrained" / f"{record_id}.npz"
        position_key, orientation_key = "full_pose_constrained_position_B", "full_pose_constrained_orientation_B"
    if not path.is_file():
        raise ValueError(f"{record_id}: missing {method.upper()} constrained artifact")
    with np.load(path, allow_pickle=False) as z:
        position = np.asarray(z[position_key], float)
        orientation = np.asarray(z[orientation_key], float)
        time_s = np.asarray(z["time_s"], float)
        normalized_phase = np.asarray(z["normalized_phase"], float)
        identity = str(np.asarray(z["record_id"]).item())
    if identity != record_id or position.shape != (FRAME_COUNT, 3) or orientation.shape != (FRAME_COUNT, 3, 3):
        raise ValueError(f"{record_id}: invalid {method.upper()} artifact identity or shape")
    if not np.array_equal(np.asarray(g1.generated.arrays["normalized_phase"], float),
                          normalized_phase):
        raise ValueError(f"{record_id}: {method.upper()} phase grid differs from G1")
    psi = _display_psi(g1, position, orientation, time_s, base)
    for value in (position, orientation, psi):
        value.setflags(write=False)
    return G2ComparisonRecord(g1, method, position, orientation, psi)


def _g2_overlay(record: G2ComparisonRecord, index: int, robot: Any,
                q: np.ndarray, show_mouth_proxy: bool = False):
    """Overlay G2 desired paths; no G2 robot FK is implied."""
    from sew_mimic.visualization import AxisPrimitive, LinePrimitive, OverlayFrame, SpherePrimitive

    g1 = record.g1
    generated = g1.generated
    measured = g1.reference
    path, orientation = _base_path(g1, record.constrained_position_B,
                                   record.constrained_orientation_B)
    measured_p = np.asarray(measured.arrays["measured_U_position_base"], float)
    measured_R = np.asarray(measured.arrays["measured_U_rotation_base"], float)
    nominal_p = np.asarray(generated.arrays["generated_U_position_base"], float)
    nominal_R = np.asarray(generated.arrays["generated_U_rotation_base"], float)
    lines = [LinePrimitive("g2_constrained_U_path", a, b, .0035, (.15, 1., .25, .9))
             for a, b in zip(path[:-1], path[1:])]
    lines += [LinePrimitive("measured_U_path", a, b, .0025, (1., .55, .1, .75))
              for a, b in zip(measured_p[:-1], measured_p[1:])]
    lines += [LinePrimitive("g0_nominal_U_path", a, b, .0018, (.55, .55, .55, .55))
              for a, b in zip(nominal_p[:-1], nominal_p[1:])]
    axes = [AxisPrimitive("g2_constrained_U", path[index], orientation[index], .07),
            AxisPrimitive("measured_U", measured_p[index], measured_R[index], .055),
            AxisPrimitive("g0_nominal_U", nominal_p[index], nominal_R[index], .045)]
    R, p = np.asarray(generated.mounting["R_B_from_base"], float), np.asarray(generated.mounting["p_B_of_base"], float)
    plate = base_transform(np.asarray(generated.arrays["plate_position_B"], float)[None], R, p)[0]
    spheres = [SpherePrimitive("plate_context", plate, .035, (.8, .8, .8, .5))]
    if show_mouth_proxy:
        mouth = base_transform(np.asarray(generated.arrays["mouth_proxy_position_B"], float)[None], R, p)[0]
        spheres.append(SpherePrimitive("derived_mouth_proxy_not_measured", mouth, .025, (.3, .6, 1., .55)))
    return OverlayFrame(tuple(spheres), tuple(lines), tuple(axes))


def _render_scene(renderer: Any, robot: Any, data: Any, camera: Any, overlay: Any,
                  q: np.ndarray) -> np.ndarray:
    from sew_mimic.visualization import overlay_base_to_world

    set_stored_q(robot, data, q)
    renderer.update_scene(data, camera=camera)
    base_id = int(robot.frame_body_ids[0])
    world = overlay_base_to_world(overlay, data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
    append_overlay_to_scene(renderer.scene, world)
    return renderer.render()


def export_g2_video(record: G2ComparisonRecord, output: Path, speed: float = .25,
                    show_mouth_proxy: bool = False) -> dict[str, Any]:
    if speed not in (.25, .5, 1., 2.):
        raise ValueError("speed must be one of 0.25, 0.5, 1, 2")
    import mujoco

    g1 = record.g1
    robot, data = mount_g1_record(g1.generated)
    camera = mujoco.MjvCamera(); _camera(camera, g1.generated, robot, data)
    renderer = mujoco.Renderer(robot.model, 450, 600)
    time_s = np.asarray(g1.generated.arrays["time_s"], float)
    dt = np.diff(time_s)
    if len(dt) != 100 or np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-12, rtol=0):
        raise ValueError("G2 visualization requires uniform G1 display timing")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fps = speed / float(dt[0])
    import subprocess
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", f"{fps:.12g}", "-i", "-",
               "-pix_fmt", "yuv420p", str(output.resolve())]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    phase = np.asarray(g1.generated.arrays["normalized_phase"], float)
    constrained_base, _ = _base_path(g1, record.constrained_position_B, record.constrained_orientation_B)
    measured_base = np.asarray(g1.reference.arrays["measured_U_position_base"], float)
    nominal_base = np.asarray(g1.generated.arrays["generated_U_position_base"], float)
    psi_human = np.asarray(g1.human_psi, float)
    psi_reference = np.asarray(g1.reference.arrays["stronglocal_psi"], float)
    psi_error_reference = np.abs(wrap(psi_reference - psi_human))
    psi_error_g2 = np.abs(wrap(record.constrained_psi - psi_human))
    distance_nominal = np.linalg.norm(nominal_base - measured_base, axis=1)
    distance_g2 = np.linalg.norm(constrained_base - measured_base, axis=1)
    label = "G2A constrained U" if record.method == "g2a" else "G2B full-pose constrained U"
    try:
        for index in range(FRAME_COUNT):
            frame = replay_frame(g1, "StrongLocal", index, robot, None, show_mouth_proxy)
            images = [
                ("Measured U + human ψ", _render_scene(renderer, robot, data, camera,
                                                       frame["overlay"], frame["human_gt_q"])),
                ("Measured U + predicted ψ", _render_scene(renderer, robot, data, camera,
                                                            frame["overlay"], frame["reference_q"])),
                (label, _render_scene(renderer, robot, data, camera,
                                     _g2_overlay(record, index, robot, frame["human_gt_q"], show_mouth_proxy),
                                     frame["human_gt_q"])),
            ]
            fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
            grid = fig.add_gridspec(2, 6, height_ratios=[3, 1], hspace=.27, wspace=.08)
            for column, (title, image) in enumerate(images):
                axis = fig.add_subplot(grid[0, column * 2:(column + 1) * 2]); axis.imshow(image); axis.set_title(title, fontsize=10); axis.axis("off")
            psi_axis = fig.add_subplot(grid[1, :3])
            psi_axis.plot(phase, psi_error_reference, color="orange", label="|reference ψ − human ψ|")
            psi_axis.plot(phase, psi_error_g2, color="green", label=f"|{record.method.upper()} ψ − human ψ|")
            psi_axis.axvline(phase[index], color="red"); psi_axis.set_title("Redundancy-angle difference")
            psi_axis.set_xlabel("normalized phase"); psi_axis.set_ylabel("absolute error (rad)"); psi_axis.legend(fontsize=7)
            distance_axis = fig.add_subplot(grid[1, 3:])
            distance_axis.plot(phase, distance_nominal, color="gray", label="G0 nominal vs measured U")
            distance_axis.plot(phase, distance_g2, color="green", label=f"{record.method.upper()} vs measured U")
            distance_axis.axvline(phase[index], color="red"); distance_axis.set_title("Tool-distance difference")
            distance_axis.set_xlabel("normalized phase"); distance_axis.set_ylabel("distance (m)"); distance_axis.legend(fontsize=7)
            fig.subplots_adjust(left=.035, right=.99, bottom=.08, top=.94)
            fig.canvas.draw(); rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
            assert process.stdin is not None; process.stdin.write(rgb.tobytes())
    finally:
        if process.stdin: process.stdin.close()
        renderer.close()
    if process.wait() != 0:
        raise RuntimeError("G2 visualization ffmpeg export failed")
    return {"record_id": g1.generated.record_id, "method": record.method,
            "output": str(output.resolve()), "mode": "g2-three-way-split",
            "panels": ["measured_U_human_psi", "measured_U_predicted_psi", f"{record.method}_constrained_U"],
            "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}"}


def export_g2_playlist(records: list[G2ComparisonRecord], output: Path, speed: float = .25,
                       show_mouth_proxy: bool = False) -> dict[str, Any]:
    if not records:
        raise ValueError("G2 video playlist must contain at least one record")
    takes = {_scalar(item.g1.generated.arrays["take"]) for item in records}
    methods = {item.method for item in records}
    if len(takes) != 1:
        raise ValueError("one G2 playlist must use records from the same take")
    if len(methods) != 1:
        raise ValueError("one G2 playlist must use one method")
    if len(records) == 1:
        return export_g2_video(records[0], output, speed, show_mouth_proxy)
    import subprocess
    import tempfile

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g2v_playlist_") as temp:
        parts = []
        for index, record in enumerate(records):
            part = Path(temp) / f"part_{index:03d}.mp4"
            export_g2_video(record, part, speed, show_mouth_proxy)
            parts.append(part)
        command = ["ffmpeg", "-y", "-loglevel", "error"]
        for part in parts:
            command.extend(["-i", str(part)])
        labels = "".join(f"[{index}:v]" for index in range(len(parts)))
        command.extend(["-filter_complex", f"{labels}concat=n={len(parts)}:v=1:a=0[v]", "-map", "[v]",
                        "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(output.resolve())])
        if subprocess.run(command, check=False).returncode != 0:
            raise RuntimeError("G2 playlist concatenation failed")
    return {"record_ids": [item.g1.generated.record_id for item in records],
            "method": records[0].method, "output": str(output.resolve()),
            "mode": "g2-three-way-playlist", "segments": len(records),
            "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}"}
