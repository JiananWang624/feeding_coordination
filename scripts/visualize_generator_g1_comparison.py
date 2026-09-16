from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.generator_g1_visual_comparison import (
    export_generator_pair_video,
    export_split_video,
    interactive_overlay,
    load_comparison_record,
)


OUTPUT = ROOT / "outputs" / "generator_g1_visualization"
PRESETS = OUTPUT / "presets.json"


def load_presets() -> dict[str, dict]:
    payload = json.loads(PRESETS.read_text())
    if payload.get("schema_version") != "generator-g1-visualization-presets-v1":
        raise ValueError("unsupported G1V preset schema")
    return payload["presets"]


def export_required_videos() -> list[dict]:
    videos = OUTPUT / "videos"
    jobs = [
        ("promp_clean_transfer", "measured_vs_promp_stronglocal_transfer.mp4"),
        ("promp_clean_withdrawal", "measured_vs_promp_stronglocal_withdrawal.mp4"),
        ("retrieval_clean_example", "measured_vs_retrieval_stronglocal.mp4"),
        ("generator_error_propagation", "generator_error_propagation_comparison.mp4"),
    ]
    presets = load_presets(); reports = []
    for name, filename in jobs:
        item = presets[name]
        record = load_comparison_record(item["record_id"], item["generator"], ROOT)
        reports.append(export_split_video(record, item["strategy"], videos / filename, item["speed"]))
    pair = presets["promp_vs_retrieval"]
    left = load_comparison_record(pair["record_id"], pair["generator"], ROOT)
    right = load_comparison_record(pair["record_id"], pair["compare_generator"], ROOT)
    reports.append(export_generator_pair_video(
        left, right, videos / "promp_vs_retrieval_same_record.mp4",
        pair["strategy"], pair["speed"],
    ))
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "generator-g1-visualization-v1",
        "presentation_only": True,
        "viewer_recomputes_q": False,
        "timing_physically_retimed": False,
        "source_g1_manifest": str((ROOT / "outputs/generator_g1/manifest.json").resolve()),
        "source_g1_manifest_sha256": digest(ROOT / "outputs/generator_g1/manifest.json"),
        "source_phase1_manifest_sha256": digest(ROOT / "outputs/phase1/manifest.json"),
        "presets_sha256": digest(PRESETS),
        "videos": reports,
        "snapshots": [str(path.relative_to(OUTPUT)).replace("\\", "/")
                      for path in sorted((OUTPUT / "snapshots").glob("*.png"))],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only measured/generated G1 stored-q comparison")
    parser.add_argument("--record-id")
    parser.add_argument("--generator", choices=("contextual_promp", "retrieval"), default="contextual_promp")
    parser.add_argument("--compare-generator", choices=("contextual_promp", "retrieval"))
    parser.add_argument("--mode", choices=("overlay", "split"))
    parser.add_argument("--strategy", choices=("B0", "StrongLocal", "RobotSmooth"), default="StrongLocal")
    parser.add_argument("--preset")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--list-records", action="store_true")
    parser.add_argument("--record-video", type=Path)
    parser.add_argument("--export-required-videos", action="store_true")
    parser.add_argument("--speed", type=float, choices=(.25, .5, 1., 2.))
    parser.add_argument("--show-mouth-proxy", action="store_true")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--compare-b0", action="store_true")
    controls.add_argument("--compare-robotsmooth", action="store_true")
    args = parser.parse_args()
    presets = load_presets()
    if args.list_presets:
        print(json.dumps(presets, indent=2)); return
    if args.list_records:
        metrics = pd.read_csv(ROOT / "outputs/generator_g1/pipeline_metrics.csv")
        print(metrics[["record_id", "phase", "generator", "stronglocal_complete_pipeline_success"]]
              .sort_values(["record_id", "generator"]).to_string(index=False)); return
    if args.export_required_videos:
        print(json.dumps(export_required_videos(), indent=2)); return
    item = presets.get(args.preset, {}) if args.preset else {}
    if args.preset and not item:
        parser.error(f"unknown preset {args.preset!r}; use --list-presets")
    record_id = args.record_id or item.get("record_id")
    generator = item.get("generator", args.generator)
    compare_generator = args.compare_generator or item.get("compare_generator")
    mode = args.mode or item.get("mode", "overlay")
    speed = args.speed if args.speed is not None else item.get("speed", .5)
    strategy = "B0" if args.compare_b0 else "RobotSmooth" if args.compare_robotsmooth else item.get("strategy", args.strategy)
    if not record_id:
        parser.error("--record-id or --preset is required")
    record = load_comparison_record(record_id, generator, ROOT)
    if args.record_video:
        if mode != "split":
            parser.error("video export uses the reliable composed split mode; pass --mode split")
        if compare_generator:
            other = load_comparison_record(record_id, compare_generator, ROOT)
            report = export_generator_pair_video(record, other, args.record_video, strategy, speed, args.show_mouth_proxy)
        else:
            report = export_split_video(record, strategy, args.record_video, speed, args.show_mouth_proxy)
        print(json.dumps(report, indent=2)); return
    if mode != "overlay":
        parser.error("interactive viewing supports overlay mode; split mode is available through --record-video")
    interactive_overlay(record, strategy, speed, args.show_mouth_proxy)


if __name__ == "__main__":
    main()
