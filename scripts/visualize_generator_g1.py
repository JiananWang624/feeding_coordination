from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.generator_g1_visualization import (
    export_g1_video, export_representative_videos, load_g1_record, viewer_parity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only stored-q G1 MuJoCo replay")
    parser.add_argument("--record-id")
    parser.add_argument("--generator", choices=("retrieval", "contextual_promp", "measured_reference"))
    parser.add_argument("--strategy", choices=("B0", "StrongLocal", "RobotSmooth"), default="StrongLocal")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--export-representatives", action="store_true")
    args = parser.parse_args()
    if args.export_representatives:
        print(json.dumps(export_representative_videos(ROOT), indent=2)); return
    if not args.record_id or not args.generator:
        parser.error("--record-id and --generator are required unless --export-representatives is used")
    record = load_g1_record(args.record_id, args.generator, ROOT)
    report = export_g1_video(record, args.strategy, args.video) if args.video else viewer_parity(record, args.strategy)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
