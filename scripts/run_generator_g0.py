from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.generator_g0 import GeneratorG0Config, process_generator_g0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G0 fork-trajectory generation feasibility")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "generator_g0.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = GeneratorG0Config.load(args.config)
    summary = process_generator_g0(args.input or ROOT / cfg.phase1_output_path,
                                   args.output or ROOT / cfg.output_path, cfg)
    print(json.dumps({"dataset": summary["dataset"], "generation": summary["generation"]}, indent=2))


if __name__ == "__main__":
    main()
