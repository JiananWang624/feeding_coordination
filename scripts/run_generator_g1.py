from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.generator_g1 import GeneratorG1Config, process_generator_g1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1 end-to-end simulated feeding feasibility")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "generator_g1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); cfg = GeneratorG1Config.load(args.config)
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    summary = process_generator_g1(resolve(cfg.phase1_output_path), resolve(cfg.phase32_output_path),
        resolve(cfg.generator_g0_output_path), resolve(cfg.robot_r0_output_path),
        args.output or resolve(cfg.output_path), cfg)
    print(json.dumps({"eligibility": summary["eligibility"], "fold_model_parity": summary["fold_model_parity"]}, indent=2))


if __name__ == "__main__":
    main()
