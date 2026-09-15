from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feeding_coordination.phase3 import Phase3Config, process_phase3

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 human coordination information hierarchy")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase3.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    source = args.input if args.input else ROOT / cfg.phase1_output_path
    output = args.output if args.output else ROOT / cfg.output_path
    print(process_phase3(source, output, cfg)["overall"])


if __name__ == "__main__":
    main()
