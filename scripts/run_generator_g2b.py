from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.generator_g2b import process_generator_g2b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transfer-only G2B oracle full-pose feasibility study")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "generator_g2b")
    args = parser.parse_args()
    summary = process_generator_g2b(ROOT, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
