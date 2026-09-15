from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_coordination.robot_r1 import RobotR1Config, process_robot_r1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "robot_r1.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = RobotR1Config.load(args.config)
    summary = process_robot_r1(
        args.input or ROOT / config.robot_r0_output_path,
        args.output or ROOT / config.output_path,
        config,
    )
    print(summary["overall"])


if __name__ == "__main__":
    main()
