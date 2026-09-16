from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from feeding_coordination.robot_r3 import RobotR3Config, process_robot_r3
p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/robot_r3.json"); p.add_argument("--output",type=Path)
a=p.parse_args(); cfg=RobotR3Config.load(a.config); print(process_robot_r3(ROOT,a.output or ROOT/cfg.output_path,cfg))
