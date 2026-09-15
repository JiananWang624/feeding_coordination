from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from feeding_coordination.robot_r2 import RobotR2Config,process_robot_r2
p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"configs/robot_r2.json");p.add_argument("--output",type=Path);a=p.parse_args();c=RobotR2Config.load(a.config);print(process_robot_r2(ROOT,a.output or ROOT/c.output_path,c))
