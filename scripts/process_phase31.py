from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from feeding_coordination.phase31 import Phase31Config, process_phase31

def main() -> None:
    parser=argparse.ArgumentParser(description="Run Phase 3.1 nested local-ablation audit")
    parser.add_argument("--config",type=Path,default=ROOT/"configs"/"phase31.json"); parser.add_argument("--input",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); cfg=Phase31Config.load(args.config)
    print(process_phase31(args.input or ROOT/cfg.phase1_output_path,args.output or ROOT/cfg.output_path,cfg)["overall"])
if __name__ == "__main__": main()
