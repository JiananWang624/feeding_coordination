from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from feeding_coordination.phase32 import Phase32Config, process_phase32
def main():
    p=argparse.ArgumentParser(description="Run Phase 3.2 human baselines and frozen local predictor"); p.add_argument("--config",type=Path,default=ROOT/"configs"/"phase32.json"); p.add_argument("--input",type=Path); p.add_argument("--output",type=Path); a=p.parse_args(); cfg=Phase32Config.load(a.config); print(process_phase32(a.input or ROOT/cfg.phase1_output_path,a.output or ROOT/cfg.output_path,cfg)["overall"])
if __name__ == "__main__": main()
