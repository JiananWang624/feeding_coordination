from __future__ import annotations
import argparse
from pathlib import Path
from feeding_coordination.phase2 import Phase2Config, process_phase2
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output"); p.add_argument("--config",default=ROOT/"configs/phase2.json")
if __name__ == "__main__":
    a=p.parse_args(); cfg=Phase2Config.load(Path(a.config)); source=Path(a.input) if a.input else ROOT/cfg.phase1_output_path; output=Path(a.output) if a.output else ROOT/cfg.output_path; print(process_phase2(source,output,cfg)["summary"]["overall"])
