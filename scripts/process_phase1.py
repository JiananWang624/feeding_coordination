from __future__ import annotations
import argparse
from pathlib import Path
from feeding_coordination.phase1 import Phase1Config
from feeding_coordination.phase15 import process_csv
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--config',default=ROOT/'configs/phase1.json')
if __name__=='__main__':
 a=p.parse_args(); c=Phase1Config.load(Path(a.config)); i=Path(a.input) if a.input else ROOT/c.input_path; o=Path(a.output) if a.output else ROOT/c.output_path; print(process_csv(i,o,c)['summary'])
