"""CLI entry point for the read-only Robot R1 replay loader."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "exact_sew" / "src"))
from feeding_coordination.robot_visualization import STRATEGIES, dry_run, event_at, list_records, load_record, interactive_replay, export_video

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--record-id"); p.add_argument("--strategy", choices=STRATEGIES, default="StrongLocal"); p.add_argument("--config",type=Path,default=ROOT/"configs/robot_visualization.json")
    p.add_argument("--list-records", action="store_true"); p.add_argument("--continuity-event", type=int); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--speed", type=float, choices=(.25,.5,1.,2.), default=1.); p.add_argument("--record-video", type=Path)
    p.add_argument("--show-elbow-trail", action="store_true"); p.add_argument("--show-tool-trail", action="store_true"); args=p.parse_args()
    config=json.loads(args.config.read_text()); output_root=(ROOT/config["output_path"]).resolve()
    if args.list_records: print(list_records(strategy=args.strategy).to_string(index=False)); return
    start_index=0
    if args.continuity_event is not None:
        row=event_at(None,args.continuity_event); args.record_id=str(row["record_id"]); args.strategy=str(row["strategy"])
        import numpy as np
        record=load_record(args.record_id); start_index=max(0,int(np.flatnonzero(record.arrays["motive_frame"]==row["current_motive_frame"])[0])-3); print(row)
    if not args.record_id: p.error("--record-id (or --continuity-event) is required")
    if args.dry_run: print(dry_run(args.record_id,args.strategy)); return
    record=load_record(args.record_id)
    if args.record_video:
        if output_root not in args.record_video.resolve().parents: p.error("--record-video must be under configured output_path")
        export_video(record,args.strategy,args.record_video,speed=args.speed,show_elbow_trail=args.show_elbow_trail,show_tool_trail=args.show_tool_trail); return
    print("Controls: Space pause, J/L step, R restart, [/] strategy, E next event, H human, T tool.")
    interactive_replay(record,args.strategy,speed=args.speed,show_elbow_trail=args.show_elbow_trail,show_tool_trail=args.show_tool_trail,start_index=start_index)
if __name__ == "__main__": main()
