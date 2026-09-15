# Robot Visualization V0A

This is a read-only MuJoCo replay of saved Robot R1 trajectories. It does not
import or call an IK solver, recompute psi or continuity labels, smooth or
interpolate configurations, or write Robot R0/R1/Phase artifacts.

## Artifact schema and mounting

The loader reads `outputs/robot_r1/manifest.json`, verifies its recorded Robot
R0 manifest SHA-256, and opens the listed R1 NPZ. Record identity comes from
`take`, `parent_bite_id`, `record_id`, `phase`, `original_frame`,
`motive_frame`, `time_s`, and `run_id`. Desired task data comes from
`desired_U_position_base` and `desired_U_rotation_base`. Each of B0, B1, B2,
B3, StrongLocal, H_star, HumanGT, and RobotSmooth is read from the saved
`{strategy}_q`, `solver_status`, `strategy_psi`, `continuity_violation`,
`branch_id`, `search_branch`, and `actual_virtual_U_*_base` arrays.

Per-take `anchor_B`, `robot_world_offset_m`, `R_B_from_base`, and
`p_B_of_base` come from the provenance-locked Robot R0 manifest. Before replay,
the loader rebuilds the frozen Gen3 mount and requires 1e-12 agreement for the
saved rotation/translation and the derived joint-1 position. It also requires
exact equality between the R1 and matching Phase 1.5 `motive_frame` arrays.

## Commands and controls

```text
python scripts/visualize_robot_r1.py --list-records --strategy RobotSmooth
python scripts/visualize_robot_r1.py --record-id trial_0017_bite_015_transfer --strategy StrongLocal --dry-run
python scripts/visualize_robot_r1.py --continuity-event 78
python scripts/visualize_robot_r1.py --record-id trial_0017_bite_015_transfer --strategy StrongLocal --record-video outputs/robot_r1_visualization/videos/example.mp4
```

`--speed` accepts 0.25, 0.5, 1, and 2 and changes wall-clock playback only.
Optional `--show-elbow-trail` and `--show-tool-trail` use FK of saved successful
q values and never bridge a failure or source-run gap.

Interactive controls are: Space pause/resume; J/L previous/next frame; R
restart; [/] previous/next saved strategy; E exact next stored continuity
violation; H human overlay; and T desired/actual tool frames and paths. The
camera remains a normal free MuJoCo camera after initialization.

## Overlay and event semantics

The orange shoulder/elbow/wrist and arm segments are the **Human measured
reference**, not robot targets. Cyan markers use the validated Gen3 SEW
joint-1/elbow/wrist definitions. Desired virtual U is a green origin plus an XYZ
triad and bounded path; actual virtual U is an aligned-pinch origin plus an XYZ
triad. The grey sphere is only a measured plate-position marker and implies no
physical plate dimensions.

At solver failure, the viewer holds only the preceding successful q in the same
saved run (or neutral q if none), adds a magenta desired-U marker, and prints
`IK FAILURE — VIEWER HOLD ONLY` with record, frame, strategy, and saved status.
It never changes the saved status. Continuity highlighting and diagnostics use
the existing R1 boolean and CSV row verbatim. `--continuity-event N` starts
three frames before that CSV event; E jumps to the exact next event frame.

The HumanGT/StrongLocal preset is chosen deterministically from stored metrics:
maximum StrongLocal-minus-HumanGT continuity-clean travel on exact common
support, with lexical record-id tie-breaking. It is not selected by appearance.

Validated examples are `videos/clean_transfer_stronglocal.mp4` (332 frames,
3.32 s) and `videos/clean_withdrawal_stronglocal.mp4` (227 frames, 2.27 s), both
under `outputs/robot_r1_visualization` and encoded at the saved 100 Hz cadence.
