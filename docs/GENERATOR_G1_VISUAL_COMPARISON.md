# G1V measured/generated visual comparison

G1V is a presentation-only, read-only replay of frozen G1 artifacts. Its video comparison has three synchronized MuJoCo panels: measured tool trajectory with measured human ψ, measured tool trajectory with predicted ψ, and generated tool trajectory with predicted ψ.

## Quick start

List the meeting presets or all eligible record/generator pairs:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py --list-presets
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py --list-records
```

Launch the interactive overlay viewer:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --record-id trial_0014_bite_004_transfer `
  --generator contextual_promp `
  --mode overlay `
  --strategy StrongLocal `
  --speed 0.25
```

The interactive viewer uses the shared replay-frame function used by video export. Overlay mode shows both desired tool paths and frames, the generated robot, its realized tool frame, the measured-reference realized frame, plate context, and failure/continuity markers. The synchronized values are printed as a compact HUD line. Split video mode renders the three-way comparison described above.

Play several segments from the same take consecutively in one MuJoCo window by repeating `--record-id`:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --record-id trial_0014_bite_004_transfer `
  --record-id trial_0014_bite_004_withdrawal `
  --generator contextual_promp `
  --mode overlay `
  --speed 0.25
```

The same ids can be placed one per line in a file and passed with `--playlist`. Playlist playback and split-video export require all segments to share a take/mounting.

For a longer split video, repeat `--record-id` and provide `--record-video`:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --record-id trial_0014_bite_001_transfer `
  --record-id trial_0014_bite_001_withdrawal `
  --record-id trial_0014_bite_002_transfer `
  --record-id trial_0014_bite_002_withdrawal `
  --generator contextual_promp --mode split --speed 0.25 `
  --record-video outputs/generator_g1_visualization/videos/three_way_playlist.mp4
```

Load a preset and export a split comparison video:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --preset promp_clean_transfer `
  --record-video outputs/generator_g1_visualization/videos/demo.mp4
```

Use `--export-required-videos` to reproduce the meeting videos. Playback speeds are limited to `0.25`, `0.5`, `1.0`, and `2.0`; the preset default is `0.25`.

Optional `--compare-b0` and `--compare-robotsmooth` select a generated control strategy instead of generated StrongLocal. `--show-mouth-proxy` adds the derived mouth proxy; it is off by default.

## Meaning of the three references

- **Measured human ψ** is the measured Phase 1.5 redundancy angle. The first panel uses the saved Robot R1 HumanGT joint replay, resampled by nearest saved source frame onto the G1 display grid; failed source samples are held and marked in the HUD.
- **Measured-reference StrongLocal** is the saved G1 StrongLocal prediction and saved robot result driven by the measured demonstration U.
- **Generated StrongLocal** is the saved G1 StrongLocal prediction and saved robot result driven by contextual ProMP or retrieval U.

The split view uses, from left to right, HumanGT on measured U, measured-reference StrongLocal, and generated StrongLocal. Desired and realized U frames are distinct: realized frames are reconstructed by FK from stored q. No inverse kinematics is run by G1V.

The lower video panel displays all three psi curves and a moving sample cursor. Its compact annotation reports desired generated-versus-measured tool position/orientation error and generated-versus-measured-reference robot wrapped-q, elbow, and arm-plane differences. A failed saved solver frame is shown as `PIPELINE FAILURE - VIEWER HOLD ONLY`; a viewer hold never changes success or saved metrics.

## Important limitations

- Timing is the provisional G0/G1 `time_s` timing. It has not been physically retimed to Gen3 velocity or acceleration limits.
- The optional mouth point is a **derived mouth proxy**, not an independently measured mouth.
- This viewer does not perform collision analysis, smoothing, physical calibration, fork-tip calibration, or real-robot execution validation.
- The G1 and upstream scientific output trees are opened read-only. G1V writes only presentation artifacts beneath `outputs/generator_g1_visualization/`.

## G2A/G2B endpoint-constrained comparison

The updated G2 visualizer keeps the three-panel layout while reading the endpoint-constrained artifacts:

1. measured tool path + human ψ;
2. measured tool path + predicted ψ;
3. the constrained G2 tool path (G2A endpoint-constrained or G2B full-pose-constrained).

The lower-left plot is the absolute redundancy-angle difference to human ψ. The lower-right plot is the tool-position distance difference to the measured path, comparing the G0 nominal path and the selected G2 path. The old lower-right numeric HUD is not rendered in these videos.

G2A and G2B artifacts are trajectory-only experiments; they do not contain a new robot-q/IK realization. The third MuJoCo panel therefore overlays the constrained tool path on the saved human joint replay rather than claiming a new robot replay.

Export eight consecutive transfer segments in 1920×1080:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --record-id trial_0014_bite_001_transfer `
  --record-id trial_0014_bite_002_transfer `
  --record-id trial_0014_bite_003_transfer `
  --record-id trial_0014_bite_004_transfer `
  --record-id trial_0014_bite_005_transfer `
  --record-id trial_0014_bite_006_transfer `
  --record-id trial_0014_bite_007_transfer `
  --record-id trial_0014_bite_008_transfer `
  --g2-method g2a --mode split --speed 0.25 `
  --record-video outputs/generator_g1_visualization/videos/g2a_three_way_8segment_1080p.mp4
```

Use `--g2-method g2b` and a different output name for G2B.
