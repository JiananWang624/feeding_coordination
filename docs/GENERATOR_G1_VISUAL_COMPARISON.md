# G1V measured/generated visual comparison

G1V is a presentation-only, read-only replay of frozen G1 artifacts. It places the measured demonstration beside a generated trajectory and replays the corresponding saved robot configurations in lockstep on the common 101-sample grid.

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
  --speed 0.5
```

The interactive viewer uses the shared replay-frame function used by video export. Overlay mode shows both desired tool paths and frames, the generated robot, its realized tool frame, the measured-reference realized frame, plate context, and failure/continuity markers. The synchronized values are printed as a compact HUD line. Split mode is used for the clearest dual-robot video presentation.

Load a preset and export a split comparison video:

```powershell
.venv\Scripts\python.exe scripts\visualize_generator_g1_comparison.py `
  --preset promp_clean_transfer `
  --record-video outputs/generator_g1_visualization/videos/demo.mp4
```

Use `--export-required-videos` to reproduce the five meeting videos. Playback speeds are limited to `0.25`, `0.5`, `1.0`, and `2.0`; the preset default is `0.5`.

Optional `--compare-b0` and `--compare-robotsmooth` select a generated control strategy instead of generated StrongLocal. `--show-mouth-proxy` adds the derived mouth proxy; it is off by default.

## Meaning of the three references

- **Measured human psi** is the measured Phase 1.5 redundancy angle, resampled only for display onto the same normalized 101-sample grid.
- **Measured-reference StrongLocal** is the saved G1 StrongLocal prediction and saved robot result driven by the measured demonstration U.
- **Generated StrongLocal** is the saved G1 StrongLocal prediction and saved robot result driven by contextual ProMP or retrieval U.

The split view uses the measured-reference stored q on the left and the generated-pipeline stored q on the right. Desired and realized U frames are distinct: realized frames are reconstructed by FK from the stored q and checked against the saved G1 realized poses. No inverse kinematics is run by G1V.

The lower video panel displays all three psi curves and a moving sample cursor. Its compact annotation reports desired generated-versus-measured tool position/orientation error and generated-versus-measured-reference robot wrapped-q, elbow, and arm-plane differences. A failed saved solver frame is shown as `PIPELINE FAILURE - VIEWER HOLD ONLY`; a viewer hold never changes success or saved metrics.

## Important limitations

- Timing is the provisional G0/G1 `time_s` timing. It has not been physically retimed to Gen3 velocity or acceleration limits.
- The optional mouth point is a **derived mouth proxy**, not an independently measured mouth.
- This viewer does not perform collision analysis, smoothing, physical calibration, fork-tip calibration, or real-robot execution validation.
- The G1 and upstream scientific output trees are opened read-only. G1V writes only presentation artifacts beneath `outputs/generator_g1_visualization/`.
