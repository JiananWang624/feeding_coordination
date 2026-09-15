# Phase 1 contract

Each source `trajectory_id` is one immutable demonstration/segment and is stored
as `outputs/phase1/demos/<trajectory_id>.npz`; `manifest.json` is deterministic
metadata and schema index. Arrays retain time, source event/phase and phase
progress, B-frame landmarks, raw and interpolation masks, Euler degrees and
reconstructed hand SO(3), position-only plate/mouth target semantics, SEW,
and diagnostic body features.

`L` is Motive Global in mm. `B` is project body/world in m. The only transform
is configured in `configs/phase1.json`: `p_B=.001 R_B_L p_L + t`, with the
audited matrix and zero translation. Rotations use `R_B_L R_LH`.

Raw wrist Euler is the upstream fork's quaternion-derived **extrinsic xyz**
degrees (equivalently `Rz @ Ry @ Rx`); Phase 1 reconstructs it with SciPy
lowercase `from_euler('xyz', degrees=True)`. The upstream recorder quaternion
ordering was xyzw and its normalization happened upstream. No quaternion is
reprocessed here.

`Hand_XYZ` is the upstream derived fork-center plus 100 mm times negative local
Y, not an independent marker. It is retained as an observed derived hand field.
There is no reliable H-to-U calibration: `tool_position` and `tool_orientation`
are NaN and `tool_pose_valid` false with `missing_tool_calibration`; no identity
transform is assumed. `target_xyz` is mouth-target XYZ only and untrusted as an
actual mouth measurement; plate has position only.

SEW uses the frozen `StereoSew(project_stereo_sew_reference())` directly on
measured B-frame S/E/W. Its frozen reference is defined in native Gen3 base
coordinates and is explicitly rotated into B with
`R_B_from_exact_sew_base = Rx(+90 degrees)` before constructing `StereoSew`.
Wrapped angles are lifted only in contiguous valid
runs. Near geometry is flagged before frozen forward evaluation.

NPZ keys are grouped as source indexing (`time`, `source_time_s`,
`motive_frame`, `event_frame_index`, phase fields); B-frame landmarks;
position-only plate/mouth fields; raw Euler and hand SO(3); unavailable tool
pose; ψ and diagnostics; and per-landmark `valid_observation`, `interpolated`,
and `long_missing` masks. `orientation_invalid`, `missing_calibration`,
`psi_invalid`, `psi_near_singular`, and `processing_status` are explicit.
Interpolation is linear only across up to three absent frames and 0.03 s in one
segment. ψ is rejected when S--W < 0.01 m, arm-plane sine < 0.01, or the frozen
stereo reference denominator < 0.01. Manifest records add source take/bite,
per-record ψ/processing counts, provenance, conventions, config/code hashes,
and aggregate quality statistics.
