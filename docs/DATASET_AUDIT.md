# Dataset audit

## Included recordings and acquisition layer

The actual acquisition dataset is in the external sibling directory
`../feeding_data_analysis/data`. Phase 1 includes the seven complete recorder
trials `0012`, `0013`, `0014`, `0015`, `0017`, `0019`, and `0020`.

Each trial contains `metadata.json`, OptiTrack rigid-body and marker CSVs, two
RealSense RGB-D streams and timestamp CSVs, audio data/block metadata, and key
or bite-timing events. The recorder metadata describes 640 x 480 RealSense
color/depth at 30 Hz (depth aligned to color) and mono float32 audio at 48 kHz
with 1024-sample blocks. `host_time_s` is recorder-relative
`time.perf_counter_ns` time; cross-sensor synchronization accuracy has not been
established.

| Trial | Bites | Annotation rows | OptiTrack rows | Marker rows | Camera 1 / 2 frames | Audio blocks | Key events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0012 | 15 | 60 | 25,468 | 276,792 | 7,614 / 7,496 | 11,802 | 16 |
| 0013 | 12 | 48 | 36,600 | 399,447 | 10,977 / 10,811 | 17,023 | 10 |
| 0014 | 21 | 84 | 45,219 | 496,008 | 13,545 / 13,334 | 20,996 | 19 |
| 0015 | 19 | 76 | 21,801 | 238,095 | 6,502 / 6,365 | 10,021 | 19 |
| 0017 | 15 | 60 | 37,665 | 404,173 | 11,295 / 11,135 | 17,534 | 10 |
| 0019 | 26 | 104 | 35,308 | 387,139 | 10,566 / 10,402 | 16,380 | 24 |
| 0020 | 19 | 76 | 37,563 | 400,103 | 11,261 / 11,111 | 17,497 | 54 |
| **Total** | **127** | **508** | **239,624** | **2,601,757** | **71,760 / 70,654** | **111,253** | **152** |

The recorder `optitrack.csv` schema is `host_time_s`,
`natnet_frame_number`, `rigid_body_id`, `rigid_body_name`, `x/y/z`,
`qx/qy/qz/qw`, and `tracking_valid`. It contains the single named rigid body
`fork`; its numeric positions are metre-valued and its quaternion order is
xyzw. Marker tables contain the fork marker set (`top right`, `bottom right`,
`bottom left`) plus unlabeled/single markers. Model definitions contain a fork
asset and an arm asset, but no measured human-joint trajectory definition.

## Landmark and annotation layer used by Phase 1

Measured shoulder/elbow/wrist trajectories are available only in the Motive
export/annotation pipeline under `../optitrack_data_preprecessing`. Motive
headers establish Global coordinates, millimetres, and quaternion rotation.
The annotations have `take`, `bite_id`, `event`, `motive_frame`,
`natnet_timestamp_s`, and `annotated_at`. Every included bite has four ordered
boundaries: `transfer_start`, `transfer_end`, `withdrawal_start`, and
`withdrawal_end`. No finer semantic phases are present.

The direct deterministic Phase 1 input is
`../optitrack_data_preprecessing/result/trajectory_data/all_labeled_trajectories.csv`
(SHA-256 recorded in each generated manifest). It contains 33,999 rows and 254
independent `trajectory_id` segments: 127 transfer and 127 withdrawal segments
belonging to 127 parent bites across seven takes. Segment lengths are 52--332
frames (median 126.5), and `event_time_s` advances by 0.01 s, approximately
100 Hz. The available columns preserve:

- source/event time, Motive frame, event frame index, phase, and source
  `phase_progress`;
- `Shoulder`, `Elbow`, `Wrist`, and `Hand` XYZ with validity flags;
- `Wrist_Rx/Ry/Rz` and `wrist_rotation_valid`;
- position-only `Plate` XYZ and derived `target_x/y/z`;
- take, bite, trajectory identifiers, and `is_complete`.

`phase_progress` agrees with a deterministic normalization of event time to
`[0,1]` within `5e-10`; Phase 1 stores both values.

## Orientation, hand, plate, mouth, and calibration semantics

`Wrist_Rx/Ry/Rz` is not an independent anatomical wrist sensor. Upstream code
normalizes the fork rigid-body quaternion and converts it to extrinsic XYZ
Euler degrees (`Rz @ Ry @ Rx`). `Hand_X/Y/Z` is also derived: fork centre plus
100 mm along fork local negative Y. Phase 1 preserves these facts and rebuilds
SO(3), but does not reinterpret the fields as a measured anatomical hand pose.

Plate data is a static position estimate derived from persistent unlabeled
markers; there is no plate orientation. The so-called mouth target is the
transfer-end derived point `Hand + 80 mm * ForkMinusY`, held constant for the
bite. It is not a measured mouth position or pose and is stored as untrusted.

No independently measured hand-to-utensil transform, calibration ID,
tool/grasp calibration, grasp event, or physical justification for the 100 mm
and 80 mm offsets was found. The only named tool is `fork`. Phase 1.5 uses its
tracked rigid-body origin and axes directly as `U(t)`; this does not need an
anatomical hand-to-tool transform. Fork-tip/food-point calibration is still
missing and distinct from availability of the rigid tool trajectory.

## Missing data and exclusions

There are 32,234 complete and 1,765 incomplete source rows. Raw wrist XYZ is
missing in 1,698 rows and elbow XYZ in 197 rows; shoulder, derived hand, source
Euler, and source time are present throughout. By trial, wrist-invalid counts
are 269 (0013), 114 (0015), 935 (0017), 134 (0019), and 246 (0020); trial 0017
also has 197 elbow-invalid rows. Trials 0012 and 0014 have no incomplete rows.

Nineteen older Motive-export files (`0010`, `0011`, `0018`, `0021`, and
`003`--`009` among them) lack the complete matching recorder/annotation set and
are excluded. `trial_009` additionally has 100% missing wrist XYZ. The frozen
dependency's `external/exact_sew/data/test.csv` is a 4,344-row trial-0012
regression fixture, not the Phase 1 dataset, and is excluded from all dataset
statistics.

## Unresolved semantics

Each take is its own session. The frozen metadata mapping is `trial_0015 -> D1`
and all six other included takes -> `D0`; `recipient_id` remains null because
recipient identities are unavailable. Cross-sensor synchronization accuracy,
true mouth pose, plate orientation, anatomical hand pose, and a fork-tip/
food-point transform remain unresolved.
