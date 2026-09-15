# Robot R2

R2 is a read-only analysis layer over Robot R1. It uses stored successful q,
saved statuses, saved continuity labels, frozen mounted FK and the aligned
`pinch_site` Jacobian. It never runs IK, changes a strategy, smooths, retimes,
or calibrates a mount/tool. Supports are A source-valid, B pairwise exact
success, and C pairwise common success split by the union of saved violations.

The seven CSVs contain feasibility, HumanGT consistency, motion, kinematic,
parent-bite bootstrap, per-bite, and per-take results. Single-strategy motion
and kinematic rows are labelled `single_strategy`; their strict direct-comparison
rows are labelled `pairwise_support_c` and expose both A and B values with
common/clean-frame and derivative support counts. Pairwise HumanGT comparisons
use the exact triple A/B/HumanGT success support for wrapped q, psi, elbow, and
arm-plane differences. Aggregate estimates are parent-bite balanced; bootstrap
units are parent bites, not phase records.

`RobotSmooth` branch counts are adjacent successful contiguous transitions of
the saved `search_branch`; all other strategies use saved R0 `branch_changed`.
Jacobians are reported separately for translation and rotation. The collision
audit inspects the frozen model's active collision meshes, explicit contact
pairs/exclusions, and zero-configuration robot contacts. It reports the metric
unavailable unless the capability is actually validated. Human/environment
clearance is excluded. The retiming audit scans the frozen model files and
reports unavailable when authoritative velocity/acceleration limits are absent;
it does not invent limits from the q2/q4/q6 position ranges.

Run `python scripts/run_robot_r2.py` with the project `.venv` Python.
