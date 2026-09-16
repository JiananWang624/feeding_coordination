# Robot R3

Robot R3 tests the frozen redundancy evolutions at initial offsets −0.25, 0,
and +0.25 rad. It consumes Robot R0/R1 source runs, desired virtual-utensil
targets, per-take mounting, and anchored OOF delta-psi arrays without rebuilding
the task or retraining a predictor.

Each non-RobotSmooth run/strategy/condition gets a fresh stateful Exact-SEW
solver. An infeasible first target terminates that run immediately. All
strategies share and verify the same first target, initial status, and (when
successful) numerical q. RobotSmooth starts from that newly solved shifted q
and calls the unchanged Robot R1 causal search.

Continuity uses the frozen 0.5 rad threshold. Motion is computed only inside
continuity-clean intervals. HumanGT comparisons use shifted HumanGT under the
same condition; paired statistics use triple A/B/HumanGT support and parent
bites as bootstrap units. No calibration, smoothing, retiming, collision model,
learning model, or physical safety claim is added.

Run `python scripts/run_robot_r3.py` with the project `.venv` Python.
