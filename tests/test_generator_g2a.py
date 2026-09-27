from pathlib import Path
import numpy as np

from feeding_coordination.generator_g0 import _proper_so3
from feeding_coordination.generator_g2a import (GRID, START_CANDIDATES, constrain_position_endpoint,
    constrain_orientation_endpoint, quintic_smoothstep_first, quintic_smoothstep_second, select_adaptation_start, virtual_transfer_goal_position)
from test_generator_g0 import synthetic_record, training_records


def test_target_uses_proxy_and_nominal_orientation_not_gt_endpoint():
    record = synthetic_record("held_bite_001_transfer", "held")
    nominal_R = record.gt_orientation_B.copy()
    goal = virtual_transfer_goal_position(record, nominal_R)
    changed = synthetic_record("held_bite_001_transfer", "held")
    changed.gt_position_B[-1] += 99
    assert np.array_equal(goal, virtual_transfer_goal_position(changed, nominal_R))


def test_constraint_anchors_start_and_exactly_satisfies_endpoint():
    nominal = np.c_[GRID, GRID**2, np.zeros(101)]
    goal = np.array([.4, -.2, .1])
    constrained, delta = constrain_position_endpoint(nominal, goal, .65)
    assert np.array_equal(constrained[0], nominal[0])
    assert np.array_equal(constrained[-1], goal)
    assert np.array_equal(constrained[GRID < .65], nominal[GRID < .65])
    assert np.array_equal(delta[GRID < .65], np.zeros((sum(GRID < .65), 3)))


def test_quintic_has_zero_first_second_boundary_derivatives():
    assert np.allclose(quintic_smoothstep_first([0., 1.]), 0.)
    assert np.allclose(quintic_smoothstep_second([0., 1.]), 0.)


def test_orientation_is_unmodified_and_remains_so3():
    record = synthetic_record("held_bite_001_transfer", "held")
    _, delta = constrain_position_endpoint(record.gt_position_B, record.mouth_proxy_B, .5)
    assert delta.shape == (101, 3)
    assert _proper_so3(record.gt_orientation_B).all()
    assert np.array_equal(constrain_orientation_endpoint(record.gt_orientation_B, .5, None), record.gt_orientation_B)


def test_start_selection_uses_only_outer_training_takes():
    trace = {}
    outer = training_records()
    chosen, scores = select_adaptation_start(outer, alpha=1., trace=trace)
    assert chosen in START_CANDIDATES and len(scores) == len(START_CANDIDATES)
    assert "held" not in trace["selection_training_takes"]
    assert all("held" not in row["inner_validation_takes"] for row in scores)


def test_source_does_not_import_robot_or_measured_endpoint_target():
    source = Path("src/feeding_coordination/generator_g2a.py").read_text().lower()
    assert "from .robot" not in source and "from .exact_sew" not in source
