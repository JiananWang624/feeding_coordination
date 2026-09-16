from pathlib import Path

import numpy as np

from feeding_coordination.generator_g0 import (
    GENERATORS,
    GRID,
    G0Record,
    _proper_so3,
    generated_u_contract_valid,
    load_phase15_records,
    choose_promp_alpha,
    promp_generate,
    retrieval_generate,
    slerp_rotations,
    training_phase_median_duration,
    validate_generated,
)


def synthetic_record(record_id: str, take: str, phase: str = "transfer", offset: float = 0.0,
                     duration: float = 1.0) -> G0Record:
    p0 = np.array([offset, 0.0, 0.0])
    R0 = np.eye(3)
    rel = np.column_stack([0.1 * GRID, 0.02 * GRID ** 2, np.zeros(101)])
    angles = 0.2 * GRID
    relative_R = np.zeros((101, 3, 3))
    relative_R[:, 0, 0] = np.cos(angles); relative_R[:, 0, 1] = -np.sin(angles)
    relative_R[:, 1, 0] = np.sin(angles); relative_R[:, 1, 1] = np.cos(angles)
    relative_R[:, 2, 2] = 1.0
    rotvec = np.column_stack([np.zeros(101), np.zeros(101), angles])
    plate = p0 + np.array([0.2 + offset, -0.1, 0.0])
    mouth = p0 + np.array([0.4 + offset, 0.1, 0.1])
    return G0Record(
        record_id, record_id.rsplit("_", 1)[0], take, phase, p0, R0, plate, mouth,
        np.r_[plate - p0, mouth - p0], duration, GRID.copy(), np.c_[rel, rotvec],
        rel, relative_R, p0 + rel, relative_R, {},
    )


def training_records(phase: str = "transfer") -> list[G0Record]:
    return [synthetic_record(f"take_{i}_bite_001_{phase}", f"take_{i}", phase, .01 * i, 1 + .1 * i) for i in range(6)]


def test_g0_reads_phase15_only(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema_version":"phase2-v1","records":[]}')
    try:
        load_phase15_records(tmp_path)
    except ValueError as error:
        assert "Phase 1.5 only" in str(error)
    else:
        raise AssertionError("non-Phase-1.5 input must be rejected")


def test_mouth_target_is_only_an_untrusted_mouth_proxy_label():
    source = Path("src/feeding_coordination/generator_g0.py").read_text()
    assert '"mouth_proxy_is_independently_measured": False' in source
    assert '"deployment_context_validated": False' in source
    assert "true_mouth" not in source and "measured_mouth" not in source


def test_outer_held_take_cannot_enter_retrieval_pool_or_scaler():
    query = synthetic_record("held_bite_001_transfer", "held")
    trace = {}
    retrieval_generate(query, training_records(), trace)
    event = trace["retrieval_events"][0]
    assert "held" not in event["candidate_takes"]
    assert "held" not in event["scaler_takes"]


def test_outer_held_take_cannot_enter_promp_fit_scaler_or_alpha_selection():
    query = synthetic_record("held_bite_001_transfer", "held")
    trace = {}
    promp_generate(query, training_records(), alpha=1.0, trace=trace)
    choose_promp_alpha(training_records(), (0.1, 1.0), trace=trace)
    assert trace["promp_events"]
    assert all("held" not in event["takes"] for event in trace["promp_events"])


def test_retrieval_candidates_are_same_phase():
    query = synthetic_record("held_bite_001_withdrawal", "held", "withdrawal")
    mixed = training_records("withdrawal") + training_records("transfer")
    trace = {}
    retrieval_generate(query, mixed, trace)
    assert trace["retrieval_events"][0]["candidate_phases"] == ["withdrawal"]


def test_endpoint_adaptation_leaves_initial_position_unchanged():
    query = synthetic_record("held_bite_001_transfer", "held", offset=.5)
    position, _, _ = retrieval_generate(query, training_records())
    assert np.allclose(position[0], query.p0, atol=1e-12, rtol=0)


def test_rotation_interpolation_is_so3_slerp_not_euler():
    source = Path("src/feeding_coordination/generator_g0.py").read_text()
    assert "Slerp(" in source and "from_euler" not in source
    matrices = synthetic_record("a_b_transfer", "a").relative_orientation_grid
    interpolated = slerp_rotations(GRID, matrices, np.linspace(0, 1, 17))
    assert _proper_so3(interpolated).all()


def test_promp_initial_se3_pose_is_exactly_anchored():
    query = synthetic_record("held_bite_001_transfer", "held", offset=.3)
    position, orientation, _ = promp_generate(query, training_records(), alpha=1.0)
    assert np.allclose(position[0], query.p0, atol=1e-12, rtol=0)
    assert np.allclose(orientation[0], query.R0, atol=1e-12, rtol=0)


def test_generated_orientations_are_proper_so3():
    query = synthetic_record("held_bite_001_transfer", "held")
    for generator in GENERATORS:
        if generator == "retrieval":
            _, orientation, _ = retrieval_generate(query, training_records())
        else:
            _, orientation, _ = promp_generate(query, training_records(), alpha=1.0)
        assert _proper_so3(orientation).all()


def test_both_methods_output_exactly_101_samples():
    query = synthetic_record("held_bite_001_transfer", "held")
    outputs = [retrieval_generate(query, training_records())[:2],
               promp_generate(query, training_records(), alpha=1.0)[:2]]
    for position, orientation in outputs:
        assert position.shape == (101, 3) and orientation.shape == (101, 3, 3)
        assert validate_generated(position, orientation, query) == (True, "SUCCESS")


def test_duration_uses_outer_training_phase_only():
    training = training_records()
    query = synthetic_record("held_bite_001_transfer", "held", duration=999.0)
    expected = np.median([record.duration_s for record in training])
    assert training_phase_median_duration(training, query.phase) == expected
    assert training_phase_median_duration(training, query.phase) != query.duration_s


def test_generated_u_artifact_satisfies_stronglocal_input_contract():
    query = synthetic_record("held_bite_001_transfer", "held")
    arrays = {
        "normalized_phase": GRID, "time_s": GRID,
        "generated_tool_position_B": query.gt_position_B,
        "generated_tool_orientation_B": query.gt_orientation_B,
        "initial_tool_position_B": query.p0, "initial_tool_orientation_B": query.R0,
        "phase": np.array(query.phase), "plate_position_B": query.plate_B,
        "plate_position_valid": np.array(True), "generation_status": np.array("SUCCESS"),
    }
    assert generated_u_contract_valid(arrays)


def test_no_psi_robot_q_exact_sew_or_robot_metric_is_a_generator_feature():
    source = Path("src/feeding_coordination/generator_g0.py").read_text()
    assert 'z["psi' not in source
    assert "feeding_coordination.robot" not in source
    assert "feeding_coordination.phase2" not in source
    assert "feeding_coordination.phase3" not in source
    assert "exact_sew import" not in source
