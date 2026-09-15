"""Dynamic import boundary for the frozen Exact-SEW implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ExactSewDependency:
    gen3_kinematics: Any
    geometry_type: Any
    stereo_type: Any
    project_reference: Any
    solver_type: Any
    target_type: Any


def load_exact_sew_dependency() -> ExactSewDependency:
    """Load production classes without reimplementing or configuring them."""
    try:
        kinematics = import_module("sew_mimic.kinematics")
        sew = import_module("sew_mimic.sew")
        exact = import_module("sew_mimic.exact")
        common = import_module("sew_mimic.common")
    except ImportError as error:
        raise ImportError(
            "Frozen Exact-SEW is unavailable. Install external/exact_sew from "
            "the repository root with `python -m pip install -e external/exact_sew`."
        ) from error
    return ExactSewDependency(
        gen3_kinematics=kinematics.gen3_kinematics,
        geometry_type=sew.Gen3StereoSewGeometry,
        stereo_type=sew.StereoSew,
        project_reference=sew.project_stereo_sew_reference,
        solver_type=exact.ExactSewSolver,
        target_type=common.ExactSewTarget,
    )

