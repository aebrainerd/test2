from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


@dataclass(slots=True)
class FieldSample:
    x: ArrayF
    y: ArrayF
    p: ArrayF
    slip_rate: ArrayF
    temperature: ArrayF


@dataclass(slots=True)
class MesoState:
    theta: float = 1.0
    contact_fraction: float = 0.0
    mu_eff: float = 0.0


@dataclass(slots=True)
class MesoResponse:
    tau: float
    mu_eff: float
    heat_partition_to_ball: float
    updated_state: MesoState


@dataclass(slots=True)
class NanoBoundaryCondition:
    normal_pressure: float
    slip_rate: float
    temperature: float
    state_theta: float
    adhesion_guess: float = 0.0


@dataclass(slots=True)
class NanoResponse:
    traction_mean: float
    traction_std: float
    adhesion_work: float
    thermal_boundary_conductance: float
    metadata: dict[str, float] = field(default_factory=dict)
