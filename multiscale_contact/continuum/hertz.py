from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiscale_contact.types import FieldSample


@dataclass(slots=True)
class MaterialPair:
    e_ball: float
    nu_ball: float
    e_table: float
    nu_table: float

    @property
    def reduced_modulus(self) -> float:
        inv_e_star = (1.0 - self.nu_ball**2) / self.e_ball + (1.0 - self.nu_table**2) / self.e_table
        return 1.0 / inv_e_star


@dataclass(slots=True)
class SphereLoadCase:
    radius: float
    normal_load: float
    ambient_temperature: float = 300.0
    sliding_speed: float = 1e-3


@dataclass(slots=True)
class HertzContactModel:
    materials: MaterialPair
    load_case: SphereLoadCase

    @property
    def contact_radius(self) -> float:
        r = self.load_case.radius
        w = self.load_case.normal_load
        e_star = self.materials.reduced_modulus
        return ((3.0 * w * r) / (4.0 * e_star)) ** (1.0 / 3.0)

    @property
    def max_pressure(self) -> float:
        a = self.contact_radius
        return 3.0 * self.load_case.normal_load / (2.0 * np.pi * a**2)

    def pressure_at_radius(self, r: np.ndarray) -> np.ndarray:
        a = self.contact_radius
        p0 = self.max_pressure
        inside = np.clip(1.0 - (r / a) ** 2, a_min=0.0, a_max=None)
        return p0 * np.sqrt(inside)

    def sample_fields(self, n: int = 64) -> FieldSample:
        a = self.contact_radius
        xs = np.linspace(-a, a, n)
        ys = np.linspace(-a, a, n)
        xg, yg = np.meshgrid(xs, ys)
        rg = np.sqrt(xg**2 + yg**2)
        p = self.pressure_at_radius(rg)
        mask = rg <= a
        p = np.where(mask, p, 0.0)
        slip = np.where(mask, self.load_case.sliding_speed, 0.0)
        temp = np.where(
            mask,
            self.load_case.ambient_temperature,
            self.load_case.ambient_temperature - 1.0,
        )
        return FieldSample(x=xg, y=yg, p=p, slip_rate=slip, temperature=temp)
