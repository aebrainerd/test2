from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from multiscale_contact.types import NanoBoundaryCondition, NanoResponse


@dataclass(slots=True)
class NanoCache:
    dp: float = 2e6
    dv: float = 5e-4
    dt: float = 10.0
    dtheta: float = 0.2
    _cache: dict[tuple[int, int, int, int], NanoResponse] = field(default_factory=dict)

    def key(self, bc: NanoBoundaryCondition) -> tuple[int, int, int, int]:
        return (
            int(np.round(bc.normal_pressure / self.dp)),
            int(np.round(bc.slip_rate / self.dv)),
            int(np.round(bc.temperature / self.dt)),
            int(np.round(bc.state_theta / self.dtheta)),
        )

    def get(self, bc: NanoBoundaryCondition) -> NanoResponse | None:
        return self._cache.get(self.key(bc))

    def put(self, bc: NanoBoundaryCondition, response: NanoResponse) -> None:
        self._cache[self.key(bc)] = response


@dataclass(slots=True)
class LJNanoPatchRunner:
    n_particles: int = 216
    n_steps: int = 300
    dt: float = 2e-3
    sigma: float = 1.0
    epsilon: float = 1.0
    mass: float = 1.0
    cutoff: float = 2.5
    seed: int = 4

    def run(self, bc: NanoBoundaryCondition) -> NanoResponse:
        rng = np.random.default_rng(self.seed)
        n = self.n_particles
        nx = int(round(n ** (1.0 / 3.0)))
        n = nx**3
        spacing = 1.2 * self.sigma
        box = nx * spacing
        coords = np.array(
            [
                (i * spacing, j * spacing, k * spacing)
                for i in range(nx)
                for j in range(nx)
                for k in range(nx)
            ],
            dtype=float,
        )
        vel = rng.normal(0.0, np.sqrt(max(bc.temperature, 1.0) / 300.0), size=coords.shape)
        vel[:, 0] += bc.slip_rate
        cutoff2 = (self.cutoff * self.sigma) ** 2
        shear_samples: list[float] = []
        normal_samples: list[float] = []

        def pair_forces(pos: np.ndarray) -> tuple[np.ndarray, float, float]:
            forces = np.zeros_like(pos)
            shear = 0.0
            normal = 0.0
            for i in range(n - 1):
                rij = pos[i + 1 :] - pos[i]
                rij[:, 0] -= box * np.round(rij[:, 0] / box)
                rij[:, 1] -= box * np.round(rij[:, 1] / box)
                r2 = np.sum(rij**2, axis=1)
                mask = (r2 > 1e-12) & (r2 < cutoff2)
                if not np.any(mask):
                    continue
                rij_m = rij[mask]
                r2_m = r2[mask]
                inv_r2 = (self.sigma**2) / r2_m
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                f_mag = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / np.sqrt(r2_m)
                f_vec = (f_mag[:, None] * rij_m) / np.sqrt(r2_m)[:, None]
                forces[i] -= np.sum(f_vec, axis=0)
                forces[i + 1 :][mask] += f_vec
                shear += float(np.sum(f_vec[:, 0] * rij_m[:, 2]))
                normal += float(np.sum(f_vec[:, 2] * rij_m[:, 2]))
            return forces, shear, normal

        # simple top-layer driving and bottom anchoring
        z = coords[:, 2]
        z_hi = np.percentile(z, 90)
        z_lo = np.percentile(z, 10)
        top = z >= z_hi
        bottom = z <= z_lo

        force, _, _ = pair_forces(coords)
        for step in range(self.n_steps):
            vel += 0.5 * self.dt * force / self.mass
            coords += self.dt * vel
            coords[:, 0] %= box
            coords[:, 1] %= box
            vel[top, 0] = bc.slip_rate
            vel[bottom] = 0.0
            coords[bottom, 2] = z[bottom]
            coords[top, 2] = z[top] + 0.01 * np.sin(step * self.dt)
            force, shear, normal = pair_forces(coords)
            confine = bc.normal_pressure * (coords[:, 2] - np.mean(coords[:, 2]))
            force[:, 2] -= confine * 1e-4
            vel += 0.5 * self.dt * force / self.mass
            if step > self.n_steps // 2:
                area = box**2
                shear_samples.append(shear / area)
                normal_samples.append(normal / area + bc.normal_pressure)

        shear_arr = np.array(shear_samples)
        normal_arr = np.array(normal_samples)
        traction = float(np.mean(shear_arr)) if shear_arr.size else 0.0
        traction_std = float(np.std(shear_arr)) if shear_arr.size else 0.0
        mean_normal = (
            float(np.mean(normal_arr)) if normal_arr.size else max(bc.normal_pressure, 1.0)
        )
        adhesion = float(max(0.0, 0.02 * self.epsilon - 1e-10 * bc.temperature))
        tbc = float(1e7 * (1.0 + 0.1 * np.tanh(abs(bc.slip_rate) * 1e3)))
        return NanoResponse(
            traction_mean=traction,
            traction_std=traction_std,
            adhesion_work=adhesion,
            thermal_boundary_conductance=tbc,
            metadata={"normal_mean": mean_normal, "n_particles": float(n)},
        )
