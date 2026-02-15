from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiscale_contact.continuum.hertz import HertzContactModel
from multiscale_contact.meso.tile import MesoTileModel
from multiscale_contact.nano.runner import LJNanoPatchRunner, NanoCache
from multiscale_contact.types import MesoState, NanoBoundaryCondition


@dataclass(slots=True)
class CouplingConfig:
    n_tiles: int = 8
    n_hotspots: int = 2
    dt: float = 1e-3
    iterations: int = 4
    relax: float = 0.4


@dataclass(slots=True)
class CouplingSnapshot:
    iteration: int
    mean_pressure: float
    mean_tau: float
    mean_mu: float
    nano_adhesion: float
    heat_partition: float


@dataclass(slots=True)
class CouplingEngine:
    continuum: HertzContactModel
    meso_model: MesoTileModel
    nano_runner: LJNanoPatchRunner
    cache: NanoCache
    config: CouplingConfig

    def run(self) -> list[CouplingSnapshot]:
        fields = self.continuum.sample_fields(n=96)
        contact_mask = fields.p > 0.0
        cp = fields.p[contact_mask]
        cv = fields.slip_rate[contact_mask]
        ctemp = fields.temperature[contact_mask]
        if cp.size == 0:
            return []

        rng = np.random.default_rng(42)
        picks = rng.choice(cp.size, size=min(self.config.n_tiles, cp.size), replace=False)
        tile_p = cp[picks]
        tile_v = cv[picks]
        tile_t = ctemp[picks]
        states = [MesoState() for _ in range(len(tile_p))]
        nano_mu_shift = np.zeros_like(tile_p)
        nano_adhesion = np.zeros_like(tile_p)
        snapshots: list[CouplingSnapshot] = []

        for it in range(self.config.iterations):
            taus = np.zeros_like(tile_p)
            mus = np.zeros_like(tile_p)
            heat_parts = np.zeros_like(tile_p)
            for i, (p, v, t) in enumerate(zip(tile_p, tile_v, tile_t, strict=True)):
                resp = self.meso_model.update(
                    pressure=float(p),
                    slip_rate=float(v),
                    temperature=float(t),
                    dt=self.config.dt,
                    state=states[i],
                    nano_mu_shift=float(nano_mu_shift[i]),
                    nano_adhesion=float(nano_adhesion[i]),
                )
                states[i] = resp.updated_state
                taus[i] = resp.tau
                mus[i] = resp.mu_eff
                heat_parts[i] = resp.heat_partition_to_ball

            hotspot_idx = np.argsort(tile_p * np.abs(tile_v))[-self.config.n_hotspots :]
            for idx in hotspot_idx:
                bc = NanoBoundaryCondition(
                    normal_pressure=float(tile_p[idx]),
                    slip_rate=float(tile_v[idx]),
                    temperature=float(tile_t[idx]),
                    state_theta=states[idx].theta,
                    adhesion_guess=float(nano_adhesion[idx]),
                )
                nano = self.cache.get(bc)
                if nano is None:
                    nano = self.nano_runner.run(bc)
                    self.cache.put(bc, nano)
                target_mu_shift = float(
                    np.tanh(nano.traction_mean / (abs(tile_p[idx]) + 1e-9)) * 0.08
                )
                nano_mu_shift[idx] = (
                    (1 - self.config.relax) * nano_mu_shift[idx]
                    + self.config.relax * target_mu_shift
                )
                nano_adhesion[idx] = (
                    (1 - self.config.relax) * nano_adhesion[idx]
                    + self.config.relax * nano.adhesion_work
                )

            mean_tau = float(np.mean(taus))
            mean_mu = float(np.mean(mus))
            mean_heat = float(np.mean(heat_parts))
            snapshots.append(
                CouplingSnapshot(
                    iteration=it,
                    mean_pressure=float(np.mean(tile_p)),
                    mean_tau=mean_tau,
                    mean_mu=mean_mu,
                    nano_adhesion=float(np.mean(nano_adhesion)),
                    heat_partition=mean_heat,
                )
            )
        return snapshots
