from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiscale_contact.types import MesoResponse, MesoState


@dataclass(slots=True)
class RoughnessConfig:
    nx: int = 32
    ny: int = 32
    rms_height: float = 1e-6
    correlation_length: float = 5e-6
    seed: int = 1


@dataclass(slots=True)
class MesoTileModel:
    config: RoughnessConfig
    mu0: float = 0.35
    a_state: float = 0.012
    b_state: float = 0.018
    v0: float = 1e-5
    d_c: float = 2e-6

    def generate_heightfield(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        z = rng.normal(0.0, self.config.rms_height, size=(self.config.nx, self.config.ny))
        fx = np.fft.fftfreq(self.config.nx)
        fy = np.fft.fftfreq(self.config.ny)
        kx, ky = np.meshgrid(fx, fy, indexing="ij")
        k2 = kx**2 + ky**2
        filt = np.exp(-k2 * (self.config.correlation_length * 1e6))
        zh = np.fft.ifft2(np.fft.fft2(z) * filt).real
        zh -= float(np.mean(zh))
        return zh

    def microcontact_fraction(self, pressure: float) -> float:
        p_ref = 1e9 * self.config.rms_height
        frac = 1.0 - np.exp(-max(pressure, 0.0) / max(p_ref, 1e-6))
        return float(np.clip(frac, 0.0, 1.0))

    def update(
        self,
        pressure: float,
        slip_rate: float,
        temperature: float,
        dt: float,
        state: MesoState,
        nano_mu_shift: float = 0.0,
        nano_adhesion: float = 0.0,
    ) -> MesoResponse:
        v_eff = max(abs(slip_rate), 1e-12)
        theta_dot = 1.0 - (v_eff * state.theta / self.d_c)
        theta_new = max(1e-9, state.theta + dt * theta_dot)
        contact_fraction = self.microcontact_fraction(pressure + nano_adhesion)
        log_v = np.log(v_eff / self.v0)
        log_theta = np.log((theta_new * self.v0) / self.d_c)
        mu = self.mu0 + self.a_state * log_v + self.b_state * log_theta + nano_mu_shift
        temp_softening = np.exp(-(temperature - 300.0) / 500.0)
        mu_eff = float(max(0.02, mu * temp_softening) * (0.4 + 0.6 * contact_fraction))
        tau = mu_eff * pressure
        heat_partition = float(np.clip(0.5 + 0.1 * np.tanh((temperature - 300.0) / 50.0), 0.1, 0.9))
        updated = MesoState(theta=theta_new, contact_fraction=contact_fraction, mu_eff=mu_eff)
        return MesoResponse(
            tau=tau,
            mu_eff=mu_eff,
            heat_partition_to_ball=heat_partition,
            updated_state=updated,
        )
