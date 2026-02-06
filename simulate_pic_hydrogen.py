#!/usr/bin/env python3
"""Pedagogical 2D electromagnetic PIC-style hydrogen simulation.

This script models one electron orbiting one proton using:
  * A Yee-like grid for self-consistent radiative EM fields (E_x, E_y, B_z)
  * Cloud-in-cell (CIC) current deposition and field interpolation
  * A Boris pusher for particle motion
  * Direct Coulomb attraction between electron and proton (near field)
  * Optional Landau-Lifshitz-inspired radiation reaction damping

The goal is educational: in classical electrodynamics, a bound accelerating
charge radiates, loses energy, and spirals inward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    q: float
    m: float
    x: np.ndarray
    v: np.ndarray


def cic_weights(x: np.ndarray, dx: float, nx: int, ny: int):
    """Return CIC indices and weights for 2D periodic grid."""
    gx = (x[0] / dx) % nx
    gy = (x[1] / dx) % ny

    i0 = int(np.floor(gx))
    j0 = int(np.floor(gy))
    fx = gx - i0
    fy = gy - j0

    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny

    inds = [(i0, j0), (i1, j0), (i0, j1), (i1, j1)]
    w = np.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    return inds, w


def deposit_current(jx: np.ndarray, jy: np.ndarray, p: Particle, dx: float, dt: float):
    """Simple first-order current deposition from particle velocity at step n+1/2."""
    inds, w = cic_weights(p.x, dx, *jx.shape)
    area = dx * dx
    for (i, j), wk in zip(inds, w):
        jx[i, j] += p.q * p.v[0] * wk / area
        jy[i, j] += p.q * p.v[1] * wk / area


def interpolate_field(ex: np.ndarray, ey: np.ndarray, bz: np.ndarray, x: np.ndarray, dx: float):
    inds, w = cic_weights(x, dx, *ex.shape)
    ex_p = 0.0
    ey_p = 0.0
    bz_p = 0.0
    for (i, j), wk in zip(inds, w):
        ex_p += wk * ex[i, j]
        ey_p += wk * ey[i, j]
        bz_p += wk * bz[i, j]
    return np.array([ex_p, ey_p]), bz_p


def boris_push(p: Particle, e: np.ndarray, b: float, dt: float):
    """2D Boris pusher with out-of-plane magnetic field b_z."""
    qm = p.q / p.m
    v_minus = p.v + 0.5 * dt * qm * e

    t = 0.5 * dt * qm * b
    s = 2.0 * t / (1.0 + t * t)

    # v' = v- + v- x t_hat, with B = (0,0,b)
    v_prime = np.array([v_minus[0] + v_minus[1] * t, v_minus[1] - v_minus[0] * t])
    # v+ = v- + v' x s_hat
    v_plus = np.array([v_minus[0] + v_prime[1] * s, v_minus[1] - v_prime[0] * s])

    p.v = v_plus + 0.5 * dt * qm * e
    p.x = p.x + dt * p.v


def curl_e(ex: np.ndarray, ey: np.ndarray, dx: float):
    """Return z-component of curl(E) using periodic differences."""
    dey_dx = (np.roll(ey, -1, axis=0) - ey) / dx
    dex_dy = (np.roll(ex, -1, axis=1) - ex) / dx
    return dey_dx - dex_dy


def curl_bz_to_e(bz: np.ndarray, dx: float):
    """Return (dBz/dy, -dBz/dx) terms needed for (Ex,Ey) updates."""
    dbz_dy = (bz - np.roll(bz, 1, axis=1)) / dx
    dbz_dx = (bz - np.roll(bz, 1, axis=0)) / dx
    return dbz_dy, -dbz_dx


def wrap_periodic(x: np.ndarray, lbox: float):
    x[:] = x % lbox


def min_image(r: np.ndarray, lbox: float):
    return r - lbox * np.round(r / lbox)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=128, help="grid size in x and y")
    ap.add_argument("--dx", type=float, default=0.5, help="grid spacing")
    ap.add_argument("--steps", type=int, default=5000, help="number of timesteps")
    ap.add_argument("--dt", type=float, default=0.03, help="timestep")
    ap.add_argument("--r0", type=float, default=8.0, help="initial orbital radius")
    ap.add_argument("--no-rad-reaction", action="store_true", help="disable radiation damping term")
    ap.add_argument("--fixed-proton", action="store_true", default=True, help="keep proton fixed at origin")
    ap.add_argument("--no-plots", action="store_true", help="do not show plots")
    ap.add_argument("--out-prefix", type=str, default="pic_hydrogen", help="prefix for output figures")
    args = ap.parse_args()

    # Normalized units
    eps0 = 1.0
    mu0 = 1.0
    c = 1.0
    k_coul = 1.0 / (4.0 * np.pi * eps0)

    nx = ny = args.nx
    dx = args.dx
    dt = args.dt
    lbox = nx * dx

    # CFL suggestion for 2D FDTD: dt <= dx/(c*sqrt(2))
    cfl = dx / (c * np.sqrt(2.0))
    if dt > cfl:
        print(f"WARNING: dt={dt:.4f} exceeds CFL~{cfl:.4f}; simulation may be unstable.")

    ex = np.zeros((nx, ny), dtype=float)
    ey = np.zeros((nx, ny), dtype=float)
    bz = np.zeros((nx, ny), dtype=float)

    proton = Particle(
        q=+1.0,
        m=1836.0,
        x=np.array([0.5 * lbox, 0.5 * lbox]),
        v=np.zeros(2),
    )

    e_mass = 1.0
    electron = Particle(
        q=-1.0,
        m=e_mass,
        x=np.array([0.5 * lbox + args.r0, 0.5 * lbox]),
        v=np.zeros(2),
    )

    # Circular speed from central Coulomb force
    v_circ = np.sqrt(k_coul * abs(electron.q * proton.q) / (electron.m * args.r0))
    electron.v = np.array([0.0, +v_circ])

    # Nonrelativistic LL timescale tau = q^2/(6*pi*eps0*m*c^3)
    tau_rr = electron.q**2 / (6.0 * np.pi * eps0 * electron.m * c**3)

    t_hist = np.zeros(args.steps)
    r_hist = np.zeros(args.steps)
    ke_hist = np.zeros(args.steps)
    pe_hist = np.zeros(args.steps)
    fe_hist = np.zeros(args.steps)
    exy = np.zeros((args.steps, 2))

    for n in range(args.steps):
        t = n * dt
        t_hist[n] = t

        # Build current from particles
        jx = np.zeros_like(ex)
        jy = np.zeros_like(ey)
        deposit_current(jx, jy, electron, dx, dt)
        if not args.fixed_proton:
            deposit_current(jx, jy, proton, dx, dt)

        # Yee-like leapfrog updates
        bz -= dt * curl_e(ex, ey, dx)
        dbz_dy, minus_dbz_dx = curl_bz_to_e(bz, dx)
        ex += dt * ((1.0 / eps0) * dbz_dy - jx / eps0)
        ey += dt * ((1.0 / eps0) * minus_dbz_dx - jy / eps0)

        # Lorentz field at electron (for radiation visualization, not particle push)
        e_grid, b_grid = interpolate_field(ex, ey, bz, electron.x, dx)

        # Direct Coulomb near-field (proton -> electron)
        rvec = min_image(electron.x - proton.x, lbox)
        r = np.linalg.norm(rvec) + 1e-9
        e_coul = k_coul * proton.q * rvec / (r**3)

        # Use only Coulomb for particle dynamics (avoid numerical self-force from grid)
        # Grid fields still computed for radiation visualization
        e_total = e_coul  # was: e_grid + e_coul
        b_grid = 0.0      # ignore self B-field too

        # Radiation reaction (nonrelativistic LL-inspired dissipation)
        if not args.no_rad_reaction:
            a_lor = (electron.q / electron.m) * (e_total + np.array([electron.v[1] * b_grid, -electron.v[0] * b_grid]))
            v2 = np.dot(electron.v, electron.v) + 1e-12
            a2 = np.dot(a_lor, a_lor)
            a_rr = -tau_rr * a2 * electron.v / v2
            e_total = e_total + (electron.m / electron.q) * a_rr

        boris_push(electron, e_total, b_grid, dt)
        wrap_periodic(electron.x, lbox)

        if not args.fixed_proton:
            e_on_p, b_on_p = interpolate_field(ex, ey, bz, proton.x, dx)
            rvec_pe = min_image(proton.x - electron.x, lbox)
            rp = np.linalg.norm(rvec_pe) + 1e-9
            e_coul_p = k_coul * electron.q * rvec_pe / (rp**3)
            boris_push(proton, e_on_p + e_coul_p, b_on_p, dt)
            wrap_periodic(proton.x, lbox)

        rvec = min_image(electron.x - proton.x, lbox)
        r = np.linalg.norm(rvec)
        r_hist[n] = r
        exy[n] = electron.x.copy()

        ke_hist[n] = 0.5 * electron.m * np.dot(electron.v, electron.v)
        pe_hist[n] = k_coul * electron.q * proton.q / (r + 1e-9)
        fe_hist[n] = 0.5 * eps0 * np.mean(ex**2 + ey**2) + 0.5 / mu0 * np.mean(bz**2)

    print("Done.")
    print(f"Initial radius: {r_hist[0]:.4f}, final radius: {r_hist[-1]:.4f}")

    total_e = ke_hist + pe_hist + fe_hist

    if not args.no_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

        ax[0].plot(exy[:, 0], exy[:, 1], lw=1.2)
        ax[0].plot(proton.x[0], proton.x[1], "ro", ms=6, label="proton")
        ax[0].set_aspect("equal", adjustable="box")
        ax[0].set_title("Electron trajectory")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()

        ax[1].plot(t_hist, r_hist, label="r(t)")
        ax[1].set_title("Orbital radius vs time")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("r")

        ax[2].plot(t_hist, ke_hist, label="K_e")
        ax[2].plot(t_hist, pe_hist, label="U_C")
        ax[2].plot(t_hist, fe_hist, label="U_field")
        ax[2].plot(t_hist, total_e, "k--", label="Total")
        ax[2].set_title("Energy channels")
        ax[2].set_xlabel("t")
        ax[2].legend(fontsize=8)

        plt.tight_layout()
        out_path = f"{args.out_prefix}_summary.png"
        fig.savefig(out_path, dpi=160)
        print(f"Saved {out_path}")
        plt.show()


if __name__ == "__main__":
    main()
