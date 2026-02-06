# Pedagogical PIC-like Hydrogen Simulation (Classical Inspiral)

This repository contains a compact **2D electromagnetic PIC-style** simulation of an electron orbiting a proton in classical electrodynamics.

It is intentionally pedagogical: if radiation reaction is included, the electron loses orbital energy and spirals inward, illustrating why classical mechanics/electrodynamics cannot explain stable atoms.

## Model ingredients

- **Particles:** one electron and one proton.
- **Fields:** grid-based `E_x, E_y, B_z` updated with a Yee-like finite-difference Maxwell step.
- **PIC coupling:**
  - cloud-in-cell (CIC) current deposition from particle velocity,
  - CIC interpolation of fields to particle position.
- **Particle pusher:** Boris integrator.
- **Near field:** direct Coulomb attraction (proton→electron) added explicitly so bound motion is visible.
- **Radiation reaction:** nonrelativistic Landau–Lifshitz-inspired damping term (toggleable).

> This is not a high-fidelity atomic physics solver. It is designed for intuition and classroom-style exploration.

## Run

```bash
python simulate_pic_hydrogen.py --steps 5000
```

Useful options:

```bash
python simulate_pic_hydrogen.py --help
python simulate_pic_hydrogen.py --no-rad-reaction --steps 5000
python simulate_pic_hydrogen.py --steps 8000 --dt 0.02 --r0 10
python simulate_pic_hydrogen.py --no-plots --steps 1000
```

The default plotted summary is saved as `pic_hydrogen_summary.png`.

## What to look for

- With radiation reaction **on** (default), the orbital radius trends downward.
- With radiation reaction **off**, the orbit is much less dissipative.
- Energy plot shows kinetic + potential + field channels and their exchange.

## Notes

- The script uses normalized units (`c=1`, `eps0=1`, `mu0=1`).
- Keep `dt` below the printed CFL suggestion to avoid numerical instability.
