# multiscale-contact

Starter codebase for a **zoomable multiscale contact simulation** of a metal ball resting/sliding on a plastic table.

## Model overview

The package implements three concurrent levels with explicit coupling hooks:

1. **Continuum (`continuum/`)**
   - Hertz sphere-on-half-space analytic contact.
   - Produces `p(x)`, slip-rate field `v(x)`, and temperature field `T(x)`.
   - Includes an abstract/stub path for FEM solver replacement later.

2. **Mesoscale (`meso/`)**
   - Roughness tile represented by Gaussian heightfield.
   - Greenwood–Williamson-style contact fraction surrogate from pressure.
   - Diet rate-and-state friction state variable `theta`.
   - Returns shear traction, effective friction, heat partition, and updated state.

3. **Nanoscale (`nano/`)**
   - Minimal Lennard-Jones MD patch runner.
   - Applies boundary forcing using pressure/slip/temperature from mesoscale hotspot.
   - Returns average traction, variance, adhesion work estimate, thermal boundary conductance.
   - Wrapped by cache keyed by binned `(p, v, T, theta)` for speed.

## Coupling strategy

`coupling/engine.py` runs fixed-point iterations:

```text
continuum -> meso -> nano -> meso -> continuum-facing closure summary
```

Key closure channels:
- nano -> meso: friction shift + adhesion work
- meso -> continuum: tile-averaged traction and heat split summaries

Design assumptions for no ghost-force style artifacts:
- Continuum load is primary normal-force budget.
- Meso/nano closures modify constitutive response (traction/adhesion), not duplicate externally applied loads.
- Blending/relaxation is used when injecting nano corrections.

See `docs/design.md` for details.

## Installation

This project uses **PEP 621 `pyproject.toml`** with Hatchling.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional integrations (future):
- `pip install -e .[fenicsx]`
- `pip install -e .[lammps]`
- `pip install -e .[viz]`

## Run demo

```bash
python -m multiscale_contact.demo
```

This prints coupling iteration summaries and writes `demo_multiscale.png`.

## Testing and quality

```bash
pytest
ruff check .
mypy
```

## Extending to real FEM/MD

- Replace `FEMSolverStub` with a Fenicsx-backed solver implementing `sample_fields`.
- Replace `LJNanoPatchRunner` with a backend adapter (LAMMPS/ASE) preserving `run(NanoBoundaryCondition)->NanoResponse`.
- Add richer energy accounting and conservative projection operators in `coupling/`.
