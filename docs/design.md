# Coupling design: concurrent overlap/handshake strategy

## Scope

The starter implementation targets a physically-minded architecture with conservative intent, while using toy models for speed.

## Domains and overlap

- **Continuum domain**: full contact patch from Hertz model.
- **Meso tiles**: selected points/patches inside continuum contact where roughness-mediated constitutive effects are needed.
- **Nano subtiles**: hotspot subset of meso tiles where atomistic corrections are sampled.

Overlap/handshake regions are conceptualized as:
1. Continuum field sampling points collocated with meso tile centers.
2. Nano patch boundary conditions inherited from meso state at hotspots.

## Exchange variables

### Continuum -> Meso
- Normal pressure `p`
- Slip rate `v`
- Temperature `T`

### Meso -> Nano
- Local `(p, v, T, theta)` as atomistic boundary condition
- Prior adhesion estimate for warm-start

### Nano -> Meso
- Mean traction correction / uncertainty
- Adhesion/work of separation estimate
- Thermal boundary conductance estimate

### Meso -> Continuum (closure)
- Effective friction/traction law summaries
- Heat flux partition
- State-dependent constitutive updates

## Conservation-oriented assumptions

1. **No load double counting**: normal load imposed by continuum; lower scales alter constitutive closure only.
2. **Energy split accounting**: frictional work `tau * v` is partitioned by meso heat split, with nano conductance used as modifier (future).
3. **Relaxed blending**: nano-informed corrections are under-relaxed to avoid oscillations and ghost-force style inconsistency near overlap boundaries.

## Current approximations

- Meso contact fraction and GW response are surrogates, not full asperity populations.
- Nano model is reduced LJ slab with simplistic driving/thermostat assumptions.
- Continuum update is not yet a full field re-solve from heterogeneous meso closure.

## Planned upgrades

- Mortar/projection operators for conservative field transfer.
- Explicit momentum and heat residual checks each coupling iteration.
- FEM backend for continuum and LAMMPS/ASE backend for atomistics via adapter interfaces.
