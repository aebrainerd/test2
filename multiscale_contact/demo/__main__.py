from __future__ import annotations

from pathlib import Path

import numpy as np

from multiscale_contact.continuum import HertzContactModel, MaterialPair, SphereLoadCase
from multiscale_contact.coupling import CouplingConfig, CouplingEngine
from multiscale_contact.meso import MesoTileModel, RoughnessConfig
from multiscale_contact.nano import LJNanoPatchRunner, NanoCache
from multiscale_contact.types import NanoBoundaryCondition
from multiscale_contact.viz import plot_multiscale_state


def main() -> None:
    materials = MaterialPair(e_ball=210e9, nu_ball=0.29, e_table=3.2e9, nu_table=0.35)
    load = SphereLoadCase(
        radius=0.01,
        normal_load=120.0,
        ambient_temperature=303.0,
        sliding_speed=1e-2,
    )
    continuum = HertzContactModel(materials=materials, load_case=load)

    meso = MesoTileModel(config=RoughnessConfig(seed=7))
    nano = LJNanoPatchRunner(n_particles=216, n_steps=250)
    cache = NanoCache()

    engine = CouplingEngine(
        continuum=continuum,
        meso_model=meso,
        nano_runner=nano,
        cache=cache,
        config=CouplingConfig(n_tiles=12, n_hotspots=2, iterations=5),
    )
    snaps = engine.run()
    for snap in snaps:
        print(
            f"iter={snap.iteration} p={snap.mean_pressure:.3e} "
            f"tau={snap.mean_tau:.3e} mu={snap.mean_mu:.3f} "
            f"adh={snap.nano_adhesion:.3e} heat_split={snap.heat_partition:.3f}"
        )

    fields = continuum.sample_fields(n=64)
    roughness = meso.generate_heightfield()
    demo_bc = NanoBoundaryCondition(
        normal_pressure=float(np.max(fields.p)),
        slip_rate=load.sliding_speed,
        temperature=load.ambient_temperature,
        state_theta=1.0,
    )
    _ = nano.run(demo_bc)

    side = 6
    spacing = 1.2
    atom_positions = np.array(
        [
            (i * spacing, j * spacing, k * spacing)
            for i in range(side)
            for j in range(side)
            for k in range(side)
        ],
        dtype=float,
    )
    out_path = Path("demo_multiscale.png")
    plot_multiscale_state(
        fields=fields,
        roughness=roughness,
        atom_positions=atom_positions,
        output_path=str(out_path),
    )
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
