from multiscale_contact.continuum import HertzContactModel, MaterialPair, SphereLoadCase
from multiscale_contact.coupling import CouplingConfig, CouplingEngine
from multiscale_contact.meso import MesoTileModel, RoughnessConfig
from multiscale_contact.nano import LJNanoPatchRunner, NanoCache


def test_coupling_runs_iterations() -> None:
    engine = CouplingEngine(
        continuum=HertzContactModel(
            materials=MaterialPair(e_ball=210e9, nu_ball=0.29, e_table=3e9, nu_table=0.35),
            load_case=SphereLoadCase(radius=0.01, normal_load=80.0),
        ),
        meso_model=MesoTileModel(config=RoughnessConfig(seed=3)),
        nano_runner=LJNanoPatchRunner(n_particles=125, n_steps=100),
        cache=NanoCache(),
        config=CouplingConfig(iterations=3, n_tiles=6, n_hotspots=1),
    )
    out = engine.run()
    assert len(out) == 3
    assert all(s.mean_tau >= 0.0 for s in out)
