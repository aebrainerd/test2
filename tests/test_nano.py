from multiscale_contact.nano import LJNanoPatchRunner
from multiscale_contact.types import NanoBoundaryCondition


def test_nano_runner_deterministic_seed() -> None:
    runner = LJNanoPatchRunner(n_particles=125, n_steps=120, seed=9)
    bc = NanoBoundaryCondition(
        normal_pressure=2e6,
        slip_rate=1e-3,
        temperature=300.0,
        state_theta=1.2,
    )
    a = runner.run(bc)
    b = runner.run(bc)
    assert abs(a.traction_mean - b.traction_mean) < 1e-12
    assert a.thermal_boundary_conductance > 0.0
