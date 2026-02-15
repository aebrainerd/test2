from multiscale_contact.meso import MesoTileModel, RoughnessConfig
from multiscale_contact.types import MesoState


def test_meso_update_stable() -> None:
    model = MesoTileModel(config=RoughnessConfig(seed=11))
    state = MesoState(theta=1.0)
    response = model.update(
        pressure=2e6,
        slip_rate=1e-3,
        temperature=305.0,
        dt=1e-3,
        state=state,
    )
    assert response.mu_eff > 0.0
    assert response.tau >= 0.0
    assert 0.0 <= response.updated_state.contact_fraction <= 1.0
