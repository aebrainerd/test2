import numpy as np
from multiscale_contact.continuum import HertzContactModel, MaterialPair, SphereLoadCase


def test_hertz_fields_are_nonnegative() -> None:
    model = HertzContactModel(
        materials=MaterialPair(e_ball=210e9, nu_ball=0.3, e_table=3e9, nu_table=0.35),
        load_case=SphereLoadCase(radius=0.01, normal_load=50.0),
    )
    fields = model.sample_fields(n=32)
    assert np.all(fields.p >= 0.0)
    assert model.contact_radius > 0.0
    assert model.max_pressure > 0.0
