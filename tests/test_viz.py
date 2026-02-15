from pathlib import Path

import numpy as np
from multiscale_contact.types import FieldSample
from multiscale_contact.viz import plot_multiscale_state


def test_plot_multiscale_state_writes_file(tmp_path: Path) -> None:
    x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
    f = FieldSample(
        x=x,
        y=y,
        p=np.maximum(0.0, 1.0 - x**2 - y**2),
        slip_rate=np.ones_like(x),
        temperature=300 * np.ones_like(x),
    )
    rough = np.random.default_rng(0).normal(size=(8, 8))
    atoms = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.4]])
    out = tmp_path / "viz.png"
    plot_multiscale_state(f, rough, atoms, str(out))
    assert out.exists()
