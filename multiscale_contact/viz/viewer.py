from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from multiscale_contact.types import FieldSample


def plot_multiscale_state(
    fields: FieldSample,
    roughness: np.ndarray,
    atom_positions: np.ndarray,
    output_path: str | None = None,
) -> None:
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    c = ax1.pcolormesh(fields.x, fields.y, fields.p, shading="auto")
    fig.colorbar(c, ax=ax1, label="Pressure [Pa]")
    ax1.set_title("Continuum pressure")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(roughness, cmap="terrain")
    ax2.set_title("Meso roughness tile")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if atom_positions.size:
        ax3.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2], s=5)
    ax3.set_title("Nano atoms")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
