from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multiscale_contact.types import FieldSample


class ContinuumSolver(Protocol):
    def sample_fields(self, n: int = 64) -> FieldSample:
        """Return continuum fields over a contact patch."""


@dataclass(slots=True)
class FEMSolverStub:
    backend_name: str = "fenicsx"

    def sample_fields(self, n: int = 64) -> FieldSample:
        raise NotImplementedError(
            f"FEM backend '{self.backend_name}' is not wired yet. "
            "Use HertzContactModel for now."
        )
