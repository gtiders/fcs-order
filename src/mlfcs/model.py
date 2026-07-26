from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration in ASE units (angstrom and eV)."""

    order: int
    supercell: tuple[int, int, int] = (2, 2, 2)
    cutoff: float | int = -5
    displacement: float = 0.01
    symprec: float = 1e-5

    def __post_init__(self) -> None:
        if self.order < 3:
            raise ValueError("order must be at least 3")
        if any(n < 1 for n in self.supercell):
            raise ValueError("supercell multipliers must be positive")
        if self.cutoff == 0:
            raise ValueError("cutoff cannot be zero")
        if self.displacement <= 0:
            raise ValueError("displacement must be positive")


@dataclass(slots=True)
class ForceConstants:
    """Compact force constants and the supercell defining their indices."""

    arrays: dict[int, np.ndarray]
    supercell: Atoms
    metadata: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, order: int) -> np.ndarray:
        return self.arrays[order]

    def write(
        self,
        target: str | Path,
        *,
        format: str,
        order: int | None = None,
    ) -> None:
        from mlfcs.io import write_force_constants

        write_force_constants(self, target, format=format, order=order)
