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
class SparseOrderForceConstants:
    """Force constants stored only for symmetry-generated cluster images."""

    order: int
    n_primitive: int
    n_supercell: int
    clusters: np.ndarray
    tensors: np.ndarray

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return (self.n_primitive,) + (self.n_supercell,) * (self.order - 1) + (3,) * self.order

    @property
    def dense_nbytes(self) -> int:
        return int(np.prod(self.dense_shape, dtype=np.int64)) * self.tensors.dtype.itemsize

    def to_dense(self, *, max_bytes: int | None = 2_000_000_000) -> np.ndarray:
        """Materialize the compact tensor after an explicit memory-budget check."""
        if max_bytes is not None and self.dense_nbytes > max_bytes:
            gib = self.dense_nbytes / 1024**3
            raise MemoryError(
                f"dense order-{self.order} force constants require {gib:.2f} GiB; "
                "write HDF5 directly or increase max_bytes explicitly"
            )
        result = np.zeros(self.dense_shape, dtype=self.tensors.dtype)
        counts = np.zeros(self.dense_shape[: self.order], dtype=np.int16)
        for cluster, tensor in zip(self.clusters, self.tensors, strict=True):
            key = tuple(int(atom) for atom in cluster)
            result[key] += tensor
            counts[key] += 1
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero].reshape((-1,) + (1,) * self.order)
        return result


@dataclass(slots=True)
class ForceConstants:
    """Compact force constants and the supercell defining their indices."""

    arrays: dict[int, np.ndarray]
    supercell: Atoms
    metadata: dict[str, object] = field(default_factory=dict)
    sparse: dict[int, SparseOrderForceConstants] = field(default_factory=dict)

    def __getitem__(self, order: int) -> np.ndarray:
        if order not in self.arrays:
            self.arrays[order] = self.sparse[order].to_dense()
        return self.arrays[order]

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.arrays.keys() | self.sparse.keys()))

    def materialize(self, order: int, *, max_bytes: int | None = 2_000_000_000) -> np.ndarray:
        if order in self.arrays:
            return self.arrays[order]
        values = self.sparse[order].to_dense(max_bytes=max_bytes)
        self.arrays[order] = values
        return values

    def write(
        self,
        target: str | Path,
        *,
        format: str,
        order: int | None = None,
    ) -> None:
        from mlfcs.io import write_force_constants

        write_force_constants(self, target, format=format, order=order)
