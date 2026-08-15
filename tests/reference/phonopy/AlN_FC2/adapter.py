"""Conversions used only by the independent FC2 reference test."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from mlfcs.core.geometry import PeriodicIndex
from mlfcs.model import SparseOrderForceConstants


def full_fc2(values: SparseOrderForceConstants, index: PeriodicIndex) -> np.ndarray:
    """Expand translation-reduced MLFCS FC2 to a full supercell array."""
    compact = values.to_dense(max_bytes=None)
    result = np.empty((values.n_supercell, values.n_supercell, 3, 3), dtype=compact.dtype)
    for first in range(values.n_supercell):
        shift = -index.translations[first]
        anchored = np.fromiter(
            (index.translate_atom(atom, shift) for atom in range(values.n_supercell)),
            dtype=np.int64,
            count=values.n_supercell,
        )
        result[first] = compact[int(index.primitive[first]), anchored]
    return result


def matching_permutation(source: Atoms, target: Atoms, *, tolerance: float = 1e-6) -> np.ndarray:
    """Return source indices ordered like target using species-aware MIC matching."""
    source_scaled = source.get_scaled_positions(wrap=True)
    target_scaled = target.get_scaled_positions(wrap=True)
    delta = source_scaled[:, None, :] - target_scaled[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ np.asarray(source.cell), axis=-1)
    distances[source.numbers[:, None] != target.numbers[None, :]] = np.inf
    rows, columns = linear_sum_assignment(distances)
    permutation = np.empty(len(source), dtype=np.int64)
    permutation[columns] = rows
    residual = distances[permutation, np.arange(len(source))]
    if float(residual.max()) > tolerance:
        raise ValueError(f"atom-order match residual {float(residual.max()):.3e} exceeds tolerance")
    return permutation
