"""Validation-only conversions between compact MLFCS and full FC arrays.

This module deliberately lives in the test suite.  hiphive is an independent
development oracle and is not part of the MLFCS runtime implementation.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from hiphive import ForceConstants as HiphiveForceConstants
from scipy.optimize import linear_sum_assignment

from mlfcs.core.geometry import PeriodicIndex
from mlfcs.model import SparseOrderForceConstants


def full_array_from_sparse(
    values: SparseOrderForceConstants,
    index: PeriodicIndex,
) -> np.ndarray:
    """Expand translation-reduced MLFCS data to hiphive's full representation."""
    compact = values.to_dense(max_bytes=None)
    n_atoms = values.n_supercell
    shape = (n_atoms,) * values.order + (3,) * values.order
    full = np.empty(shape, dtype=compact.dtype)
    for first in range(n_atoms):
        shift = -index.translations[first]
        anchored = [
            np.fromiter(
                (index.translate_atom(atom, shift) for atom in range(n_atoms)),
                dtype=np.int64,
                count=n_atoms,
            )
            for _ in range(values.order - 1)
        ]
        full[(first, *([slice(None)] * (values.order - 1)))] = compact[
            (int(index.primitive[first]), *np.ix_(*anchored))
        ]
    return full


def full_cluster_mask(values: SparseOrderForceConstants, index: PeriodicIndex) -> np.ndarray:
    """Return the full translation-expanded atomic-cluster support."""
    block_shape = (3,) * values.order
    support = SparseOrderForceConstants(
        order=values.order,
        n_primitive=values.n_primitive,
        n_supercell=values.n_supercell,
        clusters=values.clusters,
        tensors=np.ones((len(values.clusters),) + block_shape),
    )
    expanded = full_array_from_sparse(support, index)
    axes = tuple(range(values.order, 2 * values.order))
    return np.any(expanded != 0, axis=axes)


def matching_permutation(source: Atoms, target: Atoms, *, tolerance: float = 1e-6) -> np.ndarray:
    """Return source indices ordered like target using species-aware MIC matching."""
    if len(source) != len(target):
        raise ValueError("structures contain different numbers of atoms")
    if not np.allclose(source.cell, target.cell, atol=tolerance, rtol=0):
        raise ValueError("structures use different cells")

    source_scaled = source.get_scaled_positions(wrap=True)
    target_scaled = target.get_scaled_positions(wrap=True)
    delta = source_scaled[:, None, :] - target_scaled[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ np.asarray(source.cell), axis=-1)
    species_mismatch = source.numbers[:, None] != target.numbers[None, :]
    distances[species_mismatch] = np.inf
    rows, columns = linear_sum_assignment(distances)
    permutation = np.empty(len(source), dtype=np.int64)
    permutation[columns] = rows
    residual = distances[permutation, np.arange(len(source))]
    if not np.all(np.isfinite(residual)) or float(residual.max()) > tolerance:
        raise ValueError(f"atom-order match residual {float(residual.max()):.3e} exceeds tolerance")
    return permutation


def hiphive_full_fc3(
    supercell: Atoms,
    fc3: np.ndarray,
    *,
    target_supercell: Atoms | None = None,
) -> np.ndarray:
    """Canonicalize a full FC3 array with hiphive and optionally reorder it."""
    canonical = HiphiveForceConstants.from_arrays(supercell, fc3_array=fc3).get_fc_array(3)
    if target_supercell is None:
        return canonical
    permutation = matching_permutation(supercell, target_supercell)
    return canonical[np.ix_(permutation, permutation, permutation)]
