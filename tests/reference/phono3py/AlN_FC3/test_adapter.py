from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.model import SparseOrderForceConstants
from tests.reference.phono3py.AlN_FC3.adapter import (
    full_array_from_sparse,
    hiphive_full_fc3,
    matching_permutation,
)


@pytest.mark.reference
def test_hiphive_adapter_expands_translation_and_matches_atom_order():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    supercell, index = make_supercell(primitive, (2, 1, 1))
    clusters = np.array([(p, j, k) for p in range(2) for j in range(4) for k in range(4)])
    tensors = np.arange(len(clusters) * 27, dtype=float).reshape((-1, 3, 3, 3))
    sparse = SparseOrderForceConstants(3, 2, 4, clusters, tensors)

    full = full_array_from_sparse(sparse, index)
    np.testing.assert_allclose(full[0], sparse.to_dense(max_bytes=None)[0])
    np.testing.assert_allclose(full[2, 0, 1], sparse.to_dense(max_bytes=None)[0, 2, 3])

    grouped = index.group_atoms(supercell)
    permutation = matching_permutation(supercell, grouped)
    np.testing.assert_array_equal(permutation, index.grouped_permutation)
    converted = hiphive_full_fc3(supercell, full, target_supercell=grouped)
    np.testing.assert_allclose(
        converted,
        full[np.ix_(permutation, permutation, permutation)],
    )
