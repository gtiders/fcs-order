from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from hiphive import ForceConstants as HiphiveForceConstants

from mlfcs.model import ForceConstants
from tests.reference.phono3py.K4As4Pt2_FC23.case import calculation_and_reference


def _phonopy_grouped_permutation(index):
    return np.concatenate(
        [np.flatnonzero(index.primitive == site) for site in range(index.n_primitive)]
    )


@pytest.mark.reference
def test_K4As4Pt2_raw_FC3_shengbte_roundtrip_is_faithful(tmp_path):
    data, calculation, sparse = calculation_and_reference(3)
    result = ForceConstants(
        {},
        calculation.supercell,
        metadata={"cutoff_angstrom": calculation.cutoff, "acoustic_sum_rule": False},
        sparse={3: sparse},
    )
    target = tmp_path / "FORCE_CONSTANTS_3RD"
    result.write(target, format="shengbte")

    permutation = _phonopy_grouped_permutation(calculation.index)
    grouped_supercell = calculation.supercell[permutation]
    primitive = Atoms(
        numbers=data["unitcell_numbers"],
        cell=data["unitcell_cell"],
        scaled_positions=data["unitcell_scaled_positions"],
        pbc=True,
    )
    returned = HiphiveForceConstants.read_shengBTE(
        grouped_supercell,
        target,
        primitive,
    ).get_fc_array(3)

    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    expected = sparse.to_dense(max_bytes=None)[tuple(sparse.clusters.T)]
    actual = returned[tuple(inverse[sparse.clusters].T)]
    difference = actual - expected

    with target.open() as handle:
        assert int(handle.readline()) == np.count_nonzero(sparse.support)
    assert np.max(np.abs(difference)) < 5e-10
    assert np.sqrt(np.mean(difference**2)) < 2e-12
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 1e-11
