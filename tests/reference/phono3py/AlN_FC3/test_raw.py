from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlfcs import ForceConstantCalculation
from tests.reference.phono3py.AlN_FC3.adapter import (
    full_array_from_sparse,
    full_cluster_mask,
    hiphive_full_fc3,
    matching_permutation,
)

DATA = Path(__file__).parent / "data" / "reference.npz"


@pytest.mark.reference
def test_AlN_FC3_matches_phono3py_without_ASR():
    """Compare independent finite differences in one full-supercell representation."""
    with np.load(DATA) as data:
        unitcell = Atoms(
            numbers=data["unitcell_numbers"],
            cell=data["unitcell_cell"],
            scaled_positions=data["unitcell_scaled_positions"],
            pbc=True,
        )
        reference_supercell = Atoms(
            numbers=data["phono3py_supercell_numbers"],
            cell=data["phono3py_supercell_cell"],
            scaled_positions=data["phono3py_supercell_scaled_positions"],
            pbc=True,
        )
        forces = data["mlfcs_forces"]
        reference_fc3 = data["phono3py_fc3"]
        cutoff = float(data["cutoff_angstrom"])
        maximum_mic_distance = float(data["maximum_mic_distance_angstrom"])
        cutoff_mode = str(data["cutoff_mode"])
        n_configurations = int(data["phono3py_configurations"])

    calculation = ForceConstantCalculation(
        unitcell,
        order=3,
        supercell=(2, 2, 2),
        cutoff=cutoff,
        displacement=0.01,
        report_cutoff=False,
    )
    assert np.isclose(calculation.cutoff, cutoff, atol=1e-12, rtol=0)
    assert cutoff_mode == "full_supercell"
    assert cutoff > maximum_mic_distance
    assert np.isclose(cutoff - maximum_mic_distance, 1e-6, atol=1e-12, rtol=0)
    assert n_configurations == 968

    result = calculation.reap(
        forces,
        acoustic_sum_rule=False,
    )
    sparse = result.sparse[3]
    full = full_array_from_sparse(sparse, calculation.index)
    permutation = matching_permutation(calculation.supercell, reference_supercell)
    support = full_cluster_mask(sparse, calculation.index)
    support = support[np.ix_(permutation, permutation, permutation)]
    actual = hiphive_full_fc3(
        calculation.supercell,
        full,
        target_supercell=reference_supercell,
    )
    expected = hiphive_full_fc3(reference_supercell, reference_fc3)

    difference = actual[support] - expected[support]
    assert np.max(np.abs(difference)) < 3e-2
    assert np.sqrt(np.mean(difference**2)) < 5e-3
    assert np.linalg.norm(difference) / np.linalg.norm(expected[support]) < 5e-4
