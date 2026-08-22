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


def _maximum_ASR_residual(fc3: np.ndarray) -> float:
    return max(float(np.max(np.abs(np.sum(fc3, axis=axis)))) for axis in range(3))


@pytest.mark.reference
def test_AlN_FC3_matches_phono3py_with_ASR():
    """Compare strict ASR solutions and verify each translational residual independently."""
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
        reference_raw = data["phono3py_fc3"]
        reference_ASR = data["phono3py_fc3_asr"]
        plan_hash = str(data["mlfcs_plan_hash"])
        cutoff = float(data["cutoff_angstrom"])

    calculation = ForceConstantCalculation(
        unitcell,
        order=3,
        supercell=(2, 2, 2),
        cutoff=cutoff,
        displacement=0.01,
        jax_platform="cpu",
        report_cutoff=False,
    )
    result = calculation.reap(
        forces,
        plan_hash=plan_hash,
        acoustic_sum_rule=True,
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
    expected = hiphive_full_fc3(reference_supercell, reference_ASR)

    assert _maximum_ASR_residual(reference_raw) > 1e-3
    assert _maximum_ASR_residual(actual) < 1e-8
    assert _maximum_ASR_residual(expected) < 1e-8

    difference = actual[support] - expected[support]
    assert np.max(np.abs(difference)) < 2e-2
    assert np.sqrt(np.mean(difference**2)) < 1e-3
    assert np.linalg.norm(difference) / np.linalg.norm(expected[support]) < 6e-4
    assert np.corrcoef(actual[support].ravel(), expected[support].ravel())[0, 1] > 0.999999
