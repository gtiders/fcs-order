from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlfcs import ForceConstantCalculation
from tests.reference.phonopy.AlN_FC2.adapter import full_fc2, matching_permutation

DATA = Path(__file__).parent / "data" / "reference.npz"


def _asr_residual(fc2: np.ndarray) -> float:
    return float(np.max(np.abs(np.sum(fc2, axis=1))))


def _load():
    data = np.load(DATA)
    unitcell = Atoms(
        numbers=data["unitcell_numbers"],
        cell=data["unitcell_cell"],
        scaled_positions=data["unitcell_scaled_positions"],
        pbc=True,
    )
    reference_supercell = Atoms(
        numbers=data["phonopy_supercell_numbers"],
        cell=data["phonopy_supercell_cell"],
        scaled_positions=data["phonopy_supercell_scaled_positions"],
        pbc=True,
    )
    return data, unitcell, reference_supercell


def _calculate(data, unitcell: Atoms, reference_supercell: Atoms, *, acoustic_sum_rule: bool):
    calculation = ForceConstantCalculation(
        unitcell,
        order=2,
        supercell=(2, 2, 2),
        cutoff=float(data["cutoff_angstrom"]),
        displacement=float(data["displacement_angstrom"]),
        jax_platform="cpu",
        report_cutoff=False,
    )
    result = calculation.reap(
        data["mlfcs_forces"],
        plan_hash=str(data["mlfcs_plan_hash"]),
        acoustic_sum_rule=acoustic_sum_rule,
    )
    actual = full_fc2(result.sparse[2], calculation.index)
    permutation = matching_permutation(calculation.supercell, reference_supercell)
    return actual[np.ix_(permutation, permutation)]


@pytest.mark.reference
def test_AlN_FC2_matches_phonopy_without_ASR():
    data, unitcell, reference_supercell = _load()
    maximum_mic_distance = float(data["maximum_mic_distance_angstrom"])
    cutoff = float(data["cutoff_angstrom"])
    assert str(data["cutoff_mode"]) == "full_supercell"
    assert np.isclose(cutoff - maximum_mic_distance, 1e-6, atol=1e-12, rtol=0)
    assert int(data["phonopy_configurations"]) == 4
    assert data["mlfcs_forces"].shape == (12, 32, 3)
    actual = _calculate(data, unitcell, reference_supercell, acoustic_sum_rule=False)
    expected = data["phonopy_fc2"]
    difference = actual - expected
    assert np.max(np.abs(difference)) < 3.5e-3
    assert np.sqrt(np.mean(difference**2)) < 4e-4
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 2e-4
    assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > 0.9999999


@pytest.mark.reference
def test_AlN_FC2_matches_phonopy_with_ASR():
    data, unitcell, reference_supercell = _load()
    actual = _calculate(data, unitcell, reference_supercell, acoustic_sum_rule=True)
    expected = data["phonopy_fc2_asr"]
    assert _asr_residual(actual) < 1e-8
    assert _asr_residual(expected) < 1e-8
    difference = actual - expected
    assert np.max(np.abs(difference)) < 3.5e-3
    assert np.sqrt(np.mean(difference**2)) < 4e-4
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 2e-4
    assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > 0.9999999
