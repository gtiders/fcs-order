from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlfcs import ForceConstantCalculation
from tests.reference.shengbte.Si_FC3.adapter import parse_fc3

DATA = Path(__file__).parent / "data" / "reference.npz"


@pytest.mark.reference
@pytest.mark.parametrize(
    ("acoustic_sum_rule", "maximum", "rms", "relative", "correlation"),
    [
        (False, 0.09, 0.0041, 0.0070, 0.99997),
        (True, 0.09, 0.0037, 0.0064, 0.99997),
    ],
)
def test_Si_FC3_shengbte_compatibility_mode_matches_thirdorder(
    tmp_path, acoustic_sum_rule, maximum, rms, relative, correlation
):
    with np.load(DATA) as data:
        unitcell = Atoms(
            numbers=data["unitcell_numbers"],
            cell=data["unitcell_cell"],
            scaled_positions=data["unitcell_scaled_positions"],
            pbc=True,
        )
        forces = data["mlfcs_forces_grouped"]
        plan_hash = str(data["mlfcs_plan_hash"])
        expected_translations = data["reference_translations_mod_supercell"]
        expected_atoms = data["reference_primitive_atoms"]
        expected = data["reference_fc3"]

    calculation = ForceConstantCalculation(
        unitcell,
        order=3,
        supercell=(3, 3, 3),
        cutoff=-6,
        displacement=0.01,
        jax_platform="cpu",
        verbose=False,
    )
    result = calculation.reap(
        forces,
        atom_order="grouped",
        plan_hash=plan_hash,
        acoustic_sum_rule=acoustic_sum_rule,
    )
    output = tmp_path / "FORCE_CONSTANTS_3RD"
    result.write(output, format="shengbte", order=3, compatibility="thirdorder")
    translations, atoms, actual = parse_fc3(output, np.asarray(unitcell.cell))

    np.testing.assert_array_equal(translations, expected_translations)
    np.testing.assert_array_equal(atoms, expected_atoms)
    difference = actual - expected
    assert np.max(np.abs(difference)) < maximum
    assert np.sqrt(np.mean(difference**2)) < rms
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < relative
    assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > correlation
