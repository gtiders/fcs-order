from __future__ import annotations

from pathlib import Path

import pytest
from ase.io import read

from mlfcs import ForceConstantCalculation

STRUCTURES = Path(__file__).parent / "structures"


@pytest.mark.reference
def test_Si_FC3_sow_uses_the_reference_order_contract():
    calculation = ForceConstantCalculation(
        read(STRUCTURES / "POSCAR-unitcell"),
        order=3,
        supercell=(3, 3, 3),
        cutoff=-6,
        displacement=0.01,
        report_cutoff=False,
    )

    structures = calculation.sow()
    assert len(structures) == len(calculation.plan) == 168
    assert [atoms.info["mlfcs_configuration_id"] for atoms in structures] == list(range(168))
    assert {atoms.info["mlfcs_atom_order"] for atoms in structures} == {"reference"}
