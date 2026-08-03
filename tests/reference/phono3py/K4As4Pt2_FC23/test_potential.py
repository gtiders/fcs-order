from __future__ import annotations

import numpy as np
import pytest

from tests.reference.phono3py.K4As4Pt2_FC23.case import (
    DATA,
    calculation_and_reference,
)


@pytest.mark.reference
def test_packaged_potential_reproduces_first_MLFCS_force_configuration():
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    data, calculation, _ = calculation_and_reference(2)
    atoms = calculation.sow()[0]
    atoms.calc = PolymlpASECalculator(pot=DATA / "polymlp.yaml")
    assert np.allclose(atoms.get_forces(), data["fc2_forces"][0], atol=1e-10, rtol=0)
