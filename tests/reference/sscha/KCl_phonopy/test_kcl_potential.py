from __future__ import annotations

import numpy as np
import pytest

from mlfcs.sscha import SSCHA
from tests.reference.sscha.KCl_phonopy.case import DATA, conventional_cell

pytestmark = pytest.mark.reference


def test_phonopy_kcl_potential_reproduces_sscha_scale():
    """Run the upstream KCl MLP through the native MLFCS SSCHA path."""
    calculator_module = pytest.importorskip("pypolymlp.calculator.utils.ase_calculator")
    calculator = calculator_module.PolymlpASECalculator(pot=DATA / "polymlp.yaml")
    sscha = SSCHA(
        conventional_cell(),
        supercell=(2, 2, 2),
        temperature=300,
        snapshots=10,
        max_iterations=1,
        random_seed=42,
        imaginary_modes="absolute",
    )

    sscha.run(calculator)

    initialization, canonical = sscha.history
    assert initialization.free_energy is None
    np.testing.assert_allclose(initialization.force_constants[0, 0], np.eye(3) * 1.9042, atol=5e-3)
    np.testing.assert_allclose(canonical.force_constants[0, 0], np.eye(3) * 2.1625, atol=3e-2)
    # Phonopy's own 50-snapshot, three-iteration test accepts 2.1 +/- 0.1 eV/A^2.
    assert canonical.force_constants[0, 0, 0, 0] == pytest.approx(2.1, abs=0.1)
    assert canonical.free_energy is not None
    # MLFCS receives the four-primitive-cell conventional cell. Normalize its
    # result before comparing with phonopy's -0.0986 +/- 0.001 eV reference.
    assert canonical.free_energy / 4 == pytest.approx(-0.0986, abs=6e-3)
