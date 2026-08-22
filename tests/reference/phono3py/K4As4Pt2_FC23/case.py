from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs import ForceConstantCalculation

DATA = Path(__file__).parent / "data"
REFERENCE = DATA / "reference.npz"


def calculation_and_reference(order: int):
    data = np.load(REFERENCE)
    unitcell = Atoms(
        numbers=data["unitcell_numbers"],
        cell=data["unitcell_cell"],
        scaled_positions=data["unitcell_scaled_positions"],
        pbc=True,
    )
    calculation = ForceConstantCalculation(
        unitcell,
        order=order,
        supercell=tuple(int(value) for value in data["supercell_repeats"]),
        cutoff=float(data["cutoff_angstrom"]),
        displacement=float(data["displacement_angstrom"]),
        jax_platform="cpu",
        report_cutoff=False,
        verbose=False,
    )
    result = calculation.reap(
        data[f"fc{order}_forces"],
        acoustic_sum_rule=False,
    )
    return data, calculation, result.sparse[order]


def assert_matches_phono3py(actual: np.ndarray, expected: np.ndarray, *, order: int) -> None:
    difference = actual - expected
    limits = {
        2: {"max": 2.1e-3, "rms": 8e-5, "relative": 1.4e-4},
        3: {"max": 1.1e-2, "rms": 4.3e-5, "relative": 3.6e-4},
    }[order]
    assert np.max(np.abs(difference)) < limits["max"]
    assert np.sqrt(np.mean(difference**2)) < limits["rms"]
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < limits["relative"]
    assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > 0.9999999
