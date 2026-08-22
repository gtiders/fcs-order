"""Fit and export Si FC2 from the collected forces."""

from __future__ import annotations

from mlfcs import write_force_constants
import json
from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs import ForceConstantCalculation

CASE = Path(__file__).resolve().parent


def main() -> None:
    manifest = json.loads((CASE / "input/mlfcs-plan.json").read_text())
    calculation = ForceConstantCalculation(
        read(CASE / "input/primitive.vasp"),
        reference=read(CASE / "input/supercell.vasp"),
        order=manifest["order"],
        cutoff=manifest["cutoff"],
        displacement=manifest["displacement"],
    )
    forces = np.load(CASE / "results/forces.npz")["forces"]
    result = calculation.reap(forces, acoustic_sum_rule=True)
    output = CASE / "results"
    write_force_constants(result, output / "mlfcs.h5", format="hdf5")
    write_force_constants(result, output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    write_force_constants(result, output / "fc2.h5", format="phonopy_hdf5", order=2)
    print(f"Fitted FC2 from {len(forces)} ordered force calculations")


if __name__ == "__main__":
    main()
