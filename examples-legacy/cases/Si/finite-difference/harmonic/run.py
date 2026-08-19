"""Reconstruct the Si FC2 example from its archived VASP calculations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs import ForceConstantCalculation

CASE = Path(__file__).resolve().parent


def _same_structure(left, right, tolerance: float = 1e-8) -> bool:
    return (
        left.get_chemical_symbols() == right.get_chemical_symbols()
        and np.allclose(left.cell.array, right.cell.array, atol=tolerance, rtol=0.0)
        and np.allclose(left.positions, right.positions, atol=tolerance, rtol=0.0)
    )


def main() -> None:
    manifest = json.loads((CASE / "mlfcs-plan.json").read_text())
    calculation = ForceConstantCalculation(
        read(CASE / "primitive.vasp"),
        reference=read(CASE / "supercell.vasp"),
        order=manifest["order"],
        supercell_matrix=manifest["supercell_matrix"],
        cutoff=manifest["cutoff"],
        displacement=manifest["displacement"],
    )

    generated = calculation.sow()
    filenames = manifest["filenames"]
    if len(generated) != len(filenames):
        raise ValueError(f"plan contains {len(filenames)} names, MLFCS generated {len(generated)}")
    for filename, atoms in zip(filenames, generated, strict=True):
        archived = read(CASE / "structures" / filename)
        if not _same_structure(atoms, archived):
            raise ValueError(f"archived displacement no longer matches the MLFCS plan: {filename}")

    output_name = manifest["calculator_output"]
    forces = np.asarray(
        [
            read(CASE / "calculations" / name / output_name, index=-1).get_forces()
            for name in filenames
        ]
    )
    configuration_ids = np.arange(len(filenames), dtype=int)
    np.savez_compressed(
        CASE / "forces.npz",
        forces=forces,
        configuration_ids=configuration_ids,
        atom_order=np.asarray("reference"),
    )

    result = calculation.reap(forces, acoustic_sum_rule=True)
    result.write(CASE / "mlfcs.h5", format="hdf5")
    result.write(CASE / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    result.write(CASE / "fc2.h5", format="phonopy_hdf5", order=2)
    print(f"Reconstructed FC2 from {len(forces)} ordered force calculations")


if __name__ == "__main__":
    main()
