"""Convert the ALAMODE DFSET into the strict MLFCS extxyz input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import Bohr, Rydberg

CASE = Path(__file__).resolve().parents[1]
SOURCE = CASE.parent / "fiting"
NATOMS = 120
NSNAPSHOTS = 250


def read_dfset(path: Path, reference):
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        try:
            rows.append([float(value) for value in fields])
        except ValueError:
            continue
    values = np.asarray(rows, dtype=float)
    expected = NSNAPSHOTS * len(reference)
    if values.shape != (expected, 6):
        raise ValueError(f"expected {expected} displacement-force rows, got {len(values)}")
    snapshots = []
    for frame in values.reshape(NSNAPSHOTS, len(reference), 6):
        atoms = reference.copy()
        atoms.positions += frame[:, :3] * Bohr
        atoms.calc = SinglePointCalculator(atoms, forces=frame[:, 3:] * Rydberg / Bohr)
        snapshots.append(atoms)
    return snapshots


def main() -> None:
    source_primitive = read(SOURCE / "POSCAR")
    source_reference = read(SOURCE / "SPOSCAR")
    snapshots = read_dfset(SOURCE / "DFTSET_RAND", source_reference)
    write(CASE / "primitive.vasp", source_primitive, format="vasp", direct=True, vasp5=True)
    write(CASE / "reference.vasp", source_reference, format="vasp", direct=True, vasp5=True)
    write(CASE / "train.extxyz", snapshots)
    print(f"wrote {len(snapshots)} snapshots with {len(source_reference)} atoms each")


if __name__ == "__main__":
    main()
