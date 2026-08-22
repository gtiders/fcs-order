"""Convert the bundled ALAMODE Si DFSET into an ASE extxyz dataset."""

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import Bohr, Rydberg

CASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "alamode/example/Si/anharm_IFCs"


def read_dfset(path: Path, reference, count: int):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 6:
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                pass
    values = np.asarray(rows, dtype=float).reshape(count, len(reference), 6)
    snapshots = []
    for frame in values:
        atoms = reference.copy()
        atoms.positions += frame[:, :3] * Bohr
        atoms.calc = SinglePointCalculator(atoms, forces=frame[:, 3:] * Rydberg / Bohr)
        snapshots.append(atoms)
    return snapshots


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"未找到 ALAMODE 原始目录 {SOURCE}；已提交的 extxyz 可直接用于拟合，"
            "重新转换时请先准备原始 ALAMODE 示例目录。"
        )
    (CASE / "harmonic/input").mkdir(parents=True, exist_ok=True)
    (CASE / "anharmonic/input").mkdir(parents=True, exist_ok=True)
    primitive = read(SOURCE / "2_generate_config/POSCAR_primitive_cell")
    reference = read(SOURCE / "2_generate_config/POSCAR_supercell")
    harmonic_reference = read(SOURCE / "1_harmonic/VASP_input/POSCAR")
    harmonic_snapshots = read_dfset(SOURCE / "1_harmonic/DFSET_harmonic", harmonic_reference, 1)
    snapshots = read_dfset(SOURCE / "3_cv/DFSET_randomQ", reference, 100)
    write(
        CASE / "harmonic/input/primitive.vasp",
        primitive,
        format="vasp",
        direct=True,
        vasp5=True,
    )
    write(
        CASE / "harmonic/input/supercell.vasp",
        harmonic_reference,
        format="vasp",
        direct=True,
        vasp5=True,
    )
    write(CASE / "harmonic/input/train.extxyz", harmonic_snapshots)
    write(
        CASE / "anharmonic/input/supercell.vasp",
        reference,
        format="vasp",
        direct=True,
        vasp5=True,
    )
    write(CASE / "anharmonic/input/train.extxyz", snapshots)
    print(f"wrote {len(snapshots)} Si snapshots with {len(reference)} atoms each")


if __name__ == "__main__":
    main()
