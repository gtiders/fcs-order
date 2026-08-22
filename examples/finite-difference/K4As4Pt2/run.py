"""Compare MLFCS and phono3py finite differences using one ASE Polymlp potential."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from phonopy.interface.calculator import read_crystal_structure
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import ForceConstantCalculation

CASE = Path(__file__).resolve().parent
INPUT = CASE / "input"
RESULTS = CASE / "results"
# Use a common 6 Å interaction cutoff for the finite-difference reference.
CUTOFFS = {2: 6.0, 3: 6.0}
DISPLACEMENT = 0.01


def _calculator() -> PolymlpASECalculator:
    return PolymlpASECalculator(pot=INPUT / "polymlp.yaml")


def _ase_atoms(cell) -> Atoms:
    return Atoms(
        numbers=cell.numbers,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


def _evaluate_mlfcs_structures(
    structures: list[Atoms], archive: Path, *, overwrite: bool = False
) -> np.ndarray:
    """Evaluate in bounded calculator batches so Polymlp state is reclaimed."""
    if archive.is_file() and not overwrite:
        values = np.load(archive)["forces"]
        if len(values) == len(structures):
            return values
    values: list[np.ndarray] = []
    calculator = _calculator()
    for index, atoms in enumerate(structures, start=1):
        atoms.calc = calculator
        values.append(np.asarray(atoms.get_forces(), dtype=float))
        if index % 100 == 0 or index == len(structures):
            current = np.asarray(values)
            np.savez_compressed(archive, forces=current, configuration_ids=np.arange(index))
            print(f"MLFCS ASE force evaluations: {index}/{len(structures)}", flush=True)
            gc.collect()
            if index < len(structures):
                calculator = _calculator()
    return np.asarray(values)


def run_mlfcs(*, overwrite: bool = False) -> None:
    primitive = read(INPUT / "primitive.vasp")
    reference = read(INPUT / "supercell.vasp")
    for order, name in ((2, "harmonic"), (3, "three-phonon")):
        output = RESULTS / name
        output.mkdir(exist_ok=True)
        calculation = ForceConstantCalculation(
            primitive,
            order=order,
            reference=reference,
            cutoff=CUTOFFS[order],
            displacement=DISPLACEMENT,
            verbose=False,
        )
        structures = calculation.sow()
        archive = output / "forces.npz"
        forces = _evaluate_mlfcs_structures(structures, archive, overwrite=overwrite)
        np.savez_compressed(
            output / "forces.npz",
            forces=np.asarray(forces),
            configuration_ids=np.arange(len(forces), dtype=int),
        )
        manifest = {
            "method": "mlfcs-ase-polymlp",
            "order": order,
            "reference": str((INPUT / "supercell.vasp").name),
            "cutoff_angstrom": CUTOFFS[order],
            "displacement_angstrom": DISPLACEMENT,
            "configuration_count": len(structures),
            "atom_order": "MLFCS sow order",
        }
        (output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result = calculation.reap(forces, acoustic_sum_rule=(order == 2))
        result.write(output / "mlfcs.h5", format="hdf5")
        if order == 2:
            result.write(output / "fc2.h5", format="phonopy_hdf5", order=2)
            result.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
        else:
            result.write(output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
        print(f"MLFCS order-{order}: {len(forces)} ASE force evaluations")


def run_phono3py() -> None:
    output = RESULTS / "phono3py-reference"
    output.mkdir(exist_ok=True)
    cell, _ = read_crystal_structure(filename=str(INPUT / "supercell.vasp"), interface_mode="vasp")
    phono3py = Phono3py(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    phono3py.generate_displacements(
        distance=DISPLACEMENT,
        cutoff_pair_distance=CUTOFFS[3],
    )
    displacements = phono3py.supercells_with_displacements
    archive = output / "forces.npz"
    forces = []
    if archive.is_file():
        cached = np.load(archive)["forces"]
        forces.extend(cached)
    calculator = _calculator()
    for index, displaced in enumerate(displacements[len(forces) :], start=len(forces) + 1):
        if displaced is None:
            # phono3py keeps excluded symmetry-equivalent entries as None so
            # the force list retains the dataset's positional indexing.
            forces.append(np.zeros((len(phono3py.supercell), 3), dtype=float))
        else:
            atoms = _ase_atoms(displaced)
            atoms.calc = calculator
            forces.append(atoms.get_forces())
        if index % 250 == 0 or index == len(displacements):
            np.savez_compressed(archive, forces=np.asarray(forces))
            print(f"phono3py force evaluations: {index}/{len(displacements)}")
            gc.collect()
            if index < len(displacements):
                calculator = _calculator()
    phono3py.forces = np.asarray(forces)
    np.savez_compressed(archive, forces=np.asarray(forces))
    phono3py.produce_fc3(is_compact_fc=True, fc_calculator="traditional")
    write_fc2_to_hdf5(phono3py.fc2, filename=output / "fc2.h5", p2s_map=phono3py.primitive.p2s_map)
    write_fc3_to_hdf5(phono3py.fc3, filename=output / "fc3.h5", p2s_map=phono3py.primitive.p2s_map)
    print(f"phono3py traditional: {len(forces)} ASE force evaluations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("mlfcs", "phono3py", "both"), default="both")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.route in ("mlfcs", "both"):
        run_mlfcs(overwrite=args.overwrite)
    if args.route in ("phono3py", "both"):
        run_phono3py()


if __name__ == "__main__":
    main()
