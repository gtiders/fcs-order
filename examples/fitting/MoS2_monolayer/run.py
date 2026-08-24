"""Fit FC2, then apply strict Born-Huang and Huang postprocessing."""

from __future__ import annotations

from mlfcs import enforce_rotational_sum_rules, write_force_constants
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ase import Atoms
from ase.io import read

from mlfcs import StructureRelation
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
RESULTS = ROOT / "results"


def prepare_structures() -> tuple[Atoms, Atoms, list[Atoms]]:
    """Read the explicit MLFCS primitive, reference, and force snapshots."""
    primitive = read(INPUT / "primitive.vasp")
    reference = read(INPUT / "reference.vasp")
    snapshots = read(INPUT / "training.extxyz", index=":")
    StructureRelation.from_atoms(primitive, reference)
    return primitive, reference, snapshots


def fit_case(
    primitive: Atoms,
    reference: Atoms,
    snapshots: list[Atoms],
    *,
    constrain: bool,
    output: Path,
) -> None:
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 8.0},
        max_body_orders={2: 2},
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.2,
        seed=0,
        acoustic_sum_rule=True,
    )
    constrained = (
        enforce_rotational_sum_rules(result.force_constants, born_huang=True, huang=True)
        if constrain
        else None
    )
    force_constants = (
        constrained.force_constants if constrained is not None else result.force_constants
    )
    output.mkdir(parents=True, exist_ok=True)
    write_force_constants(force_constants, output / "mlfcs.h5", format="hdf5")
    write_force_constants(force_constants, output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    metrics = asdict(result)
    metrics["harmonic_constraints"] = (
        asdict(constrained) if constrained is not None else None
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("asr", "born-huang-huang", "both"), default="both")
    args = parser.parse_args()

    primitive, reference, snapshots = prepare_structures()
    relation = StructureRelation.from_atoms(primitive, reference)
    print(f"Primitive atoms: {len(primitive)}")
    print(f"Reference atoms: {len(reference)}")
    print(f"Supercell matrix:\n{relation.supercell_matrix}")
    print(f"Maximum mapping residual: {relation.position_residual:.3e} Angstrom")

    if args.case in {"asr", "both"}:
        fit_case(
            primitive,
            reference,
            snapshots,
            constrain=False,
            output=RESULTS / "asr",
        )
    if args.case in {"born-huang-huang", "both"}:
        fit_case(
            primitive,
            reference,
            snapshots,
            constrain=True,
            output=RESULTS / "born-huang-huang",
        )


if __name__ == "__main__":
    main()
