"""Fit graphene FC2, then apply strict Born-Huang and Huang postprocessing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.lattice.hexagonal import Graphene

from mlfcs import StructureRelation
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"


def load_data():
    primitive = Graphene(symbol="C", latticeconstant={"a": 2.466340583, "c": 40.0})
    source = read(INPUT / "phonopy_snapshot.extxyz")
    reference = source.copy()
    reference.calc = None
    snapshot = reference.copy()
    snapshot.positions += source.arrays["displacements"]
    snapshot.calc = SinglePointCalculator(snapshot, forces=source.get_forces())
    relation = StructureRelation.from_atoms(primitive, reference)
    return primitive, reference, [snapshot], relation


def fit_case(primitive, reference, snapshots, *, constrain, output):
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 8.0},
        max_body_orders={2: 2},
        verbose=True,
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.0,
        acoustic_sum_rule=True,
    )
    constrained = (
        result.force_constants.enforce_harmonic_constraints(born_huang=True, huang=True)
        if constrain
        else None
    )
    force_constants = (
        constrained.force_constants if constrained is not None else result.force_constants
    )
    output.mkdir(parents=True, exist_ok=True)
    force_constants.write(output / "mlfcs.h5", format="hdf5")
    force_constants.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "single_snapshot": True,
                "harmonic_constraints": (
                    asdict(constrained.diagnostics) if constrained is not None else None
                ),
                **asdict(result.diagnostics),
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def main():
    primitive, reference, snapshots, relation = load_data()
    write(INPUT / "primitive.vasp", primitive, format="vasp", direct=True, sort=False)
    write(INPUT / "reference.vasp", reference, format="vasp", direct=True, sort=False)
    print("primitive atoms:", len(primitive))
    print("reference atoms:", len(reference))
    print("supercell matrix:\n", relation.supercell_matrix)
    fit_case(primitive, reference, snapshots, constrain=False, output=ROOT / "mlfcs" / "asr")
    fit_case(
        primitive,
        reference,
        snapshots,
        constrain=True,
        output=ROOT / "mlfcs" / "born-huang-huang",
    )


if __name__ == "__main__":
    main()
