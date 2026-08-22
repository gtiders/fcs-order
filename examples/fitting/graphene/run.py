"""Fit graphene FC2, then apply strict Born-Huang and Huang postprocessing."""

from __future__ import annotations

from mlfcs import enforce_rotational_sum_rules, write_force_constants
import json
from dataclasses import asdict
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from mlfcs import StructureRelation
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
RESULTS = ROOT / "results"


def load_data():
    primitive = read(INPUT / "primitive.vasp")
    reference = read(INPUT / "reference.vasp")
    source = read(INPUT / "phonopy_snapshot.extxyz")
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
    print("primitive atoms:", len(primitive))
    print("reference atoms:", len(reference))
    print("supercell matrix:\n", relation.supercell_matrix)
    fit_case(primitive, reference, snapshots, constrain=False, output=RESULTS / "asr")
    fit_case(
        primitive,
        reference,
        snapshots,
        constrain=True,
        output=RESULTS / "born-huang-huang",
    )


if __name__ == "__main__":
    main()
