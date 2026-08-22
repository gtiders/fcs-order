"""Fit Si FC3-FC4 while keeping an independently supplied FC2 fixed."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs import read_hdf5
from mlfcs.fitting import ForceConstantFitter

BOHR = 0.529177210903
CASE = Path(__file__).resolve().parent
FITTING = CASE.parent
SI_CASE = FITTING.parent

BASELINES = {
    "harmonic-fit": FITTING / "harmonic" / "mlfcs.h5",
    "finite-difference": SI_CASE / "finite-difference" / "harmonic" / "mlfcs.h5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES, default="harmonic-fit")
    parser.add_argument("--max-iterations", type=int, default=10_000)
    args = parser.parse_args()

    source = FITTING / "anharmonic"
    output = CASE / args.baseline
    fitter = ForceConstantFitter(
        read(source / "primitive.vasp"),
        read(source / "supercell.vasp"),
        orders=(2, 3, 4),
        cutoffs={2: None, 3: None, 4: 11.0 * BOHR},
        max_body_orders={2: 2, 3: 3, 4: 3},
    )
    try:
        result = fitter.fit(
            read(source / "train.extxyz", index=":"),
            frozen_force_constants={2: read_hdf5(BASELINES[args.baseline])},
            validation_split=0.0,
            batch_size=4,
            damping=0.0,
            regularization=None,
            acoustic_sum_rule=True,
            cache_directory=output / "cache",
            max_iterations=args.max_iterations,
        )
    except ValueError as error:
        if args.baseline == "finite-difference":
            raise SystemExit(
                "The archived finite-difference FC2 is intentionally rejected: it uses a "
                "strained 128-atom reference, while this training set uses the 64-atom "
                "ALAMODE reference. Frozen IFCs require the same physical translation lattice. "
                f"Original validation error: {error}"
            ) from error
        raise

    output.mkdir(parents=True, exist_ok=True)
    result.force_constants.write(output / "mlfcs.h5", format="hdf5")
    result.force_constants.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    result.force_constants.write(output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    result.force_constants.write(output / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
    (output / "fit-summary.json").write_text(
        json.dumps(asdict(result.diagnostics), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
