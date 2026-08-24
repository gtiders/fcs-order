"""Fit K4As4Pt2 FC2-FC4 with a three-body FC4 truncation."""

from __future__ import annotations

from mlfcs import write_force_constants
import argparse
from pathlib import Path

from ase.io import read
from ase.units import Bohr

from mlfcs.fitting import ForceConstantFitter

CASE = Path(__file__).resolve().parent
INPUT = CASE / "input"
RESULTS = CASE / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-order-4", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--regularization",
        choices=("none", "scaled_group_lasso"),
        default="none",
    )
    args = parser.parse_args()
    output = RESULTS / ("three-body" if args.body_order_4 == 3 else "four-body")
    cache = RESULTS / "cache" / ("three-body" if args.body_order_4 == 3 else "four-body")
    fitter = ForceConstantFitter(
        read(INPUT / "primitive.vasp"),
        read(INPUT / "reference.vasp"),
        orders=(2, 3, 4),
        fitting_basis="wick",
        cutoffs={2: 6.5, 3: 12 * Bohr, 4: 8 * Bohr},
        max_body_orders={2: 2, 3: 3, 4: args.body_order_4},
    )
    result = fitter.fit(
        read(INPUT / "train.extxyz", index=":"),
        validation_split=0.1,
        batch_size=4,
        acoustic_sum_rule=True,
        tolerance=1e-5,
        regularization=None if args.regularization == "none" else args.regularization,
        max_iterations=10_000,
        cache_directory=cache,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_force_constants(result.force_constants, output / "mlfcs.h5", format="hdf5")
    write_force_constants(result.force_constants, output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    write_force_constants(result.force_constants, output / "fc2.h5", format="phonopy_hdf5", order=2)
    write_force_constants(result.force_constants, output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    write_force_constants(result.force_constants, output / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)


if __name__ == "__main__":
    main()
