#!/usr/bin/env python3
"""Fit SnSe FC2, FC3, and FC4 from the 300 K FCP trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs.fitting import ForceConstantFitter

CASE = Path(__file__).resolve().parent
INPUT = CASE / "input"
SNAPSHOTS = CASE / "md" / "T300K" / "nve.extxyz"
OUTPUT = CASE / "results" / "fc234"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    if not args.snapshots.is_file():
        raise FileNotFoundError(
            f"missing {args.snapshots}; run md/run.py after preparing input/reference.vasp"
        )

    primitive = read(INPUT / "primitive.vasp")
    reference = read(INPUT / "reference.vasp")
    snapshots = read(args.snapshots, index=":")
    cache = args.output / "cache"
    if args.overwrite_cache and cache.exists():
        shutil.rmtree(cache)
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: None, 3: 6.5, 4: 4.5},
        max_body_orders={2: 2, 3: 3, 4: 3},
        symprec=1e-4,
        verbose=True,
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.1,
        batch_size=4,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
        cache_directory=cache,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result.force_constants.write(args.output / "mlfcs.h5", format="hdf5")
    result.force_constants.write(args.output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    result.force_constants.write(args.output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    result.force_constants.write(args.output / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
    (args.output / "metrics.json").write_text(
        json.dumps(asdict(result.diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(f"wrote FC2/FC3/FC4 fit to {args.output}")


if __name__ == "__main__":
    main()
