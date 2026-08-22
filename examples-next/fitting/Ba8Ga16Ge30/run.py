#!/usr/bin/env python3
"""Fit the hiPhive Ba8Ga16Ge30 Model-4 interaction space with MLFCS."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "results"


def main() -> None:
    primitive = read(INPUT / "primitive.vasp")
    reference = read(INPUT / "reference.vasp")
    snapshots = read(INPUT / "training.extxyz", index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: 5.4, 3: 4.35, 4: 4.35},
        max_body_orders={2: 2, 3: 2, 4: 2},
        symprec=1e-4,
        verbose=True,
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.0,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
        cache_directory=OUTPUT / "cache",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.force_constants.write(OUTPUT / "mlfcs.h5", format="hdf5")
    result.force_constants.write(OUTPUT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    result.force_constants.write(OUTPUT / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    result.force_constants.write(OUTPUT / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(asdict(result.diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
