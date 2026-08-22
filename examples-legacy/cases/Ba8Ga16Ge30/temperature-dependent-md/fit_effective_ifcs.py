"""Fit the temperature-dependent FC2+FC3 model used for BaGaGe conductivity."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"


def fit(temperature_directory: Path) -> None:
    snapshots_path = temperature_directory / "nve.extxyz"
    if not snapshots_path.is_file():
        raise FileNotFoundError(f"missing NVE fitting data: {snapshots_path}")

    primitive = read(INPUT / "reference.vasp")
    reference = primitive.repeat((2, 2, 2))
    snapshots = read(snapshots_path, index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3),
        cutoffs={2: 5.4, 3: 4.35},
        max_body_orders={2: 2, 3: 2},
        symprec=1e-4,
        verbose=True,
    )
    output = temperature_directory / "mlfcs"
    result = fitter.fit(
        snapshots,
        validation_split=0.0,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
        cache_directory=output / "cache",
    )
    output.mkdir(parents=True, exist_ok=True)
    result.force_constants.write(output / "mlfcs.h5", format="hdf5")
    result.force_constants.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    result.force_constants.write(output / "fc2.h5", format="phonopy_hdf5", order=2)
    result.force_constants.write(output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    result.force_constants.write(output / "fc3.h5", format="phono3py_hdf5", order=3)
    (output / "metrics.json").write_text(
        json.dumps(asdict(result.diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} MD_TEMPERATURE_DIRECTORY")
    fit(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
