"""Compare regenerated FC3 outputs with the legacy Si case."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[3]


def datasets(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    with h5py.File(path) as handle:
        handle.visititems(
            lambda name, item: (
                values.__setitem__(name, item[()]) if isinstance(item, h5py.Dataset) else None
            )
        )
    return values


def compare_hdf5(left: Path, right: Path) -> None:
    a, b = datasets(left), datasets(right)
    if set(a) != set(b):
        raise SystemExit(f"dataset keys differ: {set(a) ^ set(b)}")
    for key in a:
        if not np.array_equal(a[key], b[key]):
            raise SystemExit(f"dataset differs: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "examples-legacy/cases/Si/finite-difference/three-phonon",
    )
    args = parser.parse_args()
    for name in ("mlfcs.h5", "fc3.h5"):
        compare_hdf5(CASE / "results" / name, args.legacy / name)
    if (CASE / "results/FORCE_CONSTANTS_3RD").read_bytes() != (
        args.legacy / "FORCE_CONSTANTS_3RD"
    ).read_bytes():
        raise SystemExit("ShengBTE text FC3 differs from the legacy result")
    print("Si FC3 regenerated outputs match the legacy datasets and text export")


if __name__ == "__main__":
    main()
