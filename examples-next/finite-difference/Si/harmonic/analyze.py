"""Compare regenerated FC2 outputs with the legacy Si case."""

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
        if a[key].dtype.kind in "OUS" or b[key].dtype.kind in "OUS":
            equal = np.array_equal(a[key], b[key])
        else:
            equal = np.array_equal(a[key], b[key])
        if not equal:
            raise SystemExit(f"dataset differs: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "examples/cases/Si/finite-difference/harmonic",
    )
    args = parser.parse_args()
    for name in ("mlfcs.h5", "fc2.h5"):
        compare_hdf5(CASE / "results" / name, args.legacy / name)
    if (CASE / "results/FORCE_CONSTANTS_2ND").read_bytes() != (
        args.legacy / "FORCE_CONSTANTS_2ND"
    ).read_bytes():
        raise SystemExit("phonopy text FC2 differs from the legacy result")
    print("Si FC2 regenerated outputs match the legacy datasets and text export")


if __name__ == "__main__":
    main()
