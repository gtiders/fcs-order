"""Compare regenerated fitted FC2 with the legacy Si fitting case."""

from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[3]


def main() -> None:
    current = CASE / "results/mlfcs.h5"
    legacy = ROOT / "examples-legacy/cases/Si/fitting/harmonic/mlfcs.h5"
    with h5py.File(current) as left, h5py.File(legacy) as right:
        keys: list[str] = []
        left.visit(keys.append)
        for key in keys:
            if isinstance(left[key], h5py.Dataset) and not np.array_equal(
                left[key][()], right[key][()]
            ):
                raise SystemExit(f"fitted FC2 differs in {key}")
    print("Si fitted FC2 matches the legacy force-constant object")


if __name__ == "__main__":
    main()
