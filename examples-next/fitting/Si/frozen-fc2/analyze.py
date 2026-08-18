"""Compare the regenerated frozen-FC2 fit with the legacy Si case."""

from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[3]


def main() -> None:
    current = CASE / "results/harmonic-fit/mlfcs.h5"
    legacy = ROOT / "examples/cases/Si/fitting/frozen-fc2/harmonic-fit/mlfcs.h5"
    with h5py.File(current) as left, h5py.File(legacy) as right:
        names: list[str] = []
        left.visit(names.append)
        for name in names:
            if isinstance(left[name], h5py.Dataset) and not np.array_equal(
                left[name][()], right[name][()]
            ):
                raise SystemExit(f"frozen fit differs in {name}")
    print("Si frozen harmonic-fit output matches the legacy result")


if __name__ == "__main__":
    main()
