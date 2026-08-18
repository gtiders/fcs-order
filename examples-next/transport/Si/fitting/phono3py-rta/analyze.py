"""Compare the regenerated fitted RTA conductivity with the legacy case."""

from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[4]


def main() -> None:
    current = np.loadtxt(CASE / "results/kappa-rta.txt")
    legacy = np.loadtxt(ROOT / "examples/cases/Si/fitting/thermal-conductivity/kappa-rta.txt")
    if not np.array_equal(current, legacy):
        raise SystemExit("fitted RTA output differs from the legacy result")
    print("Si fitted RTA output matches the legacy result")


if __name__ == "__main__":
    main()
