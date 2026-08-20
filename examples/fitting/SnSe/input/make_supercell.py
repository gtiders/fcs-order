#!/usr/bin/env python3
"""Create the phonopy-ordered SnSe force-constant reference supercell."""

from pathlib import Path

import numpy as np
from ase.io import read, write

from mlfcs import build_supercell

CASE = Path(__file__).resolve().parent


def main() -> None:
    primitive = read(CASE / "primitive.vasp")
    reference = build_supercell(primitive, np.diag((2, 4, 4)))
    write(CASE / "reference.vasp", reference, format="vasp", direct=True, sort=False, vasp5=True)
    print(f"wrote {len(reference)} atoms to {CASE / 'reference.vasp'}")


if __name__ == "__main__":
    main()
