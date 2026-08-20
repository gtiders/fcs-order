from pathlib import Path

import numpy as np
from ase.io import read, write

from mlfcs import build_supercell


def main() -> None:
    directory = Path(__file__).parent
    matrix = np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
    supercell = build_supercell(read(directory / "primitive.vasp"), matrix)
    write(directory / "supercell.vasp", supercell, format="vasp", direct=True, sort=False, vasp5=True)


if __name__ == "__main__":
    main()
