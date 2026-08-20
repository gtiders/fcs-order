from pathlib import Path

import numpy as np
from ase.io import read, write

from mlfcs import build_supercell


def main() -> None:
    directory = Path(__file__).parent
    primitive = read(directory / "primitive.vasp")
    supercell = build_supercell(primitive, np.diag([4, 4, 4]))
    write(directory / "supercell.vasp", supercell, format="vasp", direct=True, sort=False, vasp5=True)


if __name__ == "__main__":
    main()
