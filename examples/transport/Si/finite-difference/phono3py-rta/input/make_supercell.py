from pathlib import Path

import numpy as np
from ase.io import read, write

from mlfcs.tools import build_supercell


def main() -> None:
    directory = Path(__file__).parent
    supercell = build_supercell(read(directory / "primitive.vasp"), np.diag([4, 4, 4]), ordering="phonopy")
    write(directory / "supercell.vasp", supercell, format="vasp", direct=True, sort=False, vasp5=True)


if __name__ == "__main__":
    main()
