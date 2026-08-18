"""Run phono3py's iterative-free (BTE-RTA) conductivity calculation.

The directory must contain ``fc2.h5``, ``fc3.h5`` and a reference
supercell. Phono3py discovers the primitive cell from that supercell; no
primitive POSCAR or atom reordering is supplied by this wrapper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parent


def main() -> None:
    (CASE / "results").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supercell",
        type=Path,
        default=CASE / "input/supercell.vasp",
        help="reference supercell; primitive is found by phono3py",
    )
    parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11))
    parser.add_argument("--temperatures", nargs="+", type=float, default=(300.0,))
    args = parser.parse_args()

    supercell = args.supercell.resolve()
    if not supercell.is_file():
        raise FileNotFoundError(supercell)
    for filename in ("fc2.h5", "fc3.h5"):
        if not (CASE / "input" / filename).is_file():
            raise FileNotFoundError(CASE / "input" / filename)

    cell, _ = read_crystal_structure(filename=str(supercell), interface_mode="vasp")
    phono3py = Phono3py(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    phono3py.fc2 = read_fc2_from_hdf5(CASE / "input/fc2.h5", p2s_map=phono3py.primitive.p2s_map)
    phono3py.fc3 = read_fc3_from_hdf5(CASE / "input/fc3.h5", p2s_map=phono3py.primitive.p2s_map)
    phono3py.mesh_numbers = np.asarray(args.mesh, dtype=int)
    phono3py.init_phph_interaction()
    os.chdir(CASE / "results")
    phono3py.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=args.temperatures,
        write_kappa=True,
    )
    conductivity = phono3py.thermal_conductivity
    temperatures = np.asarray(conductivity.temperatures, dtype=float)
    kappa = np.asarray(conductivity.kappa, dtype=float)
    np.savez_compressed(
        CASE / "results/kappa-rta.npz",
        temperatures=temperatures,
        kappa=kappa,
        mesh=np.asarray(args.mesh, dtype=int),
    )
    np.savetxt(
        CASE / "results/kappa-rta.txt",
        np.column_stack((temperatures, kappa.reshape(len(temperatures), -1))),
        header="T_K kappa_xx kappa_yy kappa_zz kappa_xy kappa_xz kappa_yz",
    )
    print(f"phono3py RTA outputs were written under {CASE}")


if __name__ == "__main__":
    main()
