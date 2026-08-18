"""Compare phono3py RTA conductivity for MLFCS and phono3py FC files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parent.parent


def run_case(name, supercell, fc2, fc3, mesh, temperatures, primitive=None):
    source = primitive if primitive is not None else supercell
    cell, _ = read_crystal_structure(filename=str(source), interface_mode="vasp")
    matrix = np.diag((2, 2, 3)) if primitive is not None else np.eye(3, dtype=int)
    phono3py = Phono3py(
        cell, supercell_matrix=matrix, primitive_matrix="auto", is_mesh_symmetry=True
    )
    phono3py.fc2 = read_fc2_from_hdf5(fc2, p2s_map=phono3py.primitive.p2s_map)
    phono3py.fc3 = read_fc3_from_hdf5(fc3, p2s_map=phono3py.primitive.p2s_map)
    phono3py.mesh_numbers = np.asarray(mesh, dtype=int)
    phono3py.init_phph_interaction()
    phono3py.run_thermal_conductivity(is_LBTE=False, temperatures=temperatures, write_kappa=True)
    conductivity = phono3py.thermal_conductivity
    values = np.asarray(conductivity.kappa, dtype=float)
    output = CASE / f"thermal-conductivity-{name}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "kappa-rta.npz",
        temperatures=np.asarray(conductivity.temperatures),
        kappa=values,
        mesh=np.asarray(mesh),
    )
    np.savetxt(
        output / "kappa-rta.txt",
        np.column_stack((conductivity.temperatures, values.reshape(len(values), -1))),
        header="T_K kappa_xx kappa_yy kappa_zz kappa_xy kappa_xz kappa_yz",
    )
    print(f"{name}: wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11))
    parser.add_argument("--temperatures", nargs="+", type=float, default=(300.0,))
    parser.add_argument("--supercell", type=Path, default=CASE / "supercell.vasp")
    parser.add_argument("--primitive", type=Path, default=CASE / "primitive.vasp")
    parser.add_argument("--case", choices=("mlfcs", "phono3py", "both"), default="both")
    args = parser.parse_args()
    if args.case in ("mlfcs", "both"):
        run_case(
            "mlfcs",
            args.supercell,
            CASE / "harmonic/fc2.h5",
            CASE / "three-phonon/fc3.h5",
            args.mesh,
            args.temperatures,
        )
    if args.case in ("phono3py", "both"):
        run_case(
            "phono3py",
            args.supercell,
            CASE / "phono3py-reference/fc2.h5",
            CASE / "phono3py-reference/fc3.h5",
            args.mesh,
            args.temperatures,
            primitive=args.primitive,
        )


if __name__ == "__main__":
    main()
