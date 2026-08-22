#!/usr/bin/env python3
"""Calculate SnSe thermal conductivity with phono3py BTE-RTA."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supercell", type=Path, default=CASE / "mlfcs/fd_supercell.vasp")
    parser.add_argument("--fc2", type=Path, default=CASE / "mlfcs/fc2/fc2.h5")
    parser.add_argument("--fc3", type=Path, default=CASE / "mlfcs/fc3/fc3.h5")
    parser.add_argument("--output-dir", type=Path, default=CASE / "mlfcs/rta")
    parser.add_argument("--mesh", nargs=3, type=int, default=(16, 16, 16))
    parser.add_argument("--temperatures", nargs="+", type=float, default=(300.0,))
    parser.add_argument(
        "--boundary-mfp",
        type=float,
        default=1.0,
        metavar="MICROMETER",
        help="boundary mean free path in micrometers (default: 1.0)",
    )
    parser.add_argument(
        "--isotope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include natural-isotope scattering (default: enabled)",
    )
    args = parser.parse_args()

    for path in (args.supercell, args.fc2, args.fc3):
        if not path.is_file():
            raise FileNotFoundError(path)
    cell, _ = read_crystal_structure(filename=str(args.supercell), interface_mode="vasp")
    phono3py = Phono3py(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    phono3py.fc2 = read_fc2_from_hdf5(
        args.fc2, p2s_map=phono3py.primitive.p2s_map
    )
    phono3py.fc3 = read_fc3_from_hdf5(
        args.fc3, p2s_map=phono3py.primitive.p2s_map
    )
    phono3py.mesh_numbers = np.asarray(args.mesh, dtype=int)
    phono3py.init_phph_interaction()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)
    phono3py.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=args.temperatures,
        is_isotope=args.isotope,
        boundary_mfp=args.boundary_mfp,
        write_kappa=True,
    )
    conductivity = phono3py.thermal_conductivity
    temperatures = np.asarray(conductivity.temperatures, dtype=float)
    kappa = np.asarray(conductivity.kappa, dtype=float)
    np.savez_compressed(
        args.output_dir / "kappa-rta.npz",
        temperatures=temperatures,
        kappa=kappa,
        mesh=np.asarray(args.mesh, dtype=int),
        boundary_mfp_um=args.boundary_mfp,
        isotope_scattering=args.isotope,
    )
    np.savetxt(
        args.output_dir / "kappa-rta.txt",
        np.column_stack((temperatures, kappa.reshape(len(temperatures), -1))),
        header="T_K kappa_xx kappa_yy kappa_zz kappa_xy kappa_xz kappa_yz",
    )
    print(f"wrote RTA results to {args.output_dir}")


if __name__ == "__main__":
    main()
