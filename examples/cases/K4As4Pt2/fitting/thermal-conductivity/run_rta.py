"""Run phono3py RTA for an MLFCS FC2/FC3 pair on an 11^3 mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", choices=("three-body", "four-body"), default="three-body")
    parser.add_argument("--temperature", type=float, default=300.0)
    args = parser.parse_args()
    directory = CASE / "anharmonic" / args.fit
    cell, _ = read_crystal_structure(filename=str(CASE / "reference.vasp"), interface_mode="vasp")
    phonon = Phono3py(cell, supercell_matrix=np.eye(3, dtype=int), primitive_matrix="auto")
    p2s_map = phonon.primitive.p2s_map
    phonon.fc2 = read_fc2_from_hdf5(filename=str(directory / "fc2.h5"), p2s_map=p2s_map)
    phonon.fc3 = read_fc3_from_hdf5(filename=str(directory / "fc3.h5"), p2s_map=p2s_map)
    phonon.mesh_numbers = np.array([11, 11, 11], dtype=int)
    phonon.init_phph_interaction()
    phonon.run_thermal_conductivity(
        temperatures=[args.temperature], is_LBTE=False, write_kappa=True
    )
    result = phonon.thermal_conductivity
    output = CASE / "thermal-conductivity" / args.fit
    output.mkdir(parents=True, exist_ok=True)
    temperatures = np.asarray(result.temperatures, dtype=float)
    kappa = np.asarray(result.kappa, dtype=float)
    np.savez_compressed(
        output / "kappa-rta.npz",
        temperatures=temperatures,
        kappa=kappa,
        mesh=np.array([11, 11, 11], dtype=int),
    )
    np.savetxt(
        output / "kappa-rta.txt",
        np.column_stack((temperatures, kappa.reshape(len(temperatures), -1))),
        header="T_K kappa_xx kappa_yy kappa_zz kappa_xy kappa_xz kappa_yz",
    )
    print(kappa)


if __name__ == "__main__":
    main()
