"""Run phono3py RTA for an MLFCS FC2/FC3 pair on an 11^3 mesh."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parent
INPUT = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "input"
SOURCE = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "results"
OUTPUT = CASE / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("mlfcs", "phono3py"), default="mlfcs")
    parser.add_argument("--temperature", type=float, default=300.0)
    args = parser.parse_args()
    directory = SOURCE / ("phono3py-reference" if args.route == "phono3py" else "three-phonon")
    cell, _ = read_crystal_structure(filename=str(INPUT / "supercell.vasp"), interface_mode="vasp")
    phonon = Phono3py(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    p2s_map = phonon.primitive.p2s_map
    if args.route == "phono3py":
        fc2_path = directory / "fc2.h5"
        fc3_path = directory / "fc3.h5"
        temporary = None
    else:
        from mlfcs import read_hdf5

        temporary = tempfile.TemporaryDirectory(prefix="mlfcs-k4-rta-")
        temporary_path = Path(temporary.name)
        fc2_path = temporary_path / "fc2.h5"
        fc3_path = temporary_path / "fc3.h5"
        read_hdf5(SOURCE / "harmonic" / "mlfcs.h5").write(fc2_path, format="phonopy_hdf5", order=2)
        read_hdf5(directory / "mlfcs.h5").write(fc3_path, format="phono3py_hdf5", order=3)
    phonon.fc2 = read_fc2_from_hdf5(filename=str(fc2_path), p2s_map=p2s_map)
    phonon.fc3 = read_fc3_from_hdf5(filename=str(fc3_path), p2s_map=p2s_map)
    phonon.mesh_numbers = np.array([11, 11, 11], dtype=int)
    phonon.init_phph_interaction()
    phonon.run_thermal_conductivity(
        temperatures=[args.temperature], is_LBTE=False, write_kappa=True
    )
    result = phonon.thermal_conductivity
    output = OUTPUT / args.route
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
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
