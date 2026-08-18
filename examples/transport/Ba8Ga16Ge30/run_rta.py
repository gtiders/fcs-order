#!/usr/bin/env python3
"""Run phono3py RTA conductivity serially for Ba8Ga16Ge30 effective IFCs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

from mlfcs import build_supercell, read_hdf5

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
MD_ROOT = ROOT.parent.parent / "md" / "Ba8Ga16Ge30" / "results"
FIT_ROOT = ROOT.parent.parent / "fitting" / "Ba8Ga16Ge30" / "results"
PRIMITIVE = INPUT / "reference.vasp"
MLFCS_SUPERCELL = INPUT / "reference_supercell.vasp"
DEFAULT_TEMPERATURES = (300, 400, 500, 600)
MESH = np.array((3, 3, 3), dtype=int)


def run_temperature(temperature: int) -> None:
    temperature_directory = MD_ROOT / f"T{temperature}K"
    ifcs = FIT_ROOT / f"T{temperature}K" / "mlfcs"
    native = ifcs / "mlfcs.h5"
    missing = [
        str(path) for path in (native, temperature_directory / "nve.extxyz") if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"T={temperature} K effective IFCs are missing; run fit.py first:\n"
            + "\n".join(missing)
        )

    if not MLFCS_SUPERCELL.is_file():
        primitive = read(PRIMITIVE)
        reference = build_supercell(primitive, (2, 2, 2))
        write(MLFCS_SUPERCELL, reference, format="vasp", direct=True, sort=False, vasp5=True)
        print(f"Wrote MLFCS supercell for phono3py: {MLFCS_SUPERCELL}")
    cell, _ = read_crystal_structure(filename=str(MLFCS_SUPERCELL), interface_mode="vasp")
    phonon = Phono3py(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    with tempfile.TemporaryDirectory(prefix="mlfcs-bagg-rta-") as temporary:
        temporary_path = Path(temporary)
        fc2 = temporary_path / "fc2.h5"
        fc3 = temporary_path / "fc3.h5"
        force_constants = read_hdf5(native)
        force_constants.write(fc2, format="phonopy_hdf5", order=2)
        force_constants.write(fc3, format="phono3py_hdf5", order=3)
        phonon.fc2 = read_fc2_from_hdf5(fc2, p2s_map=phonon.primitive.p2s_map)
        phonon.fc3 = read_fc3_from_hdf5(fc3, p2s_map=phonon.primitive.p2s_map)
        phonon.mesh_numbers = MESH
        phonon.init_phph_interaction()
        phonon.run_thermal_conductivity(
            is_LBTE=False,
            temperatures=np.array([float(temperature)]),
            write_kappa=False,
        )
    conductivity = phonon.thermal_conductivity
    temperatures = np.asarray(conductivity.temperatures, dtype=float)
    kappa = np.asarray(conductivity.kappa, dtype=float)

    output = temperature_directory / "thermal-conductivity"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "kappa-rta.npz", temperatures=temperatures, kappa=kappa, mesh=MESH)
    np.savetxt(
        output / "kappa-rta.txt",
        np.column_stack((temperatures, kappa.reshape(len(kappa), -1))),
        header="T_K kappa_xx kappa_yy kappa_zz kappa_xy kappa_xz kappa_yz",
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "method": "phono3py RTA",
                "temperature_K": temperature,
                "mesh": MESH.tolist(),
                "force_constants": (
                    f"../../../../fitting/Ba8Ga16Ge30/results/"
                    f"T{temperature}K/mlfcs/mlfcs.h5"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    print(f"T={temperature} K: wrote {output / 'kappa-rta.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", nargs="+", type=int, default=DEFAULT_TEMPERATURES)
    args = parser.parse_args()
    unsupported = sorted(set(args.temperatures) - set(DEFAULT_TEMPERATURES))
    if unsupported:
        raise ValueError(f"this case supports 300, 400, 500, and 600 K; got {unsupported}")
    for temperature in args.temperatures:
        run_temperature(temperature)


if __name__ == "__main__":
    main()
