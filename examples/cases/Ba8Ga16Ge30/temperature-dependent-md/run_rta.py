#!/usr/bin/env python3
"""Run phono3py RTA conductivity serially for Ba8Ga16Ge30 effective IFCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent
PRIMITIVE = ROOT / "input" / "reference.vasp"
DEFAULT_TEMPERATURES = (300, 400, 500, 600)
MESH = np.array((9, 9, 9), dtype=int)


def run_temperature(temperature: int) -> None:
    temperature_directory = ROOT / "md" / f"T{temperature}K"
    ifcs = temperature_directory / "mlfcs"
    fc2 = ifcs / "fc2.h5"
    fc3 = ifcs / "fc3.h5"
    missing = [str(path) for path in (fc2, fc3) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"T={temperature} K effective IFCs are missing; run fit.py first:\n"
            + "\n".join(missing)
        )

    cell, _ = read_crystal_structure(filename=str(PRIMITIVE), interface_mode="vasp")
    phonon = Phono3py(
        cell,
        supercell_matrix=np.diag((2, 2, 2)),
        primitive_matrix="auto",
        is_mesh_symmetry=True,
    )
    phonon.fc2 = read_fc2_from_hdf5(fc2, p2s_map=phonon.primitive.p2s_map)
    phonon.fc3 = read_fc3_from_hdf5(fc3, p2s_map=phonon.primitive.p2s_map)
    phonon.mesh_numbers = MESH
    phonon.init_phph_interaction()
    phonon.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=np.array(float(temperature)),
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
                "fc2": "../mlfcs/fc2.h5",
                "fc3": "../mlfcs/fc3.h5",
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
