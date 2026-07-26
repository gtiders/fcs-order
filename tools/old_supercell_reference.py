"""Capture old internal and grouped supercell arrays for ordering tests."""

from __future__ import annotations

import argparse

import numpy as np
from mlfcs.interface.phonopy_io import read_structure
from mlfcs.thirdorder.core import normalize_SPOSCAR
from mlfcs.thirdorder.thirdorder_common import gen_SPOSCAR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("output")
    args = parser.parse_args()
    primitive = read_structure(args.structure, interface="vasp").to_dict()
    internal = gen_SPOSCAR(primitive, 2, 2, 2)
    grouped = normalize_SPOSCAR(internal)
    np.savez(
        args.output,
        cell=internal["lattvec"].T * 10.0,
        internal_positions=(internal["lattvec"] @ internal["positions"]).T * 10.0,
        grouped_positions=(grouped["lattvec"] @ grouped["positions"]).T * 10.0,
        internal_types=internal["types"],
        grouped_types=grouped["types"],
    )


if __name__ == "__main__":
    main()
