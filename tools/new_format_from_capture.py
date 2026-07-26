"""Write captured reference IFC values using the generic ShengBTE writer."""

from __future__ import annotations

import argparse

import numpy as np
from ase.io import read

from mlfcs.core.geometry import make_supercell
from mlfcs.io.shengbte import write_shengbte


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("captured_npz")
    parser.add_argument("output")
    parser.add_argument("--order", type=int, choices=(3, 4), required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    args = parser.parse_args()

    primitive = read(args.structure, format="vasp")
    supercell, index = make_supercell(primitive, (2, 2, 2))
    shape = (index.n_primitive,) + (len(supercell),) * (args.order - 1) + (3,) * args.order
    force_constants = np.zeros(shape)
    with np.load(args.captured_npz) as captured:
        for key, value in zip(captured["keys"], captured["values"], strict=True):
            directions = tuple(map(int, key[: args.order]))
            atoms = tuple(map(int, key[args.order :]))
            force_constants[atoms + directions] = value
    write_shengbte(
        args.output,
        force_constants,
        supercell,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()
