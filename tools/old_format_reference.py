"""Write a captured legacy IFC map with the installed legacy writer."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("captured_npz")
    parser.add_argument("output")
    parser.add_argument("--order", type=int, choices=(3, 4), required=True)
    parser.add_argument("--cutoff", type=float, required=True, help="angstrom")
    args = parser.parse_args()

    if args.order == 3:
        from mlfcs.thirdorder.thirdorder_common import calc_dists, gen_SPOSCAR, write_ifcs
    else:
        from mlfcs.fourthorder.fourthorder_common import calc_dists, gen_SPOSCAR, write_ifcs
    from mlfcs.interface.phonopy_io import read_structure

    primitive = read_structure(args.structure, interface="vasp").to_dict()
    supercell = gen_SPOSCAR(primitive, 2, 2, 2)
    distances, counts, shifts = calc_dists(supercell)
    with np.load(args.captured_npz) as captured:
        mapping = {
            tuple(map(int, key)): float(value)
            for key, value in zip(captured["keys"], captured["values"], strict=True)
        }
    write_ifcs(
        mapping,
        primitive,
        supercell,
        distances,
        counts,
        shifts,
        args.cutoff / 10.0,
        args.output,
    )


if __name__ == "__main__":
    main()
