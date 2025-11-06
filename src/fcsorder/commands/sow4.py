#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils.order_common import H, write_POSCAR, normalize_SPOSCAR, move_three_atoms
from ..utils.prepare_calculation import prepare_calculation4

from ..core import fourthorder_core  # type: ignore


def sow4(na: int, nb: int, nc: int, cutoff: str):
    """
    Generate 4TH.POSCAR.* files for 4-phonon calculations.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions.
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm).
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation4(
        na, nb, nc, cutoff
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} quartet equivalence classes")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred
    print(f"Total DFT runs needed: {nruns}")
    print("Writing undisplaced coordinates to 4TH.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "4TH.SPOSCAR")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = "4TH.POSCAR.{{0:0{0}d}}".format(width)
    print("Writing displaced coordinates to 4TH.POSCAR.* files")
    for i, e in enumerate(list6):
        for n in range(8):
            isign = (-1) ** (n // 4)
            jsign = (-1) ** (n % 4 // 2)
            ksign = (-1) ** (n % 2)
            number = nirred * n + i + 1
            dsposcar = normalize_SPOSCAR(
                move_three_atoms(
                    sposcar,
                    e[2],
                    e[5],
                    isign * H,
                    e[1],
                    e[4],
                    jsign * H,
                    e[0],
                    e[3],
                    ksign * H,
                )
            )
            filename = namepattern.format(number)
            write_POSCAR(dsposcar, filename)
