#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils.order_common import (
    H,
    write_POSCAR,
    normalize_SPOSCAR,
    move_two_atoms,
)
from ..utils.prepare_calculation import prepare_calculation3

from ..core import thirdorder_core  # type: ignore


def sow3(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
):
    """
    Generate 3RD.POSCAR.* files for 3-phonon calculations.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions.
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm).
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation3(
        na, nb, nc, cutoff
    )
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} triplet equivalence classes")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print(f"Total DFT runs needed: {nruns}")

    print("Writing undisplaced coordinates to 3RD.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "3RD.SPOSCAR")
    width = len(str(4 * (len(list4) + 1)))
    namepattern = f"3RD.POSCAR.{{:0{width}d}}"
    print("Writing displaced coordinates to 3RD.POSCAR.* files")
    for i, e in enumerate(list4):
        for n in range(4):
            isign = (-1) ** (n // 2)
            jsign = -((-1) ** (n % 2))
            number = nirred * n + i + 1
            dsposcar = normalize_SPOSCAR(
                move_two_atoms(sposcar, e[1], e[3], isign * H, e[0], e[2], jsign * H)
            )
            filename = namepattern.format(number)
            write_POSCAR(dsposcar, filename)
