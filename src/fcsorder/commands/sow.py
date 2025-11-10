#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer

from ..utils.order_common import (
    H,
    write_POSCAR,
    normalize_SPOSCAR,
    move_two_atoms,
    move_three_atoms,
)
from ..utils.prepare_calculation import prepare_calculation3, prepare_calculation4

from ..core import thirdorder_core  # type: ignore
from ..core import fourthorder_core  # type: ignore


def sow(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    order: int = 3,
    poscar_path: str = "POSCAR",
):
    """
    Generate displaced POSCAR files for 3-phonon or 4-phonon calculations.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions.
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm).
        order: 3 for third-order (3-phonon), 4 for fourth-order (4-phonon).
        poscar_path: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'.
    """
    if order == 3:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation3(na, nb, nc, cutoff, poscar_path)
        )
        wedge = thirdorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.print(f"Found {wedge.nlist} triplet equivalence classes")
        list4 = wedge.build_list4()
        nirred = len(list4)
        nruns = 4 * nirred
        typer.print(f"Total DFT runs needed: {nruns}")

        typer.print("Writing undisplaced coordinates to 3RD.SPOSCAR")
        write_POSCAR(normalize_SPOSCAR(sposcar), "3RD.SPOSCAR")
        width = len(str(4 * (len(list4) + 1)))
        namepattern = f"3RD.POSCAR.{{:0{width}d}}"
        typer.print("Writing displaced coordinates to 3RD.POSCAR.* files")
        for i, e in enumerate(list4):
            for n in range(4):
                isign = (-1) ** (n // 2)
                jsign = -((-1) ** (n % 2))
                number = nirred * n + i + 1
                dsposcar = normalize_SPOSCAR(
                    move_two_atoms(
                        sposcar, e[1], e[3], isign * H, e[0], e[2], jsign * H
                    )
                )
                filename = namepattern.format(number)
                write_POSCAR(dsposcar, filename)
        return

    if order == 4:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation4(na, nb, nc, cutoff, poscar_path)
        )
        wedge = fourthorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.print(f"Found {wedge.nlist} quartet equivalence classes")
        list6 = wedge.build_list4()
        nirred = len(list6)
        nruns = 8 * nirred
        typer.print(f"Total DFT runs needed: {nruns}")
        typer.print("Writing undisplaced coordinates to 4TH.SPOSCAR")
        write_POSCAR(normalize_SPOSCAR(sposcar), "4TH.SPOSCAR")
        width = len(str(8 * (len(list6) + 1)))
        namepattern = "4TH.POSCAR.{{0:0{0}d}}".format(width)
        typer.print("Writing displaced coordinates to 4TH.POSCAR.* files")
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
        return

    raise typer.BadParameter("--order must be either 3 or 4")
