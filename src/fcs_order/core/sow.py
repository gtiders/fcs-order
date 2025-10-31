#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

from ..utils.vasp_order_common import (
    H,
    write_POSCAR,
    normalize_SPOSCAR,
    move_two_atoms,
    move_three_atoms,
)

from .bin import thirdorder_core, fourthorder_core  # type: ignore
from .tools import _prepare_calculation3, _prepare_calculation4


@click.command()
@click.argument("na", type=int)
@click.argument("nb", type=int)
@click.argument("nc", type=int)
@click.option(
    "--cutoff",
    type=str,
    required=True,
    help="Cutoff value (negative for nearest neighbors such as -8, positive for distance in nm such as 0.5)",
)
def sow3(na, nb, nc, cutoff):
    """
    Generate 3RD.POSCAR.* files for 3-phonon calculations.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation3(
        na, nb, nc, cutoff
    )
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} triplet equivalence classes found")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print(f"- {nruns} DFT runs are needed")

    print("Writing undisplaced coordinates to 3RD.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "3RD.SPOSCAR")
    width = len(str(4 * (len(list4) + 1)))
    namepattern = f"3RD.POSCAR.{{:0{width}d}}"
    print("Writing displaced coordinates to 3RD.POSCAR.*")
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


@click.command()
@click.argument("na", type=int)
@click.argument("nb", type=int)
@click.argument("nc", type=int)
@click.option(
    "--cutoff",
    type=str,
    required=True,
    help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
)
def sow4(na, nb, nc, cutoff):
    """
    Generate 4TH.POSCAR.* files for 4-phonon calculations.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation4(
        na, nb, nc, cutoff
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} quartet equivalence classes found")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred
    print(f"- {nruns} DFT runs are needed")
    print("Writing undisplaced coordinates to 4TH.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "4TH.SPOSCAR")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = "4TH.POSCAR.{{0:0{0}d}}".format(width)
    print("Writing displaced coordinates to 4TH.POSCAR.*")
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
