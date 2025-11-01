#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import numpy as np

from .bin import thirdorder_core, fourthorder_core  # type: ignore
from ..utils.vasp_order_common import (
    H,
    build_unpermutation,
    read_forces,
    write_ifcs3,
    write_ifcs4,
)
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
@click.option(
    "--is_sparse",
    type=bool,
    is_flag=True,
    default=False,
    help="Use sparse tensor method for memory efficiency",
)
@click.argument("vaspruns", type=click.Path(exists=True), nargs=-1, required=True)
def reap3(na, nb, nc, cutoff, vaspruns, is_sparse):
    """
    Extract 3-phonon force constants from VASP calculation results.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        is_sparse: use sparse tensor method for memory efficiency, default is False
        vaspruns: paths to vasprun.xml files from VASP calculations, in order,such as vasprun.0001.xml,vasprun.0002.xml,...
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
        _prepare_calculation3(na, nb, nc, cutoff)
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} triplet equivalence classes found")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print(f"- {nruns} DFT runs are needed")

    if len(vaspruns) != nruns:
        raise click.ClickException(
            f"Error: {nruns} vasprun.xml files were expected, got {len(vaspruns)}"
        )

    print("Reading the forces")
    p = build_unpermutation(sposcar)
    forces = []
    for f in vaspruns:
        forces.append(read_forces(f)[p, :])
        print(f"- {f} read successfully")
        res = forces[-1].mean(axis=0)
        print("- \t Average force:")
        print(f"- \t {res} eV/(A * atom)")
    print("Computing an irreducible set of anharmonic force constants")
    phipart = np.zeros((3, nirred, ntot))
    for i, e in enumerate(list4):
        for n in range(4):
            isign = (-1) ** (n // 2)
            jsign = -((-1) ** (n % 2))
            number = nirred * n + i
            phipart[:, i, :] -= isign * jsign * forces[number].T
    phipart /= 400.0 * H * H
    print("Reconstructing the full array")
    phifull = thirdorder_core.reconstruct_ifcs(
        phipart, wedge, list4, poscar, sposcar, is_sparse
    )
    print("Writing the constants to FORCE_CONSTANTS_3RD")
    write_ifcs3(
        phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_3RD"
    )


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
@click.option(
    "--is_sparse",
    type=bool,
    is_flag=True,
    default=False,
    help="Use sparse tensor method for memory efficiency",
)
@click.argument("vaspruns", type=click.Path(exists=True), nargs=-1, required=True)
def reap4(na, nb, nc, cutoff, vaspruns, is_sparse):
    """
    Extract 4-phonon force constants from VASP calculation results.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        is_sparse: use sparse tensor method for memory efficiency, default is False
        vaspruns: paths to vasprun.xml files from VASP calculations, in order,such as vasprun.0001.xml,vasprun.0002.xml,...
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
        _prepare_calculation4(na, nb, nc, cutoff)
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} quartet equivalence classes found")
    list6 = wedge.build_list4()
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    nirred = len(list6)
    nruns = 8 * nirred
    if len(vaspruns) != nruns:
        raise click.ClickException(
            f"Error: {nruns} vasprun.xml files were expected, got {len(vaspruns)}"
        )
    print("Reading the forces")
    p = build_unpermutation(sposcar)
    forces = []
    for f in vaspruns:
        forces.append(read_forces(f)[p, :])
        print(f"- {f} read successfully")
        res = forces[-1].mean(axis=0)
        print("- \t Average force:")
        print(f"- \t {res} eV/(A * atom)")
    print("Computing an irreducible set of anharmonic force constants")
    phipart = np.zeros((3, nirred, ntot))
    for i, e in enumerate(list6):
        for n in range(8):
            isign = (-1) ** (n // 4)
            jsign = (-1) ** (n % 4 // 2)
            ksign = (-1) ** (n % 2)
            number = nirred * n + i
            phipart[:, i, :] -= isign * jsign * ksign * forces[number].T
    phipart /= 8000.0 * H * H * H
    print("Reconstructing the full array")
    phifull = fourthorder_core.reconstruct_ifcs(
        phipart, wedge, list6, poscar, sposcar, is_sparse
    )
    print("Writing the constants to FORCE_CONSTANTS_4TH")
    write_ifcs4(
        phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_4TH"
    )
