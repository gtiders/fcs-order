#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import typer

from fcsorder.core import thirdorder_core, fourthorder_core
from fcsorder.core.thirdorder_core import prepare_calculation3
from fcsorder.core.fourthorder_core import prepare_calculation4

from fcsorder.core.domain.common import (
    H,
    build_unpermutation,
    write_ifcs3,
    write_ifcs4,
)

from fcsorder.io.io_abstraction import read_atoms


def reap(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    vaspruns: list[str],
    order: int = 3,
    poscar_path: str = "POSCAR",
) -> None:
    """
    Extract 3-phonon (order=3) or 4-phonon (order=4) force constants from forces stored in
    ASE-readable files (e.g., VASP vasprun.xml/OUTCAR, extxyz with forces, etc.). The sequence of
    files must match the displaced structure order produced by 'sow'.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm)
        vaspruns: Paths to force files readable by ASE (vasprun.xml/OUTCAR, extxyz, etc.), in order
        order: 3 or 4
        poscar_path: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ)
    """
    if order == 3:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation3(na, nb, nc, cutoff, poscar_path)
        )
        natoms = len(poscar["types"])
        ntot = natoms * na * nb * nc
        wedge = thirdorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.echo(f"Found {wedge.nlist} triplet equivalence classes")
        list4 = wedge.build_list4()
        nirred = len(list4)
        nruns = 4 * nirred
        typer.echo(f"Total DFT runs needed: {nruns}")

        if len(vaspruns) != nruns:
            raise ValueError(
                f"Error: {nruns} force files were expected, got {len(vaspruns)}"
            )

        typer.echo("Reading forces from ASE-readable files")
        p = build_unpermutation(sposcar)
        forces = []
        for fpath in vaspruns:
            forces.append(read_atoms(fpath).get_forces()[p, :])
            typer.echo(f"- {fpath} read successfully")
            res = forces[-1].mean(axis=0)
            typer.echo("- \t Average force:")
            typer.echo(f"- \t {res} eV/(A * atom)")
        typer.echo("Computing an irreducible set of anharmonic force constants")
        phipart = np.zeros((3, nirred, ntot))
        for i, e in enumerate(list4):
            for n in range(4):
                isign = (-1) ** (n // 2)
                jsign = -((-1) ** (n % 2))
                number = nirred * n + i
                phipart[:, i, :] -= isign * jsign * forces[number].T
        phipart /= 400.0 * H * H
        typer.echo("Reconstructing the full array")
        phifull = thirdorder_core.reconstruct_ifcs(
            phipart, wedge, list4, poscar, sposcar
        )
        typer.echo("Writing the constants to FORCE_CONSTANTS_3RD")
        write_ifcs3(
            phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_3RD"
        )
        return

    if order == 4:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation4(na, nb, nc, cutoff, poscar_path)
        )
        wedge = fourthorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.echo(f"Found {wedge.nlist} quartet equivalence classes")
        list6 = wedge.build_list4()
        natoms = len(poscar["types"])
        ntot = natoms * na * nb * nc
        nirred = len(list6)
        nruns = 8 * nirred
        typer.echo(f"Total DFT runs needed: {nruns}")
        if len(vaspruns) != nruns:
            raise ValueError(
                f"Error: {nruns} force files were expected, got {len(vaspruns)}"
            )
        typer.echo("Reading forces from ASE-readable files")
        p = build_unpermutation(sposcar)
        forces = []
        for fpath in vaspruns:
            forces.append(read_forces(fpath)[p, :])
            typer.echo(f"- {fpath} read successfully")
            res = forces[-1].mean(axis=0)
            typer.echo("- \t Average force:")
            typer.echo(f"- \t {res} eV/(A * atom)")
        typer.echo("Computing an irreducible set of anharmonic force constants")
        phipart = np.zeros((3, nirred, ntot))
        for i, e in enumerate(list6):
            for n in range(8):
                isign = (-1) ** (n // 4)
                jsign = (-1) ** (n % 4 // 2)
                ksign = (-1) ** (n % 2)
                number = nirred * n + i
                phipart[:, i, :] -= isign * jsign * ksign * forces[number].T
        phipart /= 8000.0 * H * H * H
        typer.echo("Reconstructing the full array")
        phifull = fourthorder_core.reconstruct_ifcs(
            phipart, wedge, list6, poscar, sposcar
        )
        typer.echo("Writing the constants to FORCE_CONSTANTS_4TH")
        write_ifcs4(
            phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_4TH"
        )
        return

    raise typer.BadParameter("--order must be either 3 or 4")
