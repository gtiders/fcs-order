#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
from ..core import thirdorder_core_py as thirdorder_core
from ..core import fourthorder_core_py as fourthorder_core
from .io_abstraction import read_structure
from .order_common import (
    SYMPREC,
    _parse_cutoff,
    _validate_cutoff,
    calc_dists,
    calc_frange,
    gen_SPOSCAR,
)


def prepare_calculation3(na, nb, nc, cutoff, poscar_path: str = "POSCAR"):
    _validate_cutoff(na, nb, nc)
    nneigh, frange = _parse_cutoff(cutoff)
    typer.echo("Reading structure")
    poscar = read_structure(poscar_path, in_format="auto")
    typer.echo("Analyzing the symmetries")
    symops = thirdorder_core.SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC
    )
    typer.echo(f"Symmetry group {symops.symbol} detected")
    typer.echo(f"{symops.translations.shape[0]} symmetry operations")
    typer.echo("Creating the supercell")
    sposcar = gen_SPOSCAR(poscar, na, nb, nc)
    typer.echo("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh is not None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        typer.echo(f"Automatic cutoff: {frange} nm")
    else:
        typer.echo(f"User-defined cutoff: {frange} nm")
    typer.echo("Looking for an irreducible set of third-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh


def prepare_calculation4(na, nb, nc, cutoff, poscar_path: str = "POSCAR"):
    """
    Validate the input parameters and prepare the calculation.
    """
    _validate_cutoff(na, nb, nc)
    nneigh, frange = _parse_cutoff(cutoff)
    typer.echo("Reading structure")
    poscar = read_structure(poscar_path, in_format="auto")
    typer.echo("Analyzing the symmetries")
    symops = fourthorder_core.SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC
    )
    typer.echo(f"Symmetry group {symops.symbol} detected")
    typer.echo(f"{symops.translations.shape[0]} symmetry operations")
    typer.echo("Creating the supercell")
    sposcar = gen_SPOSCAR(poscar, na, nb, nc)
    typer.echo("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh is not None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        typer.echo(f"Automatic cutoff: {frange} nm")
    else:
        typer.echo(f"User-defined cutoff: {frange} nm")
    typer.echo("Looking for an irreducible set of fourth-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh
