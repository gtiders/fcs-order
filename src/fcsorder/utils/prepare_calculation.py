#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..core import thirdorder_core, fourthorder_core  # type: ignore
from .order_common import (
    SYMPREC,
    _validate_cutoff,
    _parse_cutoff,
    read_POSCAR,
    gen_SPOSCAR,
    calc_dists,
    calc_frange,
)


def prepare_calculation3(na, nb, nc, cutoff):
    _validate_cutoff(na, nb, nc)
    nneigh, frange = _parse_cutoff(cutoff)

    print("Reading POSCAR")
    poscar = read_POSCAR()
    print("Analyzing the symmetries")
    symops = thirdorder_core.SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC
    )
    print(f"- Symmetry group {symops.symbol} detected")
    print(f"- {symops.translations.shape[0]} symmetry operations")
    print("Creating the supercell")
    sposcar = gen_SPOSCAR(poscar, na, nb, nc)
    print("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh is not None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        print(f"- Automatic cutoff: {frange} nm")
    else:
        print(f"- User-defined cutoff: {frange} nm")
    print("Looking for an irreducible set of third-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh


def prepare_calculation4(na, nb, nc, cutoff):
    """
    Validate the input parameters and prepare the calculation.
    """
    _validate_cutoff(na, nb, nc)
    nneigh, frange = _parse_cutoff(cutoff)
    print("Reading POSCAR")
    poscar = read_POSCAR()
    print("Analyzing the symmetries")
    symops = fourthorder_core.SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC
    )
    print(f"- Symmetry group {symops.symbol} detected")
    print(f"- {symops.translations.shape[0]} symmetry operations")
    print("Creating the supercell")
    sposcar = gen_SPOSCAR(poscar, na, nb, nc)
    print("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh is not None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        print(f"- Automatic cutoff: {frange} nm")
    else:
        print(f"- User-defined cutoff: {frange} nm")
    print("Looking for an irreducible set of fourth-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh
