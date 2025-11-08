#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..core import fourthorder_core  # type: ignore
from ..utils.order_common import (
    H,
    build_unpermutation,
    read_forces,
    write_ifcs4,
)
from ..utils.prepare_calculation import prepare_calculation4


def reap4(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    vaspruns: list[str],
    is_sparse: bool = False,
    poscar_path: str = "POSCAR",
) -> None:
    """
    Extract 4-phonon force constants from VASP calculation results.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm)
        vaspruns: Paths to vasprun.xml files from VASP calculations, in order
        is_sparse: Use sparse tensor method for memory efficiency, default is False
        poscar_path: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation4(
        na, nb, nc, cutoff, poscar_path
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} quartet equivalence classes")
    list6 = wedge.build_list4()
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    nirred = len(list6)
    nruns = 8 * nirred
    print(f"Total DFT runs needed: {nruns}")
    if len(vaspruns) != nruns:
        raise ValueError(
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
