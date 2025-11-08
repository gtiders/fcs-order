#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..core import thirdorder_core  # type: ignore
from ..utils.order_common import (
    H,
    build_unpermutation,
    read_forces,
    write_ifcs3,
)
from ..utils.prepare_calculation import prepare_calculation3


def reap3(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    vaspruns: list[str],
    is_sparse: bool = False,
    poscar_path: str = "POSCAR",
) -> None:
    """
    Extract 3-phonon force constants from VASP calculation results.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm)
        vaspruns: Paths to vasprun.xml files from VASP calculations, in order
        is_sparse: Use sparse tensor method for memory efficiency, default is False
        poscar_path: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation3(
        na, nb, nc, cutoff, poscar_path
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} triplet equivalence classes")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
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
