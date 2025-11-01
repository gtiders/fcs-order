#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import sys
import numpy as np

from ..utils.vasp_order_common import (
    H,
    build_unpermutation,
    normalize_SPOSCAR,
    move_two_atoms,
    move_three_atoms,
    write_ifcs3,
    write_ifcs4,
)
from ..utils.calculator import get_atoms, initialize_calculator

from .tools import _prepare_calculation3, _prepare_calculation4
from .bin import thirdorder_core, fourthorder_core  # type: ignore


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
@click.option(
    "--calc",
    type=click.Choice(["nep", "dp", "hiphive", "ploymp"], case_sensitive=False),
    default=None,
    help="Calculator to use (nep or dp or hiphive or ploymp)",
)
@click.option(
    "--potential",
    type=click.Path(exists=True),
    default=None,
    help="Potential file to use (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml')",
)
@click.option(
    "--if_write",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to save intermediate files during the calculation process",
)
@click.option(
    "--is_sparse",
    type=bool,
    is_flag=True,
    default=False,
    help="Use sparse tensor method for memory efficiency",
)
def mlp3(na, nb, nc, cutoff, calc, potential, if_write, is_sparse):
    """
    Directly calculate 3-phonon force constants using machine learning potential functions based on thirdorder.
    Accuracy depends on potential function precision and supercell size; it is recommended to use a larger supercell.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        calc: calculator type, optional values are nep, dp, hiphive, ploymp
        potential: potential file path, corresponding to different file formats based on calc type
        if_write: whether to save intermediate files, default is not to save
        is_sparse: use sparse tensor method for memory efficiency, default is False
    """
    # Validate that calc and potential must be provided together
    if (calc is not None and potential is None) or (
        calc is None and potential is not None
    ):
        raise click.BadParameter("--calc and --potential must be provided together")

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
    # If calculator type and potential file are specified, set up the calculator
    if calc is not None and potential is not None:
        calculation = initialize_calculator(calc, potential, get_atoms(sposcar))
    else:
        print("No calculator provided")
        sys.exit(1)

    print(f"- {nruns} force calculations are runing!")
    # Write sposcar positions and forces to 3RD.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("3RD.SPOSCAR.xyz", format="extxyz")
    width = len(str(4 * (len(list4) + 1)))
    namepattern = f"3RD.POSCAR.{{:0{width}d}}.xyz"
    p = build_unpermutation(sposcar)
    forces = []
    indexs = []
    for i, e in enumerate(list4):
        for n in range(4):
            isign = (-1) ** (n // 2)
            jsign = -((-1) ** (n % 2))
            number = nirred * n + i + 1
            dsposcar = normalize_SPOSCAR(
                move_two_atoms(sposcar, e[1], e[3], isign * H, e[0], e[2], jsign * H)
            )
            atoms = get_atoms(dsposcar, calculation)
            forces.append(atoms.get_forces()[p, :])
            filename = namepattern.format(number)
            indexs.append(number)
            if if_write:
                atoms.write(filename, format="extxyz")

    # sorted indexs and forces
    sorted_indices = np.argsort(indexs)
    indexs = [indexs[i] for i in sorted_indices]
    forces = [forces[i] for i in sorted_indices]
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
    help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
)
@click.option(
    "--calc",
    type=click.Choice(["nep", "dp", "hiphive", "ploymp"], case_sensitive=False),
    default=None,
    help="Calculator to use (nep or dp or hiphive or ploymp)",
)
@click.option(
    "--potential",
    type=click.Path(exists=True),
    default=None,
    help="Potential file to use (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml')",
)
@click.option(
    "--if_write",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to save intermediate files during the calculation process",
)
@click.option(
    "--is_sparse",
    type=bool,
    is_flag=True,
    default=False,
    help="Use sparse tensor method for memory efficiency",
)
def mlp4(na, nb, nc, cutoff, calc, potential, if_write, is_sparse):
    """
    Directly calculate 4-phonon force constants using machine learning potential functions based on fourthorder.
    Accuracy depends on potential function precision and supercell size; it is recommended to use a larger supercell.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        calc: calculator type, optional values are nep, dp, hiphive, ploymp
        potential: potential file path, corresponding to different file formats based on calc type
        if_write: whether to save intermediate files, default is not to save
        is_sparse: use sparse tensor method for memory efficiency, default is False
    """
    # Validate that calc and potential must be provided together
    if (calc is not None and potential is None) or (
        calc is None and potential is not None
    ):
        raise click.BadParameter("--calc and --potential must be provided together")
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
    # If calculator type and potential file are specified, set up the calculator
    if calc is not None and potential is not None:
        calculation = initialize_calculator(calc, potential, get_atoms(sposcar))
    else:
        print("No calculator provided")
        sys.exit(1)
    print(f"- {nruns} force calculations are runing!")
    # Write sposcar positions and forces to 4TH.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("4TH.SPOSCAR.xyz", format="extxyz")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = "4TH.POSCAR.{{0:0{0}d}}.xyz".format(width)
    p = build_unpermutation(sposcar)
    forces = []
    indexs = []
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

            atoms = get_atoms(dsposcar, calculation)
            forces.append(atoms.get_forces()[p, :])
            filename = namepattern.format(number)
            indexs.append(number)
            if if_write:
                atoms.write(filename, format="extxyz")
    # sorted indexs and forces
    sorted_indices = np.argsort(indexs)
    indexs = [indexs[i] for i in sorted_indices]
    forces = [forces[i] for i in sorted_indices]
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
