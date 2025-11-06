#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys

# Third-party imports
import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator
from rich.progress import track

# Local imports
from ..core import thirdorder_core  # type: ignore
from ..utils.order_common import (
    H,
    build_unpermutation,
    normalize_SPOSCAR,
    move_two_atoms,
    write_ifcs3,
)
from ..utils.prepare_calculation import prepare_calculation3


def get_atoms(poscar, calc: Calculator = None) -> Atoms:
    symbols = np.repeat(poscar["elements"], poscar["numbers"]).tolist()
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=poscar["positions"].T,
        cell=poscar["lattvec"].T * 10,
        pbc=True,
    )
    if calc is not None:
        atoms.calc = calc
    return atoms


def calculate_phonon_force_constants(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    calculation,
    is_write: bool = False,
    is_sparse: bool = False,
):
    """
    Core function to calculate 3-phonon force constants.

    Args:
        na, nb, nc: Supercell size
        cutoff: Cutoff value
        calculation: Calculator object for force calculations
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method

    Returns:
        None (writes FORCE_CONSTANTS_3RD file)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation3(
        na, nb, nc, cutoff
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} triplet equivalence classes")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred

    print(f"Total DFT runs needed: {nruns}")

    # Write sposcar positions and forces to 3RD.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("3RD.SPOSCAR.xyz", format="extxyz")
    width = len(str(4 * (len(list4) + 1)))
    namepattern = f"3RD.POSCAR.{{:0{width}d}}.xyz"
    p = build_unpermutation(sposcar)
    forces = []
    indexs = []
    for i, e in enumerate(track(list4, description="Processing triplets")):
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
            if is_write:
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


# Create the main app
app = typer.Typer(
    help="Calculate 3-phonon force constants using machine learning potentials."
)


@app.command()
def nep(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="NEP potential file path (e.g. 'nep.txt')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", help="Use GPU calculator for faster computation"
    ),
):
    """
    Calculate 3-phonon force constants using NEP (Neural Evolution Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions\n
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)\n
        potential: NEP potential file path\n
        is_write: Whether to save intermediate files\n
        is_sparse: Use sparse tensor method for memory efficiency\n
        is_gpu: Use GPU calculator for faster computation\n
    """
    print(f"Initializing NEP calculator with potential: {potential}")
    try:
        from calorine.calculators import CPUNEP, GPUNEP

        if is_gpu:
            calc = GPUNEP(potential)
            print("Using GPU calculator for NEP")
        else:
            calc = CPUNEP(potential)
            print("Using CPU calculator for NEP")
    except ImportError:
        print("calorine not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants(na, nb, nc, cutoff, calc, is_write, is_sparse)


@app.command()
def dp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="DeepMD potential file path (e.g. 'model.pb')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
):
    """
    Calculate 3-phonon force constants using Deep Potential (DP) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions\n
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)\n
        potential: Deep Potential model file path\n
        is_write: Whether to save intermediate files\n
        is_sparse: Use sparse tensor method for memory efficiency\n
    """
    # DP calculator initialization
    print(f"Initializing DP calculator with potential: {potential}")
    try:
        from deepmd.calculator import DP

        calc = DP(model=potential)
    except ImportError:
        print("deepmd not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants(na, nb, nc, cutoff, calc, is_write, is_sparse)


@app.command()
def ploymp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(..., exists=True, help="PolyMLP potential file path"),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
):
    """
    Calculate 3-phonon force constants using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions\n
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)\n
        potential: PolyMLP potential file path\n
        is_write: Whether to save intermediate files\n
        is_sparse: Use sparse tensor method for memory efficiency\n
    """
    # PolyMLP calculator initialization
    print(f"Using ploymp calculator with potential: {potential}")
    try:
        from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

        calc = PolymlpASECalculator(pot=potential)
    except ImportError:
        print("pypolymlp not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants(na, nb, nc, cutoff, calc, is_write, is_sparse)
