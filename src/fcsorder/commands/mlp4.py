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
from ..core import fourthorder_core  # type: ignore
from ..utils.order_common import (
    H,
    build_unpermutation,
    normalize_SPOSCAR,
    move_three_atoms,
    write_ifcs4,
)
from ..utils.prepare_calculation import prepare_calculation4


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


def calculate_phonon_force_constants_4th(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    calculation,
    is_write: bool = False,
    is_sparse: bool = False,
):
    """
    Core function to calculate 4-phonon force constants.

    Args:
        na, nb, nc: Supercell size
        cutoff: Cutoff value
        calculation: Calculator object for force calculations
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method

    Returns:
        None (writes FORCE_CONSTANTS_4TH file)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation4(
        na, nb, nc, cutoff
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"Found {wedge.nlist} quartet equivalence classes")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred

    print(f"Total DFT runs needed: {nruns}")

    # Write sposcar positions and forces to 4TH.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("4TH.SPOSCAR.xyz", format="extxyz")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = f"4TH.POSCAR.{{:0{width}d}}.xyz"
    p = build_unpermutation(sposcar)
    forces = []
    indexs = []
    for i, e in enumerate(track(list6, description="Processing quartets")):
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
            if is_write:
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


# Create the main app
app = typer.Typer(
    help="Calculate 4-phonon force constants using machine learning potentials."
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
    Calculate 4-phonon force constants using NEP (Neural Evolution Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: NEP potential file path
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method for memory efficiency
        is_gpu: Use GPU calculator for faster computation
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

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, is_sparse)


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
    Calculate 4-phonon force constants using Deep Potential (DP) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: Deep Potential model file path
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method for memory efficiency
    """
    # DP calculator initialization
    print(f"Initializing DP calculator with potential: {potential}")
    try:
        from deepmd.calculator import DP

        calc = DP(model=potential)
    except ImportError:
        print("deepmd not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, is_sparse)


@app.command()
def hiphive(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="Hiphive potential file path (e.g. 'potential.fcp')"
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
    Calculate 4-phonon force constants using hiphive force constant potential.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
                    Note: The supercell size must be greater than or equal to the size used
                    for training the fcp potential. It cannot be smaller.
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: Hiphive potential file path
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method for memory efficiency
    """
    # Hiphive calculator initialization
    print(f"Using hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential
        from hiphive.calculators import ForceConstantCalculator

        fcp = ForceConstantPotential.read(potential)
        # Create a dummy atoms object to get force constants
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation4(na, nb, nc, cutoff)
        )
        atoms = get_atoms(normalize_SPOSCAR(sposcar))
        force_constants = fcp.get_force_constants(atoms)
        calc = ForceConstantCalculator(force_constants)
    except ImportError:
        print("hiphive not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, is_sparse)


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
    Calculate 4-phonon force constants using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: PolyMLP potential file path
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method for memory efficiency
    """
    # PolyMLP calculator initialization
    print(f"Using ploymp calculator with potential: {potential}")
    try:
        from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

        calc = PolymlpASECalculator(pot=potential)
    except ImportError:
        print("pypolymlp not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, is_sparse)
