#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys

# Third-party imports
import typer
import numpy as np
from ase.io import read
from ase.calculators.calculator import Calculator

# Local imports
from ..core import get_force_constants


def calculate_phonon_force_constants_2nd(
    supercell_array: np.ndarray,
    calculation: Calculator,
    outfile: str = "FORCE_CONSTANTS_2ND",
):
    """
    Core function to calculate 2-phonon force constants.

    Args:
        supercell_array: Supercell expansion matrix
        calculation: Calculator object for force calculations
        outfile: Output file name for force constants

    Returns:
        None (writes FORCE_CONSTANTS_2ND file)
    """

    atoms = read("POSCAR")

    try:
        from phonopy import Phonopy
        from phonopy.file_IO import write_FORCE_CONSTANTS
    except Exception as e:
        print(f"Error importing Phonopy module from phonopy: {e}")
        sys.exit(1)

    phonon: Phonopy = get_force_constants(atoms, calculation, supercell_array)
    fcs2 = phonon.force_constants
    write_FORCE_CONSTANTS(fcs2, filename=outfile)


# Create the main app
app = typer.Typer(
    help="Calculate 2-phonon force constants using machine learning potentials."
)


@app.command()
def nep(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="NEP potential file path (e.g. 'nep.txt')"
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", help="Use GPU calculator for faster computation"
    ),
):
    """
    Calculate 2-phonon force constants using NEP (Neural Evolution Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: NEP potential file path\n
        outfile: Output file path for force constants\n
        is_gpu: Use GPU calculator for faster computation\n
    """
    # Validate supercell matrix dimensions
    if len(supercell_matrix) not in [3, 9]:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # Convert supercell matrix to 3x3 format
    if len(supercell_matrix) == 3:
        # Diagonal matrix: [na, nb, nc] -> [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        na, nb, nc = supercell_matrix
        supercell_array = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    else:
        # Full 3x3 matrix: reshape 9 numbers into 3x3
        supercell_array = np.array(supercell_matrix).reshape(3, 3)

    # NEP calculator initialization
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

    calculate_phonon_force_constants_2nd(supercell_array, calc, outfile)


@app.command()
def dp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="DeepMD potential file path (e.g. 'model.pb')"
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
):
    """
    Calculate 2-phonon force constants using Deep Potential (DP) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: Deep Potential model file path\n
        outfile: Output file path for force constants\n
    """
    # Validate supercell matrix dimensions
    if len(supercell_matrix) not in [3, 9]:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # Convert supercell matrix to 3x3 format
    if len(supercell_matrix) == 3:
        # Diagonal matrix: [na, nb, nc] -> [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        na, nb, nc = supercell_matrix
        supercell_array = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    else:
        # Full 3x3 matrix: reshape 9 numbers into 3x3
        supercell_array = np.array(supercell_matrix).reshape(3, 3)

    # DP calculator initialization
    print(f"Initializing DP calculator with potential: {potential}")
    try:
        from deepmd.calculator import DP

        calc = DP(model=potential)
    except ImportError:
        print("deepmd not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, outfile)


@app.command()
def hiphive(
    na: int,
    nb: int,
    nc: int,
    potential: str = typer.Option(
        ..., exists=True, help="Hiphive potential file path (e.g. 'potential.fcp')"
    ),
):
    """
    Calculate 4-phonon force constants using hiphive force constant potential.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
                    Note: The supercell size must be greater than or equal to the size used
                    for training the fcp potential. It cannot be smaller.
        potential: Hiphive potential file path
    """
    # Hiphive calculator initialization
    print(f"Using hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential

        fcp = ForceConstantPotential.read(potential)
        prim = fcp.primitive_structure
        supercell = prim.repeat((na, nb, nc))
        force_constants = fcp.get_force_constants(supercell)
        force_constants.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
    except ImportError:
        print("hiphive not found, please install it first")
        sys.exit(1)


@app.command()
def ploymp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(..., exists=True, help="PolyMLP potential file path"),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
):
    """
    Calculate 2-phonon force constants using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: PolyMLP potential file path\n
        outfile: Output file path for force constants\n
    """
    # Validate supercell matrix dimensions
    if len(supercell_matrix) not in [3, 9]:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # Convert supercell matrix to 3x3 format
    if len(supercell_matrix) == 3:
        # Diagonal matrix: [na, nb, nc] -> [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        na, nb, nc = supercell_matrix
        supercell_array = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    else:
        # Full 3x3 matrix: reshape 9 numbers into 3x3
        supercell_array = np.array(supercell_matrix).reshape(3, 3)

    # PolyMLP calculator initialization
    print(f"Using ploymp calculator with potential: {potential}")
    try:
        from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

        calc = PolymlpASECalculator(pot=potential)
    except ImportError:
        print("pypolymlp not found, please install it first")
        sys.exit(1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, outfile)
