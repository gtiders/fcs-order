#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports

# Third-party imports
import numpy as np
import typer
from ase.calculators.calculator import Calculator

# Local imports
from fcsorder.phonon.domain.secondorder_core import get_force_constants
from fcsorder.calc.calculators import (
    make_dp,
    make_mtp,
    make_nep,
    make_polymp,
    make_tace,
)
from fcsorder.io.io_abstraction import parse_supercell_matrix, read_atoms


def calculate_phonon_force_constants_2nd(
    supercell_array: np.ndarray,
    calculation: Calculator,
    poscar_path: str = "POSCAR",
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

    atoms = read_atoms(poscar_path, in_format="auto")

    try:
        from phonopy import Phonopy
        from phonopy.file_IO import write_FORCE_CONSTANTS
    except Exception as e:
        typer.echo(f"Error importing Phonopy module from phonopy: {e}")
        raise typer.Exit(code=1)

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
        ...,
        "--potential",
        "-p",
        exists=True,
        help="NEP potential file path (e.g. 'nep.txt')",
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        "-o",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", "-g", help="Use GPU calculator for faster computation"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 2-phonon force constants using NEP (Neural Evolution Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: NEP potential file path\n
        outfile: Output file path for force constants\n
        is_gpu: Use GPU calculator for faster computation\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Parse/validate supercell matrix using shared utility (preserve error type/message)
    try:
        supercell_array = parse_supercell_matrix(supercell_matrix)
    except ValueError:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # NEP calculator initialization
    typer.echo(f"Initializing NEP calculator with potential: {potential}")
    try:
        calc = make_nep(potential, is_gpu=is_gpu)
        typer.echo(
            "Using GPU calculator for NEP" if is_gpu else "Using CPU calculator for NEP"
        )
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, poscar, outfile)


@app.command()
def tace(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        exists=True,
        help="Path to the TACE model checkpoint (.pt/.pth/.ckpt)",
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        "-o",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Compute device, e.g., 'cpu' or 'cuda'"
    ),
    dtype: str = typer.Option(
        "float64",
        "--dtype",
        "-t",
        help="Tensor dtype: 'float32' | 'float64' | None (string 'None' to disable)",
    ),
    level: int = typer.Option(0, "--level", "-l", help="Fidelity level for TACE model"),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 2-phonon force constants using TACE model.
    """
    # Parse/validate supercell matrix
    try:
        supercell_array = parse_supercell_matrix(supercell_matrix)
    except ValueError:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # Normalize dtype option
    dtype_opt = None if dtype.lower() == "none" else dtype

    # TACE calculator initialization
    typer.echo(f"Initializing TACE calculator with model: {model_path}")
    try:
        calc = make_tace(
            model_path=model_path,
            device=device,
            dtype=dtype_opt,
            level=level,
        )
    except ImportError as e:
        print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, poscar, outfile)


@app.command()
def dp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="DeepMD potential file path (e.g. 'model.pb')",
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        "-o",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 2-phonon force constants using Deep Potential (DP) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: Deep Potential model file path\n
        outfile: Output file path for force constants\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Parse/validate supercell matrix using shared utility (preserve error type/message)
    try:
        supercell_array = parse_supercell_matrix(supercell_matrix)
    except ValueError:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # DP calculator initialization
    typer.echo(f"Initializing DP calculator with potential: {potential}")
    try:
        calc = make_dp(potential)
    except ImportError as e:
        print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, poscar, outfile)


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
    typer.echo(f"Using hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential

        fcp = ForceConstantPotential.read(potential)
        prim = fcp.primitive_structure
        supercell = prim.repeat((na, nb, nc))
        force_constants = fcp.get_force_constants(supercell)
        force_constants.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
    except ImportError:
        typer.echo("hiphive not found, please install it first")
        raise typer.Exit(code=1)


@app.command()
def ploymp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(
        ..., "--potential", "-p", exists=True, help="PolyMLP potential file path"
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        "-o",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 2-phonon force constants using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: PolyMLP potential file path\n
        outfile: Output file path for force constants\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Parse/validate supercell matrix using shared utility (preserve error type/message)
    try:
        supercell_array = parse_supercell_matrix(supercell_matrix)
    except ValueError:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # PolyMLP calculator initialization
    typer.echo(f"Using ploymp calculator with potential: {potential}")
    try:
        calc = make_polymp(potential)
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, poscar, outfile)


@app.command()
def mtp2(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="MTP potential file path (e.g. 'pot.mtp')",
    ),
    outfile: str = typer.Option(
        "FORCE_CONSTANTS_2ND",
        "--outfile",
        "-o",
        help="Output file path, default is 'FORCE_CONSTANTS_2ND'",
    ),
    mtp_exe: str = typer.Option(
        "mlp", "--mtp-exe", "-x", help="Path to MLP executable, default is 'mlp'"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 2-phonon force constants using MTP (Moment Tensor Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        potential: MTP potential file path\n
        outfile: Output file path for force constants\n
        mtp_exe: Path to MLP executable\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Parse/validate supercell matrix using shared utility (preserve error type/message)
    try:
        supercell_array = parse_supercell_matrix(supercell_matrix)
    except ValueError:
        raise typer.BadParameter(
            "Supercell matrix must have either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
        )

    # Read atoms to get unique elements
    atoms = read_atoms(poscar, in_format="auto")
    unique_elements = list(dict.fromkeys(atoms.get_chemical_symbols()))

    # MTP calculator initialization
    typer.echo(f"Initializing MTP calculator with potential: {potential}")
    try:
        calc = make_mtp(potential, mtp_exe=mtp_exe, unique_elements=unique_elements)
        typer.echo(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_2nd(supercell_array, calc, poscar, outfile)
