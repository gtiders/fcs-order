#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import os
import sys
from typing import List

# Third-party imports
import typer

# Local imports

from fcsorder.calc.calculators import (
    make_dp,
    make_mtp,
    make_nep,
    make_polymp,
    make_tace,
)
from fcsorder.io.io_abstraction import read_atoms

from fcsorder.phonon.domain.self_consistent_phonons import (
    analyze_scph_convergence,
    run_scph,
)
from fcsorder.phonon.domain.secondorder_core import build_supercell_from_matrix


def parse_temperatures(s: str) -> List[float]:
    """
    Parse a comma-separated temperatures string into a list of floats.

    This mirrors the existing behavior used in commands:
    - Splits on ',' and converts each token using float().
    - Will raise ValueError if any token is not a valid float (same as before).
    """
    return [float(t) for t in s.split(",")]


# Create the main app
app = typer.Typer(
    help="Run self-consistent phonon calculations using machine learning potentials."
)


@app.command()
def nep(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="NEP potential file path (e.g., 'nep.txt')",
    ),
    alpha: float = typer.Option(
        0.2, "--alpha", "-a", help="Mixing parameter for SCPH iterations"
    ),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", "-g", help="Use GPU calculator for faster computation"
    ),
):
    """
    Run self-consistent phonon calculation using NEP (Neural Evolution Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
        temperatures: Temperatures for calculation, e.g., '100,200,300'\n
        cutoff: Cutoff radius for the cluster space\n
        potential: NEP potential file path\n
        alpha: Mixing parameter for SCPH iterations\n
        n_iterations: Number of iterations for SCPH\n
        n_structures: Number of structures to generate\n
        fcs_2nd: Path to the FCS2 file for initial parameters\n
        is_qm: Whether to use quantum statistics\n
        imag_freq_factor: Factor for handling imaginary frequencies\n
        is_gpu: Use GPU calculator for faster computation\n
    """
    # Parse temperatures string to list of floats
    T = parse_temperatures(temperatures)

    # NEP calculator initialization
    typer.echo(f"Initializing NEP calculator with potential: {potential}")
    try:
        calc = make_nep(potential, is_gpu=is_gpu)
        typer.echo(
            "Using GPU calculator for NEP" if is_gpu else "Using CPU calculator for NEP"
        )
    except ImportError as e:
        typer.echo(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read_atoms(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Run SCPH calculation
    run_scph(
        primcell=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )


@app.command()
def tace(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        exists=True,
        help="Path to the TACE model checkpoint (.pt/.pth/.ckpt)",
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Compute device, e.g., 'cpu' or 'cuda'"
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        "-t",
        help="Tensor dtype: 'float32' | 'float64' | None (string 'None' to disable)",
    ),
    level: int = typer.Option(0, "--level", "-l", help="Fidelity level for TACE model"),
    alpha: float = typer.Option(
        0.2, "--alpha", "-a", help="Mixing parameter for SCPH iterations"
    ),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
):
    """
    Run self-consistent phonon calculation using TACE model.
    """
    # Parse temperatures
    T = parse_temperatures(temperatures)

    # Initialize TACE calculator
    typer.echo(f"Initializing TACE calculator with model: {model_path}")
    try:
        dtype_opt = None if dtype.lower() == "none" else dtype
        calc = make_tace(
            model_path=model_path,
            device=device,
            dtype=dtype_opt,
            level=level,
        )
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    # Read primitive cell and build supercell
    poscar = read_atoms(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Run SCPH calculation
    run_scph(
        primcell=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )


@app.command()
def dp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="DeepMD model file path (e.g., 'graph.pb')",
    ),
    alpha: float = typer.Option(
        0.2, "--alpha", "-a", help="Mixing parameter for SCPH iterations"
    ),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
):
    """
    Run self-consistent phonon calculation using Deep Potential (DP) model.

    Args:
        primcell: Path to the primitive cell file (e.g., 'POSCAR')\n
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        temperatures: Temperatures for calculation, e.g., '100,200,300'\n
        cutoff: Cutoff radius for the cluster space\n
        potential: DeepMD potential file path\n
        alpha: Mixing parameter for SCPH iterations\n
        n_iterations: Number of iterations for SCPH\n
        n_structures: Number of structures to generate\n
        fcs_2nd: Path to the FCS2 file for initial parameters\n
        is_qm: Whether to use quantum statistics\n
        imag_freq_factor: Factor for handling imaginary frequencies\n
    """
    # Parse temperatures string to list of floats
    T = parse_temperatures(temperatures)

    # DP calculator initialization
    typer.echo(f"Initializing DP calculator with potential: {potential}")
    try:
        calc = make_dp(potential)
    except ImportError as e:
        typer.echo(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read_atoms(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Run SCPH calculation
    run_scph(
        primcell=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )


@app.command()
def hiphive(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="Hiphive model file path (e.g., 'model.fcp')",
    ),
    alpha: float = typer.Option(
        0.2, "--alpha", "-a", help="Mixing parameter for SCPH iterations"
    ),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
):
    """
    Run self-consistent phonon calculation using Hiphive potential model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        poscar: Path to the primitive cell file (e.g., 'POSCAR')\n
        temperatures: Temperatures for calculation, e.g., '100,200,300'\n
        cutoff: Cutoff radius for the cluster space\n
        potential: Hiphive potential file path\n
        alpha: Mixing parameter for SCPH iterations\n
        n_iterations: Number of iterations for SCPH\n
        n_structures: Number of structures to generate\n
        fcs_2nd: Path to the FCS2 file for initial parameters\n
        is_qm: Whether to use quantum statistics\n
        imag_freq_factor: Factor for handling imaginary frequencies\n
    """
    # Parse temperatures string to list of floats
    T = parse_temperatures(temperatures)

    # Read primitive cell and build supercell from matrix
    poscar = read_atoms(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Hiphive calculator initialization
    typer.echo(f"Initializing Hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential
        from hiphive.calculators import ForceConstantCalculator

        fcp = ForceConstantPotential.read(potential)
        # Create a calculator from the force constant potential
        force_constants = fcp.get_force_constants(supercell)
        calc = ForceConstantCalculator(force_constants)
    except ImportError:
        typer.echo("hiphive not found, please install it first")
        sys.exit(1)

    # Run SCPH calculation
    run_scph(
        primcell=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )


@app.command()
def ploymp(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="Ploymp potential file path (e.g., 'model.mp')",
    ),
    alpha: float = typer.Option(
        0.2, "--alpha", "-a", help="Mixing parameter for SCPH iterations"
    ),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
):
    """
    Run self-consistent phonon calculation using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        poscar: Path to the primitive cell file (e.g., 'POSCAR')\n
        temperatures: Temperatures for calculation, e.g., '100,200,300'\n
        cutoff: Cutoff radius for the cluster space\n
        potential: PolyMLP potential file path\n
        alpha: Mixing parameter for SCPH iterations\n
        n_iterations: Number of iterations for SCPH\n
        n_structures: Number of structures to generate\n
        fcs_2nd: Path to the FCS2 file for initial parameters\n
        is_qm: Whether to use quantum statistics\n
        imag_freq_factor: Factor for handling imaginary frequencies\n
    """
    # Parse temperatures string to list of floats
    T = [float(t) for t in temperatures.split(",")]

    # PolyMLP calculator initialization
    typer.echo(f"Using PolyMLP calculator with potential: {potential}")
    try:
        calc = make_polymp(potential)
    except ImportError as e:
        typer.echo(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Create output directories
    os.makedirs("fcps/", exist_ok=True)

    # Run SCPH calculation
    run_scph(
        primcell=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )


@app.command()
def mtp2(
    supercell_matrix: list[int] = typer.Argument(
        ...,
        help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures for calculation, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ..., "--cutoff", "-c", help="Cutoff radius for the cluster space"
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="MTP potential file path (e.g., 'pot.mtp')",
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(
        100, "--n-iterations", "-i", help="Number of iterations for SCPH"
    ),
    n_structures: int = typer.Option(
        50, "--n-structures", "-n", help="Number of structures to generate"
    ),
    fcs_2nd: str = typer.Option(
        None, "--fcs-2nd", "-F", help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(
        True, "--is-qm", "-q", help="Whether to use quantum statistics"
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        "-I",
        help="Factor for handling imaginary frequencies",
    ),
    mtp_exe: str = typer.Option(
        "mlp", "--mtp-exe", "-x", help="Path to MLP executable, default is 'mlp'"
    ),
):
    """
    Run self-consistent phonon calculation using MTP (Moment Tensor Potential) model.

    Args:
        primcell: Path to the primitive cell file (e.g., 'POSCAR')\n
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
        temperatures: Temperatures for calculation, e.g., '100,200,300'\n
        cutoff: Cutoff radius for the cluster space\n
        potential: MTP potential file path\n
        alpha: Mixing parameter for SCPH iterations\n
        n_iterations: Number of iterations for SCPH\n
        n_structures: Number of structures to generate\n
        fcs_2nd: Path to the FCS2 file for initial parameters\n
        is_qm: Whether to use quantum statistics\n
        imag_freq_factor: Factor for handling imaginary frequencies\n
        mtp_exe: Path to MLP executable\n
    """
    # Parse temperatures string to list of floats
    T = [float(t) for t in temperatures.split(",")]

    # Read primitive cell and build supercell from matrix
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)
    supercell.write("scph_SPOSCAR", format="vasp", direct=True)

    # Get unique elements from primitive cell
    unique_elements = sorted(set(poscar.get_chemical_symbols()))

    # MTP calculator initialization
    typer.echo(f"Initializing MTP calculator with potential: {potential}")
    try:
        calc = make_mtp(potential, mtp_exe=mtp_exe, unique_elements=unique_elements)
        typer.echo(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        typer.echo(str(e))
        sys.exit(1)

    # Run SCPH calculation
    run_scph(
        poscar=poscar,
        calc=calc,
        supercell=supercell,
        temperatures=T,
        cutoff=cutoff,
        alpha=alpha,
        n_iterations=n_iterations,
        n_structures=n_structures,
        fcs_2nd=fcs_2nd,
        is_qm=is_qm,
        imag_freq_factor=imag_freq_factor,
    )
