#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys
import os

# Third-party imports
import typer
from ase.io import read
from typing import List


# Local imports
from ..core.scph_core import run_scph, analyze_scph_convergence
from ..core.secondorder_core import  build_supercell_from_matrix
from ..utils.calculators import make_nep, make_dp, make_polymp, make_mtp, make_tace


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
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    potential: str = typer.Option(
        ..., exists=True, help="NEP potential file path (e.g., 'nep.txt')"
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", help="Use GPU calculator for faster computation"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
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
    typer.print(f"Initializing NEP calculator with potential: {potential}")
    try:
        calc = make_nep(potential, is_gpu=is_gpu)
        typer.print("Using GPU calculator for NEP" if is_gpu else "Using CPU calculator for NEP")
    except ImportError as e:
        typer.print(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)

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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)


@app.command()
def tace(
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    model_path: str = typer.Option(
        ..., exists=True, help="Path to the TACE model checkpoint (.pt/.pth/.ckpt)"
    ),
    device: str = typer.Option("cuda", help="Compute device, e.g., 'cpu' or 'cuda'"),
    dtype: str = typer.Option(
        "float32",
        help="Tensor dtype: 'float32' | 'float64' | None (string 'None' to disable)",
    ),
    level: int = typer.Option(0, help="Fidelity level for TACE model"),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
    ),
):
    """
    Run self-consistent phonon calculation using TACE model.
    """
    # Parse temperatures
    T = parse_temperatures(temperatures)

    # Initialize TACE calculator
    typer.print(f"Initializing TACE calculator with model: {model_path}")
    try:
        dtype_opt = None if dtype.lower() == "none" else dtype
        calc = make_tace(
            model_path=model_path,
            device=device,
            dtype=dtype_opt,
            level=level,
        )
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    # Read primitive cell and build supercell
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)

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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)


@app.command()
def dp(
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    potential: str = typer.Option(
        ..., exists=True, help="DeepMD model file path (e.g., 'graph.pb')"
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
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
    typer.print(f"Initializing DP calculator with potential: {potential}")
    try:
        calc = make_dp(potential)
    except ImportError as e:
        typer.print(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)

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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)


@app.command()
def hiphive(
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    potential: str = typer.Option(
        ..., exists=True, help="Hiphive model file path (e.g., 'model.fcp')"
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
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
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)

    # Hiphive calculator initialization
    typer.print(f"Initializing Hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential
        from hiphive.calculators import ForceConstantCalculator

        fcp = ForceConstantPotential.read(potential)
        # Create a calculator from the force constant potential
        force_constants = fcp.get_force_constants(supercell)
        calc = ForceConstantCalculator(force_constants)
    except ImportError:
        typer.print("hiphive not found, please install it first")
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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)


@app.command()
def ploymp(
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    potential: str = typer.Option(
        ..., exists=True, help="Ploymp potential file path (e.g., 'model.mp')"
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
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
    typer.print(f"Using PolyMLP calculator with potential: {potential}")
    try:
        calc = make_polymp(potential)
    except ImportError as e:
        typer.print(str(e))
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    poscar = read(poscar)
    supercell = build_supercell_from_matrix(poscar, supercell_matrix)

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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)


@app.command()
def mtp(
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
    ),
    poscar: str = typer.Option(
        "POSCAR", "--poscar", help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'", exists=True
    ),
    temperatures: str = typer.Option(
        ..., help="Temperatures for calculation, e.g., '100,200,300'"
    ),
    cutoff: float = typer.Option(..., help="Cutoff radius for the cluster space"),
    potential: str = typer.Option(
        ..., exists=True, help="MTP potential file path (e.g., 'pot.mtp')"
    ),
    alpha: float = typer.Option(0.2, help="Mixing parameter for SCPH iterations"),
    n_iterations: int = typer.Option(100, help="Number of iterations for SCPH"),
    n_structures: int = typer.Option(50, help="Number of structures to generate"),
    fcs_2nd: str = typer.Option(
        None, help="Path to the FCS2 file for initial parameters"
    ),
    is_qm: bool = typer.Option(True, help="Whether to use quantum statistics"),
    imag_freq_factor: float = typer.Option(
        1.0, help="Factor for handling imaginary frequencies"
    ),
    mtp_exe: str = typer.Option(
        "mlp", "--mtp-exe", help="Path to MLP executable, default is 'mlp'"
    ),
    analyze_convergence: bool = typer.Option(
        True,
        "--analyze-convergence",
        help="Analyze SCPH parameter convergence after calculation",
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
    
    # Get unique elements from primitive cell
    unique_elements = sorted(set(poscar.get_chemical_symbols()))

    # MTP calculator initialization
    typer.print(f"Initializing MTP calculator with potential: {potential}")
    try:
        calc = make_mtp(potential, mtp_exe=mtp_exe, unique_elements=unique_elements)
        typer.print(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        typer.print(str(e))
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

    # Analyze convergence if requested
    if analyze_convergence:
        analyze_scph_convergence(T)
