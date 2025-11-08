#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys
import os

# Third-party imports
import typer
from ase.io import read

# Local imports
from ..core.scph_core import run_scph, analyze_scph_convergence
from ..core.secondorder_core import  build_supercell_from_matrix


# Create the main app
app = typer.Typer(
    help="Run self-consistent phonon calculations using machine learning potentials."
)


@app.command()
def nep(
    primcell: str = typer.Argument(..., help="Path to the primitive cell file (e.g., 'POSCAR')"),
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
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
        primcell: Path to the primitive cell file (e.g., 'POSCAR')\n
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
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
    T = [float(t) for t in temperatures.split(",")]

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

    # Read primitive cell and build supercell from matrix
    primcell = read(primcell)
    supercell = build_supercell_from_matrix(primcell, supercell_matrix)

    # Run SCPH calculation
    run_scph(
        primcell=primcell,
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
    primcell: str = typer.Argument(..., help="Path to the primitive cell file (e.g., 'POSCAR')"),
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
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
    T = [float(t) for t in temperatures.split(",")]

    # DP calculator initialization
    print(f"Initializing DP calculator with potential: {potential}")
    try:
        from deepmd.calculator import DP

        calc = DP(model=potential)
    except ImportError:
        print("deepmd not found, please install it first")
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    primcell = read(primcell)
    supercell = build_supercell_from_matrix(primcell, supercell_matrix)

    # Run SCPH calculation
    run_scph(
        primcell=primcell,
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
    primcell: str = typer.Argument(..., help="Path to the primitive cell file (e.g., 'POSCAR')"),
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
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
        primcell: Path to the primitive cell file (e.g., 'POSCAR')\n
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
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
    T = [float(t) for t in temperatures.split(",")]

    # Read primitive cell and build supercell from matrix
    primcell = read(primcell)
    supercell = build_supercell_from_matrix(primcell, supercell_matrix)

    # Hiphive calculator initialization
    print(f"Initializing Hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential
        from hiphive.calculators import ForceConstantCalculator

        fcp = ForceConstantPotential.read(potential)
        # Create a calculator from the force constant potential
        force_constants = fcp.get_force_constants(supercell)
        calc = ForceConstantCalculator(force_constants)
    except ImportError:
        print("hiphive not found, please install it first")
        sys.exit(1)

    # Run SCPH calculation
    run_scph(
        primcell=primcell,
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
    primcell: str = typer.Argument(..., help="Path to the primitive cell file (e.g., 'POSCAR')"),
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
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
        primcell: Path to the primitive cell file (e.g., 'POSCAR')\n
        supercell_matrix: Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)\n
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
    print(f"Using PolyMLP calculator with potential: {potential}")
    try:
        from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

        calc = PolymlpASECalculator(pot=potential)
    except ImportError:
        print("pypolymlp not found, please install it first")
        sys.exit(1)

    # Read primitive cell and build supercell from matrix
    primcell = read(primcell)
    supercell = build_supercell_from_matrix(primcell, supercell_matrix)

    # Create output directories
    os.makedirs("fcps/", exist_ok=True)

    # Run SCPH calculation
    run_scph(
        primcell=primcell,
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
    primcell: str = typer.Argument(..., help="Path to the primitive cell file (e.g., 'POSCAR')"),
    supercell_matrix: list[int] = typer.Argument(
        ..., help="Supercell expansion matrix, either 3 numbers (diagonal) or 9 numbers (3x3 matrix)"
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
    primcell_atoms = read(primcell)
    supercell = build_supercell_from_matrix(primcell_atoms, supercell_matrix)
    
    # Get unique elements from primitive cell
    unique_elements = sorted(set(primcell_atoms.get_chemical_symbols()))

    # MTP calculator initialization
    print(f"Initializing MTP calculator with potential: {potential}")
    try:
        from ..utils import MTP

        calc = MTP(
            mtp_path=potential,
            mtp_exe=mtp_exe,
            unique_elements=unique_elements
        )
        print(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        print(f"Error importing MTP: {e}")
        sys.exit(1)

    # Run SCPH calculation
    run_scph(
        primcell=primcell_atoms,
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
