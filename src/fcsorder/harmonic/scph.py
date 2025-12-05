#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI for self-consistent phonon (SCPH) calculations using various ML potentials."""

from __future__ import annotations

import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.io.reader import StructureData


def parse_temperatures(temperature_str: str) -> list[float]:
    """Parse comma-separated temperatures string into a list of floats.

    Args:
        temperature_str: Comma-separated temperature values as string.

    Returns:
        List of temperature values as floats.
    """
    return [float(temp) for temp in temperature_str.split(",")]


def analyze_scph_convergence(temperature: float) -> None:
    """Analyze the convergence of SCPH parameters.
    
    Args:
        temperature: Temperature value in Kelvin.
    
    Returns:
        None: Plots are saved to 'scph_trajs/scph_parameter_T{temperature}.png'.
    """
    # Read parameter trajectories
    parameter_trajectories = np.loadtxt(f"scph_trajs/scph_parameters_T{temperature}")

    # Calculate parameter differences between iterations
    parameter_differences = [
        np.linalg.norm(p - p_next) 
        for p, p_next in zip(parameter_trajectories, parameter_trajectories[1:])
    ]

    # Setup plot
    figure = plt.figure(figsize=(8, 3.5))
    axis_params = figure.add_subplot(121)
    axis_diffs = figure.add_subplot(122)

    # Plot parameters
    axis_params.plot(parameter_trajectories)

    # Plot parameter differences
    axis_diffs.plot(parameter_differences)

    # Set labels and limits for parameter plot
    axis_params.set_xlabel(f"SCPH iteration {temperature} K")
    axis_params.set_ylabel("Parameters")
    axis_params.set_xlim([0, len(parameter_trajectories)])

    # Set labels and limits for parameter difference plot
    axis_diffs.set_xlabel(f"SCPH iteration {temperature} K")
    axis_diffs.set_ylabel("$\\Delta$ Parameters")
    axis_diffs.set_xlim([0, len(parameter_differences)])
    axis_diffs.set_ylim(bottom=0.0)

    figure.tight_layout()
    figure.savefig(f"scph_trajs/scph_parameter_T{temperature}.png")
    typer.echo(
        f"SCPH parameter convergence plot for T={temperature} K saved to 'scph_trajs/scph_parameter_T{temperature}.png'"
    )


def run_scph(
    structure: Atoms,
    calc: Calculator,
    supercell: Atoms,
    temperatures: list[float],
    cutoff: float,
    alpha: float = 0.2,
    num_iterations: int = 100,
    num_structures: int = 50,
    use_qm_statistics: bool = True,
    imaginary_frequency_factor: float = 1.0,
) -> None:
    """Run the self-consistent phonon calculation.

    Args:
        structure: The primitive cell structure (ASE Atoms object).
        calc: The calculator for computing forces.
        supercell: The supercell structure (ASE Atoms object).
        temperatures: List of temperatures for the calculation.
        cutoff: Cutoff radius for the cluster space.
        alpha: The mixing parameter for SCPH iterations. Defaults to 0.2.
        num_iterations: The number of iterations for SCPH. Defaults to 100.
        num_structures: The number of structures to generate. Defaults to 50.
        use_qm_statistics: Whether to use quantum-mechanical statistics. Defaults to True.
        imaginary_frequency_factor: Factor for treating imaginary frequencies. Defaults to 1.0.

    Returns:
        None: Results are saved to files in the 'fcps/' and 'scph_trajs/' directories.
    """
    # Lazy import hiphive here to avoid hard dependency at import-time
    try:
        from hiphive import ClusterSpace, ForceConstantPotential
        from hiphive.self_consistent_phonons import self_consistent_harmonic_model
    except ImportError as e:
        raise ImportError(
            "Failed to import hiphive. Please install hiphive package first. "
            "You can install it using: pip install hiphive"
        ) from e

    # Setup parameters
    cutoff_list = [cutoff]
    cluster_space = ClusterSpace(structure, cutoff_list)

    # Run SCPH
    os.makedirs("scph_trajs/", exist_ok=True)
    os.makedirs("fcps/", exist_ok=True)
    for temperature in temperatures:
        typer.echo(f"Running SCPH at {temperature} K...")
        parameter_trajectory = self_consistent_harmonic_model(
            atoms_ideal=supercell,
            calc=calc,
            cs=cluster_space,
            T=temperature,
            alpha=alpha,
            n_iterations=num_iterations,
            n_structures=num_structures,
            QM_statistics=use_qm_statistics,
            imag_freq_factor=imaginary_frequency_factor,
        )
        force_constant_potential = ForceConstantPotential(cluster_space, parameter_trajectory[-1])
        force_constant_potential.get_force_constants(supercell).write_to_phonopy(
            f"fcps/{temperature}_FORCE_CONSTANTS", format="text"
        )

        force_constant_potential.write(f"fcps/scph_T{temperature}.fcp")
        np.savetxt(f"scph_trajs/scph_parameters_T{temperature}", np.array(parameter_trajectory))
        analyze_scph_convergence(temperature)


# Create the main app
app = typer.Typer(
    help="Run self-consistent phonon (SCPH) calculations using ML potentials."
)


@app.command()
def scph(
    na: int = typer.Argument(
        ...,
        help="Supercell repetition along first lattice vector",
        min=1,
    ),
    nb: int = typer.Argument(
        ...,
        help="Supercell repetition along second lattice vector",
        min=1,
    ),
    nc: int = typer.Argument(
        ...,
        help="Supercell repetition along third lattice vector",
        min=1,
    ),
    calculator_type: str = typer.Option(
        ...,
        "--calculator",
        "-c",
        help=f"Calculator type: {', '.join(CalculatorFactory.list_available())}",
    ),
    structure_file: str = typer.Option(
        "POSCAR",
        "--structure",
        "-s",
        help="Path to structure file (POSCAR, CIF, XYZ, etc.)",
        exists=True,
    ),
    temperatures: str = typer.Option(
        ...,
        "--temperatures",
        "-T",
        help="Temperatures in Kelvin, e.g., '100,200,300'",
    ),
    cutoff: float = typer.Option(
        ...,
        "--cutoff",
        "-k",
        help="Cutoff radius for cluster space",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        help="Path to potential/model file",
    ),
    alpha: float = typer.Option(
        0.2,
        "--alpha",
        "-a",
        help="Mixing parameter for SCPH iterations",
    ),
    num_iterations: int = typer.Option(
        100,
        "--num-iterations",
        "-i",
        help="Number of SCPH iterations",
        min=1,
    ),
    num_structures: int = typer.Option(
        50,
        "--num-structures",
        "-n",
        help="Number of structures to generate",
        min=1,
    ),
    use_qm_statistics: bool = typer.Option(
        True,
        "--use-qm-statistics/--no-qm-statistics",
        help="Use quantum-mechanical statistics",
    ),
    imaginary_frequency_factor: float = typer.Option(
        1.0,
        "--imaginary-frequency-factor",
        help="Factor for treating imaginary frequencies",
        min=0.0,
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Compute device: cpu or cuda",
    ),
    dtype: Optional[str] = typer.Option(
        None,
        "--dtype",
        "-t",
        help="Data type: float32 or float64",
    ),
):
    """
    Run self-consistent phonon (SCPH) calculation with any registered calculator.
    
    Example:
        scph 2 2 2 --calculator nep --potential model.nep --temperatures 100,200,300 --cutoff 5.0
        scph 2 2 2 --calculator dp --potential model.pb --temperatures 100,200 --cutoff 5.0
        scph 2 2 2 --calculator tace --potential model.pt --temperatures 100 --cutoff 5.0 --device cuda
    """
    # Parse temperatures
    temperature_list = parse_temperatures(temperatures)
    
    # Read structure and build supercell
    typer.echo(f"Reading structure from {structure_file}")
    structure = StructureData.from_file(structure_file)
    structure_atoms = structure.to_atoms()
    
    # Build supercell using na, nb, nc
    supercell_structure = structure.make_supercell(na, nb, nc)
    supercell_atoms = supercell_structure.to_atoms()
    supercell_atoms.write("scph_SPOSCAR", format="vasp", direct=True)
    typer.echo("Supercell written to scph_SPOSCAR")
    
    # Build calculator arguments - pass all parameters, calculators use what they need
    calculator_kwargs = {
        "potential": potential,
        "device": device,
        "dtype": dtype,
        "supercell": supercell_atoms, 
        "structure": structure_atoms,
    }
    
    # Create calculator
    typer.echo(f"Creating {calculator_type} calculator...")
    try:
        calc = CalculatorFactory.create(calculator_type, **calculator_kwargs)
        typer.echo(f"✓ {calculator_type.upper()} calculator initialized")
    except (ValueError, ImportError) as e:
        typer.secho(f"✗ Error creating calculator: {e}", fg=typer.colors.RED)
        sys.exit(1)
    
    # Run SCPH
    typer.echo(f"Starting SCPH calculation with {len(temperature_list)} temperature(s)...")
    try:
        run_scph(
            structure=structure_atoms,
            calc=calc,
            supercell=supercell_atoms,
            temperatures=temperature_list,
            cutoff=cutoff,
            alpha=alpha,
            num_iterations=num_iterations,
            num_structures=num_structures,
            use_qm_statistics=use_qm_statistics,
            imaginary_frequency_factor=imaginary_frequency_factor,
        )
        typer.secho("✓ SCPH calculation completed successfully", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"✗ SCPH calculation failed: {e}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    app()
