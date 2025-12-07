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

from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer

try:
    from trainstation import Optimizer
except ImportError:
    # Fallback or assume it will be available at runtime
    pass
from hiphive import utilities
from hiphive.structure_generation import (
    generate_phonon_rattled_structures,
    generate_rattled_structures,
)
# from hiphive.self_consistent_phonons import self_consistent_harmonic_model # Replaced by local implementation

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.io.reader import StructureData


# =============================================================================
# Anderson Acceleration Helper Classes & Functions
# =============================================================================


class AndersonMixer:
    """Anderson/Pulay mixer for accelerating fixed-point iterations.

    Minimizes the norm of the residual vector in the subspace spanned by
    previous iterations.
    """

    def __init__(self, alpha: float, history_size: int = 5):
        self.alpha = alpha
        self.history_size = history_size
        self.x_history = []
        self.r_history = []

    def update(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Calculate the next guess for x.

        Args:
            x_in: The input vector used to generate x_out.
            x_out: The output vector from the current iteration.

        Returns:
            The mixed vector for the next iteration.
        """
        residual = x_out - x_in

        # Add to history
        self.x_history.append(x_in)
        self.r_history.append(residual)

        # Prune history if too long
        if len(self.x_history) > self.history_size:
            self.x_history.pop(0)
            self.r_history.pop(0)

        n = len(self.x_history)

        # If only one point, use simple linear mixing
        if n == 1:
            return x_in + self.alpha * residual

        # Solve for coefficients theta that minimize |sum theta_i R_i|^2
        # subject to sum theta_i = 1.
        # This is equivalent to solving the linear system:
        # [ R^T R   1 ] [ theta ] = [ 0 ]
        # [ 1^T     0 ] [ lambda]   [ 1 ]

        R = np.array(self.r_history).T  # Shape (dim, n)
        GTG = R.T @ R  # Shape (n, n)

        # Build system matrix
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = GTG
        A[:n, n] = 1.0
        A[n, :n] = 1.0

        # Build RHS
        b = np.zeros(n + 1)
        b[n] = 1.0

        try:
            # Solve linear system
            solution = np.linalg.solve(A, b)
            theta = solution[:n]

            # Compute new x
            # x_new = sum(theta_i * (x_i + alpha * R_i))
            #       = sum(theta_i * x_i) + alpha * sum(theta_i * R_i)
            # Or distinct beta can be used. We use self.alpha as beta.

            x_mixed = np.zeros_like(x_in)
            r_mixed = np.zeros_like(x_in)

            for i in range(n):
                x_mixed += theta[i] * self.x_history[i]
                r_mixed += theta[i] * self.r_history[i]

            x_next = x_mixed + self.alpha * r_mixed
            return x_next

        except np.linalg.LinAlgError:
            # Fallback to linear mixing if system is singular
            return x_in + self.alpha * residual


def self_consistent_harmonic_model_anderson(
    atoms_ideal: Atoms,
    calc: Calculator,
    cs: ClusterSpace,
    T: float,
    alpha: float = 0.2,
    n_iterations: int = 100,
    n_structures: int = 50,
    QM_statistics: bool = True,
    imag_freq_factor: float = 1.0,
    history_size: int = 5,
) -> list[np.ndarray]:
    """Self-consistent phonon calculation with Anderson Acceleration.

    Re-implementation of the Hiphive SCPH loop to include Pulay mixing.
    """
    n_params = cs.n_dofs

    # Initialize parameters (zeros if starting from scratch)
    # Ideally we should start from a guess or run a small fitting first?
    # Hiphive usually handles cold start. Let's assume start with zeros.
    parameters = np.zeros(n_params)
    parameter_trajectory = [parameters.copy()]

    mixer = AndersonMixer(alpha=alpha, history_size=history_size)

    # Initial print
    typer.echo(f"  Starting Anderson SCPH loop (history={history_size})")

    for i in range(n_iterations):
        # 1. Generate finite temperature displacements
        # We need to construct a temporary ForceConstantPotential
        fcp = ForceConstantPotential(cs, parameters)

        # Get force constants for the supercell
        fc = fcp.get_force_constants(atoms_ideal)

        if i == 0 and np.allclose(parameters, 0):
            # Initial iteration with zero parameters: use logic similar to hiphive's cold start
            # Standard rattle with 0.015 A std dev (typical guess)
            typer.echo("    Generating initial random rattled structures (std=0.015 A)")
            atoms_list_raw = generate_rattled_structures(
                atoms_ideal, n_structures, 0.015
            )
        else:
            # Phonon rattled structures
            # Need fc2 array
            fc2 = fc.get_fc_array(order=2, format="ase")
            try:
                atoms_list_raw = generate_phonon_rattled_structures(
                    atoms_ideal,
                    fc2,
                    n_structures,
                    T,
                    QM_statistics=QM_statistics,
                    imag_freq_factor=imag_freq_factor,
                )
            except Exception as e:
                typer.echo(f"  Error generating phonon structures: {e}")
                break

        # Prepare structures (calc forces via prepare_structures which calls calc)
        # Note: Original code uses prepare_structures(structures, atoms_ideal, calc, check_permutation=False)
        # prepare_structures in utilities calculates forces if calc is passed!
        try:
            atoms_list = utilities.prepare_structures(
                atoms_list_raw, atoms_ideal, calc, check_permutation=False
            )
        except Exception as e:
            typer.echo(f"  Error preparing structures: {e}")
            break

        # 2. Fit new parameters
        sc = StructureContainer(cs)
        for atoms in atoms_list:
            sc.add_structure(atoms)

        # Standard least squares fit
        try:
            opt = Optimizer(sc.get_fit_data(), fit_method="least-squares")
            opt.train()
            x_out = opt.parameters
        except NameError:
            # Fallback if Optimizer not imported
            # Try direct fit from StructureContainer if available? No.
            # Assume import failure and raise
            raise ImportError("Could not import hiphive.fitting.Optimizer")

        # 4. Mix parameters
        x_in = parameters
        x_next = mixer.update(x_in, x_out)

        parameters = x_next
        parameter_trajectory.append(parameters.copy())

        # Compute change
        diff = np.linalg.norm(x_in - parameters)
        typer.echo(
            f"  Iter {i + 1}/{n_iterations}: change = {diff:.6e} (residual norm: {np.linalg.norm(x_out - x_in):.6e})"
        )

        # Convergence check (simple)
        if diff < 1e-3:  # Adjustable tolerance
            typer.echo("  Converged!")
            break

    return parameter_trajectory


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
    axis_diffs.set_ylim(top=1.0)

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
    history_size: int = 5,
    output_format: str = "text",
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
        output_format: Output format for force constants, 'text' or 'hdf5'. Defaults to 'text'.

    Returns:
        None: Results are saved to files in the 'fcps/' and 'scph_trajs/' directories.
    """
    # Setup parameters
    cutoff_list = [cutoff]
    cluster_space = ClusterSpace(structure, cutoff_list)

    # Run SCPH
    os.makedirs("scph_trajs/", exist_ok=True)
    os.makedirs("fcps/", exist_ok=True)
    for temperature in temperatures:
        typer.echo(f"Running SCPH (Anderson Acceleration) at {temperature} K...")
        parameter_trajectory = self_consistent_harmonic_model_anderson(
            atoms_ideal=supercell,
            calc=calc,
            cs=cluster_space,
            T=temperature,
            alpha=alpha,
            n_iterations=num_iterations,
            n_structures=num_structures,
            QM_statistics=use_qm_statistics,
            imag_freq_factor=imaginary_frequency_factor,
            history_size=history_size,
        )
        force_constant_potential = ForceConstantPotential(
            cluster_space, parameter_trajectory[-1]
        )
        fc_filename = (
            f"fcps/{temperature}_FORCE_CONSTANTS"
            if output_format == "text"
            else f"fcps/{temperature}_fc.hdf5"
        )
        force_constant_potential.get_force_constants(supercell).write_to_phonopy(
            fc_filename, format=output_format
        )
        typer.echo(f"Force constants written to {fc_filename}")

        force_constant_potential.write(f"fcps/scph_T{temperature}.fcp")
        np.savetxt(
            f"scph_trajs/scph_parameters_T{temperature}", np.array(parameter_trajectory)
        )
        analyze_scph_convergence(temperature)


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
        30,
        "--num-iterations",
        "-i",
        help="Number of SCPH iterations",
        min=1,
    ),
    num_structures: int = typer.Option(
        500,
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
    history_size: int = typer.Option(
        5,
        "--history-size",
        help="History size for Anderson mixing",
        min=1,
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        "-f",
        help="Output format for force constants: text or hdf5",
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
    typer.echo(
        f"Starting SCPH calculation with {len(temperature_list)} temperature(s)..."
    )
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
            history_size=history_size,
            output_format=output_format,
        )
        typer.secho("✓ SCPH calculation completed successfully", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"✗ SCPH calculation failed: {e}", fg=typer.colors.RED)
        sys.exit(1)
