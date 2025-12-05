#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI for training second-order force constants using ensemble methods.

This command trains a second-order force constant potential (FCP) using
hiPhive + trainstation. A single calculator provides forces, while
``trainstation.EnsembleOptimizer`` performs an ensemble fit of the FCP
parameters. The training is performed iteratively by increasing the number
of rattled structures until the change in parameters between iterations
falls below a user-defined RMS threshold or a maximum number of iterations
is reached.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.io.reader import StructureData


def train_fcp_with_ensemble(
    structure: Atoms,
    supercell: Atoms,
    calc: Calculator,
    cutoff: float,
    rattled_structures: list[Atoms],
    max_iterations: int,
    convergence_threshold: float,
    n_ensembles: int,
    output_prefix: str,
) -> None:
    """Train a second-order FCP using ensemble optimization.

    Workflow:

    1. Build a :class:`ClusterSpace` from the primitive structure and cutoff.
    2. Iteratively increase the number of rattled structures used for fitting.
    3. For each iteration:
       - Evaluate forces with the provided calculator.
       - Convert structures with forces to hiPhive structures via
         :func:`hiphive.utilities.prepare_structures`.
       - Add them to a :class:`StructureContainer`.
       - Train an ensemble of models via :class:`trainstation.EnsembleOptimizer`.
       - Compare the RMS change of parameters to the previous iteration.
    4. Stop when parameters converge or ``max_iterations`` is reached.
    5. Average parameters over all iterations and write the final FCP.
    """

    try:
        from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
        from hiphive.utilities import prepare_structures
    except ImportError as e:
        raise ImportError(
            "Failed to import hiphive. Please install hiphive first (pip install hiphive)."
        ) from e

    try:
        from trainstation import EnsembleOptimizer
    except ImportError as e:
        raise ImportError(
            "Failed to import trainstation. Please install trainstation first (pip install trainstation)."
        ) from e

    cutoff_list = [cutoff]
    cluster_space = ClusterSpace(structure, cutoff_list)

    os.makedirs("fcps/", exist_ok=True)
    os.makedirs("phonon_trajs/", exist_ok=True)

    num_atoms = len(supercell)
    num_training_structures = min(num_atoms, len(rattled_structures))

    typer.echo("Training FCP with ensemble optimizer...")
    typer.echo(f"Initial number of training structures: {num_training_structures}")

    all_iteration_parameters: list[np.ndarray] = []
    converged = False

    for iteration in range(max_iterations):
        typer.echo("")
        typer.echo(f"Iteration {iteration + 1}/{max_iterations}")
        typer.echo(f"Using {num_training_structures} training structures...")

        # Select subset of rattled structures
        training_structures = rattled_structures[:num_training_structures]

        # Attach calculator and ensure forces are evaluated at least once
        for rattled in training_structures:
            rattled.set_calculator(calc)
            _ = rattled.get_forces()

        # Convert to hiPhive structures with displacements and forces
        structures = prepare_structures(training_structures, supercell)
        structure_container = StructureContainer(cluster_space)
        for structure_with_forces in structures:
            structure_container.add_structure(structure_with_forces)

        typer.echo(
            f"  Fitting ensemble with {len(structures)} structures and "
            f"cutoff {cutoff:.3f} Å..."
        )

        fit_data = structure_container.get_fit_data()
        optimizer = EnsembleOptimizer(fit_data, n_ensembles=n_ensembles)
        optimizer.train()
        parameters = optimizer.parameters

        # Save parameters for this iteration
        all_iteration_parameters.append(parameters)
        np.savetxt(
            os.path.join("phonon_trajs", f"{output_prefix}_parameters_iter{iteration:03d}"),
            parameters,
        )

        # Convergence check based on RMS difference of parameters
        if iteration > 0:
            prev_parameters = all_iteration_parameters[iteration - 1]
            rms_diff = float(np.sqrt(np.mean((parameters - prev_parameters) ** 2)))
            typer.echo(f"  Parameter RMS change: {rms_diff:.6e}")

            if rms_diff < convergence_threshold:
                typer.echo("  ✓ Parameters converged")
                converged = True
                break
        else:
            typer.echo("  First iteration (no parameter convergence check)")

        # If not converged and not last iteration, increase number of structures
        if iteration < max_iterations - 1:
            max_available = len(rattled_structures)
            if num_training_structures < max_available:
                new_num = min(num_training_structures * 2, max_available)
                typer.echo(
                    "  Increasing number of training structures: "
                    f"{num_training_structures} -> {new_num}"
                )
                num_training_structures = new_num
            else:
                typer.echo("  All available rattled structures are already in use")
        else:
            typer.echo("  Reached maximum iterations")

    typer.echo("")
    typer.echo("Averaging parameters from all iterations...")
    averaged_parameters = np.mean(np.array(all_iteration_parameters), axis=0)

    final_fcp = ForceConstantPotential(cluster_space, averaged_parameters)
    fcp_path = os.path.join("fcps", f"{output_prefix}_fcp.fcp")
    final_fcp.write(fcp_path)
    np.savetxt(
        os.path.join("fcps", f"{output_prefix}_parameters_final"),
        averaged_parameters,
    )

    typer.echo("✓ Force constant training completed")
    typer.echo(f"Final FCP written to {fcp_path}")


app = typer.Typer(help="Train second-order force constants using ensemble methods.")


@app.command()
def phonon(
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
    structure_file: str = typer.Option(
        "POSCAR",
        "--structure",
        "-s",
        help="Path to structure file (POSCAR, CIF, XYZ, etc.)",
        exists=True,
    ),
    calculator_type: str = typer.Option(
        ...,
        "--calculator",
        "-c",
        help="Calculator type (e.g., nep, dp, tace, mace)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        help="Path to potential/model file",
    ),
    cutoff: float = typer.Option(
        ...,
        "--cutoff",
        "-k",
        help="Cutoff radius for cluster space",
    ),
    rattle_amplitude: float = typer.Option(
        0.015,
        "--rattle-amplitude",
        "-r",
        help="Rattle amplitude in Angstrom",
        min=0.0,
    ),
    num_rattled_structures: int = typer.Option(
        0,
        "--num-structures",
        "-n",
        help=(
            "Number of rattled structures to generate. "
            "If 0, use number of atoms in the supercell."
        ),
        min=0,
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-i",
        help="Maximum number of training iterations",
        min=1,
    ),
    convergence_threshold: float = typer.Option(
        1.0e-3,
        "--convergence-threshold",
        help="RMS convergence threshold for FCP parameters",
        min=0.0,
    ),
    n_ensembles: int = typer.Option(
        8,
        "--ensembles",
        "-e",
        help="Number of ensemble members in EnsembleOptimizer",
        min=1,
    ),
    output_prefix: str = typer.Option(
        "phonon",
        "--output-prefix",
        help="Prefix for output files (FCP and parameter trajectories)",
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
) -> None:
    """Train second-order force constants using an ensemble optimizer.

    Example::

        phonon 2 2 2 \
            --calculator nep \
            --potential model.nep \
            --cutoff 5.0 \
            --rattle-amplitude 0.015 \
            --max-iterations 10 \
            --ensembles 8
    """

    from hiphive.structure_generation import generate_rattled_structures

    typer.echo(f"Reading structure from {structure_file}")
    structure = StructureData.from_file(structure_file)
    structure_atoms = structure.to_atoms()

    typer.echo(f"Building supercell ({na}x{nb}x{nc})...")
    supercell_structure = structure.make_supercell(na, nb, nc)
    supercell_atoms = supercell_structure.to_atoms()
    supercell_atoms.write("phonon_SPOSCAR", format="vasp", direct=True)
    typer.echo("Supercell written to phonon_SPOSCAR")

    calculator_kwargs = {
        "potential": potential,
        "device": device,
        "dtype": dtype,
        "structure": structure_atoms,
        "supercell": supercell_atoms,
    }

    typer.echo(f"Creating {calculator_type} calculator...")
    try:
        calc = CalculatorFactory.create(calculator_type, **calculator_kwargs)
        typer.echo(f"✓ {calculator_type.upper()} calculator initialized")
    except (ValueError, ImportError) as exc:
        typer.secho(f"✗ Error creating calculator: {exc}", fg=typer.colors.RED)
        sys.exit(1)

    if num_rattled_structures <= 0:
        num_rattled_structures = len(supercell_atoms)

    typer.echo(
        f"Generating {num_rattled_structures} rattled structures "
        f"with amplitude {rattle_amplitude} Å..."
    )
    rattled_structures = generate_rattled_structures(
        supercell_atoms,
        num_structures=num_rattled_structures,
        rattle_std=rattle_amplitude,
    )
    typer.echo(f"✓ Generated {len(rattled_structures)} rattled structures")

    try:
        train_fcp_with_ensemble(
            structure=structure_atoms,
            supercell=supercell_atoms,
            calc=calc,
            cutoff=cutoff,
            rattled_structures=rattled_structures,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            n_ensembles=n_ensembles,
            output_prefix=output_prefix,
        )
        typer.secho("✓ Training completed successfully", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"✗ Training failed: {exc}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    app()
