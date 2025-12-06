#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI for calculating second-order force constants using Phonopy and ML potentials."""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import typer
from ase.calculators.calculator import Calculator

from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.io.reader import StructureData


def calculate_fc2(
    structure: StructureData,
    calc: Calculator,
    supercell_matrix: np.ndarray,
    displacement_distance: float = 0.015,
) -> Phonopy:
    """Calculate second-order force constants using Phonopy.

    Args:
        structure: Primitive cell structure (StructureData object).
        calc: Calculator for computing forces.
        supercell_matrix: Supercell expansion matrix (3x3 numpy array).
        displacement_distance: Atomic displacement distance in Angstroms. Defaults to 0.015.

    Returns:
        Phonopy object containing the force constants.
    """
    # Use StructureData's built-in conversion to PhonopyAtoms
    structure_ph = structure.to_phonopy_atoms()

    # Create Phonopy object with supercell_matrix and generate displacements
    phonon = Phonopy(structure_ph, supercell_matrix)
    phonon.generate_displacements(distance=displacement_distance)

    typer.echo(
        f"Generated {len(phonon.supercells_with_displacements)} displaced structures"
    )

    # Compute forces for each displaced structure
    forces = []
    for i, structure_ph_disp in enumerate(phonon.supercells_with_displacements):
        # Use StructureData.from_phonopy_atoms for conversion
        structure_data = StructureData.from_phonopy_atoms(structure_ph_disp)
        structure_ase = structure_data.to_atoms()
        structure_ase.calc = calc
        forces.append(structure_ase.get_forces().copy())
        if (i + 1) % 10 == 0 or i == len(phonon.supercells_with_displacements) - 1:
            typer.echo(
                f"Computed forces for structure {i + 1}/{len(phonon.supercells_with_displacements)}"
            )

    phonon.forces = forces
    phonon.produce_force_constants()

    return phonon


def run_fc2(
    structure: StructureData,
    calc: Calculator,
    supercell_matrix: np.ndarray,
    output_file: str = "FORCE_CONSTANTS",
    output_format: str = "text",
    displacement_distance: float = 0.015,
) -> None:
    """Run second-order force constants calculation and save results.

    Args:
        structure: Primitive cell structure (StructureData object).
        calc: Calculator for computing forces.
        supercell_matrix: Supercell expansion matrix (3x3 numpy array).
        output_file: Output file path. Defaults to "FORCE_CONSTANTS".
        output_format: Output format, 'text' or 'hdf5'. Defaults to "text".
        displacement_distance: Atomic displacement distance in Angstroms. Defaults to 0.015.

    Returns:
        None: Force constants are saved to the specified file.
    """
    # Calculate force constants
    phonon = calculate_fc2(
        structure=structure,
        calc=calc,
        supercell_matrix=supercell_matrix,
        displacement_distance=displacement_distance,
    )

    # Get force constants
    fc2 = phonon.force_constants

    # Write force constants to file
    if output_format == "text":
        write_FORCE_CONSTANTS(fc2, filename=output_file)
    elif output_format == "hdf5":
        # Ensure filename has .hdf5 extension
        if not output_file.endswith(".hdf5"):
            output_file = f"{output_file}.hdf5"
        write_force_constants_to_hdf5(fc2, filename=output_file)
    else:
        raise ValueError(
            f"Unsupported output format: {output_format}. Use 'text' or 'hdf5'."
        )

    typer.echo(f"Force constants written to {output_file}")


def fc2(
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
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        help="Path to potential/model file",
    ),
    output_file: str = typer.Option(
        "FORCE_CONSTANTS",
        "--output",
        "-o",
        help="Output file path for force constants",
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        "-f",
        help="Output format for force constants: text or hdf5",
    ),
    displacement_distance: float = typer.Option(
        0.015,
        "--displacement",
        "-D",
        help="Atomic displacement distance in Angstroms",
        min=0.001,
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
    Calculate second-order force constants using Phonopy with any registered calculator.

    Example:
        fc2 2 2 2 --calculator nep --potential model.nep
        fc2 2 2 2 --calculator dp --potential model.pb --output-format hdf5
        fc2 2 2 2 --calculator tace --potential model.pt --device cuda
    """
    # Read structure using StructureData
    typer.echo(f"Reading structure from {structure_file}")
    structure = StructureData.from_file(structure_file)

    # Build supercell matrix
    supercell_matrix = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    typer.echo(f"Supercell matrix: {na} x {nb} x {nc}")

    # Build supercell using StructureData's make_supercell method
    supercell_structure = structure.make_supercell(na, nb, nc)
    supercell_atoms = supercell_structure.to_atoms()
    supercell_atoms.write("fc2_SPOSCAR", format="vasp", direct=True)
    typer.echo("Supercell written to fc2_SPOSCAR")

    # Build calculator arguments
    calculator_kwargs = {
        "potential": potential,
        "device": device,
        "dtype": dtype,
        "supercell": supercell_atoms,
        "structure": structure.to_atoms(),
    }

    # Create calculator
    typer.echo(f"Creating {calculator_type} calculator...")
    try:
        calc = CalculatorFactory.create(calculator_type, **calculator_kwargs)
        typer.echo(f"✓ {calculator_type.upper()} calculator initialized")
    except (ValueError, ImportError) as e:
        typer.secho(f"✗ Error creating calculator: {e}", fg=typer.colors.RED)
        sys.exit(1)

    # Run force constants calculation
    typer.echo("Starting second-order force constants calculation...")
    try:
        run_fc2(
            structure=structure,
            calc=calc,
            supercell_matrix=supercell_matrix,
            output_file=output_file,
            output_format=output_format,
            displacement_distance=displacement_distance,
        )
        typer.secho(
            "✓ Second-order force constants calculation completed successfully",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(f"✗ Force constants calculation failed: {e}", fg=typer.colors.RED)
        sys.exit(1)
