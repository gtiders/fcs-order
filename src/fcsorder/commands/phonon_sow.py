#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party imports
import numpy as np
import typer
from ase.io import read, write

# Local imports
from ..core import (
    generate_phonon_rattled_structures,
    parse_FORCE_CONSTANTS,
    plot_distributions,
)


def phononrattle(
    sposcar: str = typer.Argument(..., exists=True, help="Path to SPOSCAR file"),
    fc2: str = typer.Argument(
        ..., exists=True, help="Path to second-order force constants file"
    ),
    number: int = typer.Option(
        100,
        "--number",
        "-n",
        help="Number of rattled structures to generate per temperature",
    ),
    temperatures: str = typer.Option(
        "300",
        "--temperatures",
        "-t",
        help='Temperature in K, such as "300,400,500"',
    ),
    min_distance: float = typer.Option(
        1.5,
        "--min-distance",
        help="Minimum distance between atoms in A",
    ),
    if_qm: bool = typer.Option(
        True,
        "--if-qm",
        help="Whether to consider quantum effects",
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        help="Imaginary frequency factor",
    ),
    output: str = typer.Option(
        "structures_phonon_rattle",
        "--output",
        "-o",
        help="Output filename prefix",
    ),
):
    """
    Generate phonon rattled structures with filtering based on displacement and distance criteria.

    For each temperature, generate structures until reaching the required number,
    filtering out structures with:
    - any displacement > max_disp
    - any interatomic distance < min_distance

    Args:
        sposcar: Path to SPOSCAR file\n
        fc2: Path to second-order force constants file\n
        number: Number of rattled structures to generate per temperature\n
        temperatures: Temperature in K, such as "300,400,500"\n
        min_distance: Minimum distance between atoms in A\n
        if_qm: Whether to consider quantum effects\n
        imag_freq_factor: Imaginary frequency factor\n
        output: Output filename prefix\n
    """
    sposcar = read(sposcar)
    ref_pos = sposcar.positions.copy()
    natoms = len(sposcar)
    fc2 = parse_FORCE_CONSTANTS(fc2, natoms)
    temperatures = [float(t) for t in temperatures.split(",")]

    for t in temperatures:
        typer.print(f"Processing temperature: {t} K")
        valid_structures = []
        attempts = 0
        max_attempts = number * 50  # Prevent infinite loop, set maximum attempts
        while len(valid_structures) < number and attempts < max_attempts:
            # Generate structures in batches for efficiency
            batch_size = min(number * 2, number * 10)  # Batch size
            batch_structures = generate_phonon_rattled_structures(
                sposcar,
                fc2,
                batch_size,
                t,
                QM_statistics=if_qm,
                imag_freq_factor=imag_freq_factor,
            )

            for atoms in batch_structures:
                # Check distance
                distances = atoms.get_all_distances(mic=True)
                # Exclude self-distance (diagonal is 0)
                mask = ~np.eye(len(atoms), dtype=bool)
                min_interatomic_dist = np.min(distances[mask])
                if min_interatomic_dist < min_distance:
                    continue

                # Passed filtering, add to valid structures list
                valid_structures.append(atoms)

                # Exit early if required number reached
                if len(valid_structures) >= number:
                    break

            attempts += batch_size
            typer.print(
                f"  Generated {attempts} structures, found {len(valid_structures)} valid structures"
            )

        # Save results
        if len(valid_structures) > 0:
            output_filename = f"{output}_T{int(t)}.xyz"

            # Use uniform random selection to ensure statistical distribution
            if len(valid_structures) > number:
                # Randomly select specified number from valid structures to maintain distribution
                selected_indices = np.random.choice(
                    len(valid_structures), size=number, replace=False
                )
                selected_structures = [valid_structures[i] for i in selected_indices]
            else:
                selected_structures = valid_structures

            write(output_filename, selected_structures, format="extxyz")
            plot_distributions(selected_structures, ref_pos, T=t)
            typer.print(f"  Saved {len(selected_structures)} structures to {output_filename}")

        if len(valid_structures) < number:
            typer.print(
                f"  Warning: Only found {len(valid_structures)} valid structures out of {number} requested"
            )
