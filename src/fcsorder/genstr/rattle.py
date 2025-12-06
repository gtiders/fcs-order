#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI for generating rattled structures using simple random displacements."""

from __future__ import annotations

import numpy as np
import typer

from fcsorder.io.reader import StructureData
from fcsorder.genstr.tools import plot_distributions
from hiphive.structure_generation import generate_rattled_structures


def rattle_cli(
    structure_file: str = typer.Argument(
        ..., help="Path to structure file (POSCAR, CIF, etc.)"
    ),
    num_structures: int = typer.Option(
        10,
        "--num-structures",
        "-n",
        help="Number of structures to generate",
        min=1,
    ),
    rattle_amplitude: float = typer.Option(
        0.05,
        "--rattle-amplitude",
        "-a",
        help="Rattle amplitude (standard deviation in Angstrom)",
        min=0.0,
    ),
    random_seed: int = typer.Option(
        42, "--random-seed", "-s", help="Random seed for reproducibility"
    ),
    output_format: str = typer.Option(
        "vasp",
        "--output-format",
        "-f",
        help="Output format: vasp, cif, or xyz",
    ),
    output_prefix: str | None = typer.Option(
        None,
        "--output-prefix",
        "-p",
        help="Custom prefix for output filenames",
    ),
    apply_strain: bool = typer.Option(
        False,
        "--apply-strain/--no-strain",
        help="Apply random volumetric strain to structures",
    ),
    min_volume_ratio: float | None = typer.Option(
        0.9,
        "--min-volume-ratio",
        help="Minimum volume ratio for strain (when --apply-strain is enabled)",
    ),
    max_volume_ratio: float | None = typer.Option(
        1.05,
        "--max-volume-ratio",
        help="Maximum volume ratio for strain (when --apply-strain is enabled)",
    ),
):
    """Generate rattled structures using simple random displacements.

    Structures are generated using random Gaussian displacements with the
    specified rattle amplitude. Optionally, random volumetric strain can be
    applied to each generated structure.
    """
    # Validate output format
    fmt = output_format.lower()
    if fmt not in {"vasp", "cif", "xyz"}:
        raise typer.BadParameter(
            f"Invalid format '{output_format}'. Must be one of: vasp, cif, xyz"
        )

    # Read structure
    structure_data = StructureData.from_file(structure_file)
    atoms = structure_data.to_atoms()
    reference_positions = atoms.get_positions().copy()

    # Validate volumetric strain options
    if apply_strain:
        if min_volume_ratio is None or max_volume_ratio is None:
            raise typer.BadParameter(
                "When --apply-strain is enabled, both --min-volume-ratio "
                "and --max-volume-ratio must be specified."
            )
        if min_volume_ratio <= 0 or max_volume_ratio <= 0:
            raise typer.BadParameter("Volume ratios must be positive.")
        if min_volume_ratio > max_volume_ratio:
            raise typer.BadParameter(
                "min_volume_ratio cannot be larger than max_volume_ratio."
            )

    # Calculate zero-padding width for indices
    index_width = len(str(num_structures)) if num_structures > 0 else 1

    # Generate rattled structures
    structures = generate_rattled_structures(
        atoms=atoms,
        n_structures=num_structures,
        rattle_std=rattle_amplitude,
        seed=random_seed,
    )

    # Write structures and apply strain if requested
    for i, structure in enumerate(structures):
        if apply_strain:
            volume_ratio = np.random.uniform(min_volume_ratio, max_volume_ratio)
            scale = volume_ratio ** (1.0 / 3.0)
            structure.set_cell(structure.get_cell() * scale, scale_atoms=True)

        # Generate output filename
        if output_prefix is not None:
            filename = f"{output_prefix}{i + 1:0{index_width}d}"
        else:
            filename = f"rattle_{i + 1:0{index_width}d}"

        # Write structure
        structure_data_out = StructureData(atoms=structure)
        structure_data_out.to_file(filename, out_format=fmt)

    # Plot distributions
    plot_distributions(structures, reference_positions, rattle_amplitude)


if __name__ == "__main__":
    typer.run(rattle_cli)
