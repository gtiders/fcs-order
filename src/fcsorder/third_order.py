#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI for calculating third-order force constants using ML potentials."""

from __future__ import annotations

import copy
import itertools
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.spatial.distance
import typer
from ase.calculators.calculator import Calculator
from rich.progress import Progress

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.core import thirdorder_core
from fcsorder.io.writer import write_ifcs3, write_fc3_hdf5
from fcsorder.core.symmetry import SymmetryOperations
from fcsorder.io.reader import StructureData

# Default constants
H_DEFAULT = 1e-3  # Magnitude of finite displacements in nm
SYMPREC_DEFAULT = 1e-5  # Tolerance for symmetry search


def compute_supercell_distances(
    supercell_dict: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute minimum distances between atoms in the supercell.

    Calculates minimum interatomic distances considering periodic boundary
    conditions (27 image cells), their degeneracies, and associated shift indices.

    Args:
        supercell_dict: Supercell structure dictionary with 'lattvec' and 'positions'.

    Returns:
        Tuple of:
            - min_distances: (n_atoms, n_atoms) array of minimum distances in nm
            - n_equivalent: (n_atoms, n_atoms) count of equivalent image cells
            - shift_indices: (n_atoms, n_atoms, max_equiv) indices into 27-cell array
    """
    n_atoms = supercell_dict["positions"].shape[1]
    lattice = supercell_dict["lattvec"]
    positions_frac = supercell_dict["positions"]

    # Convert to Cartesian coordinates
    positions_cart = np.dot(lattice, positions_frac)

    # Compute squared distances for all 27 periodic images
    squared_distances = np.empty((27, n_atoms, n_atoms))
    for idx, (shift_a, shift_b, shift_c) in enumerate(
        itertools.product(range(-1, 2), range(-1, 2), range(-1, 2))
    ):
        shifted_positions = np.dot(
            lattice, (positions_frac.T + [shift_a, shift_b, shift_c]).T
        )
        squared_distances[idx, :, :] = scipy.spatial.distance.cdist(
            positions_cart.T, shifted_positions.T, "sqeuclidean"
        )

    # Find minimum distances and count equivalent images
    min_squared = squared_distances.min(axis=0)
    min_distances = np.sqrt(min_squared)

    is_equivalent = np.abs(squared_distances - min_squared) < 1e-4
    n_equivalent = is_equivalent.sum(axis=0, dtype=np.intc)
    max_equivalent = n_equivalent.max()

    # Get indices of equivalent cells (sorted by equivalence)
    sorted_indices = np.argsort(np.logical_not(is_equivalent), axis=0)
    shift_indices = np.transpose(
        sorted_indices[:max_equivalent, :, :], (1, 2, 0)
    ).astype(np.intc)

    return min_distances, n_equivalent, shift_indices


def compute_neighbor_cutoff(
    primitive_dict: dict,
    supercell_dict: dict,
    n_neighbors: int,
    min_distances: np.ndarray,
) -> float:
    """Calculate cutoff distance for nth-neighbor interactions.

    Determines the cutoff distance that includes exactly n_neighbors shells
    of neighbors around each atom in the primitive cell.

    Args:
        primitive_dict: Primitive cell dictionary with 'types'.
        supercell_dict: Supercell dictionary (unused but kept for interface).
        n_neighbors: Number of neighbor shells to include.
        min_distances: Distance matrix from compute_supercell_distances.

    Returns:
        Cutoff distance in nm (midpoint between nth and (n+1)th shells).
    """
    n_atoms = len(primitive_dict["types"])
    cutoff_candidates = []
    warned = False

    for atom_idx in range(n_atoms):
        # Get unique distance shells for this atom
        all_distances = sorted(min_distances[atom_idx, :].tolist())
        unique_shells = []
        for distance in all_distances:
            if not any(np.allclose(distance, shell) for shell in unique_shells):
                unique_shells.append(distance)

        # Calculate cutoff as midpoint between nth and (n+1)th shells
        try:
            cutoff = 0.5 * (unique_shells[n_neighbors] + unique_shells[n_neighbors + 1])
            cutoff_candidates.append(cutoff)
        except IndexError:
            if not warned:
                typer.secho(
                    "Warning: supercell too small to find n-th neighbours",
                    fg=typer.colors.RED,
                    err=True,
                )
                warned = True
            cutoff_candidates.append(1.1 * max(unique_shells))

    return max(cutoff_candidates)


def create_displaced_structure(
    structure_dict: dict,
    atom_i: int,
    coord_i: int,
    displacement_i: float,
    atom_j: int,
    coord_j: int,
    displacement_j: float,
) -> dict:
    """Create a copy of structure with two atoms displaced.

    Displaces two atoms along specified Cartesian directions. The displacements
    are converted from Cartesian to fractional coordinates internally.

    Args:
        structure_dict: Structure dictionary with 'lattvec' and 'positions'.
        atom_i: Index of first atom to displace.
        coord_i: Cartesian direction (0=x, 1=y, 2=z) for first atom.
        displacement_i: Displacement magnitude in nm for first atom.
        atom_j: Index of second atom to displace.
        coord_j: Cartesian direction (0=x, 1=y, 2=z) for second atom.
        displacement_j: Displacement magnitude in nm for second atom.

    Returns:
        Deep copy of structure_dict with displaced atom positions.
    """
    displaced = copy.deepcopy(structure_dict)
    lattice = displaced["lattvec"]

    # Displace first atom
    cartesian_disp = np.zeros(3)
    cartesian_disp[coord_i] = displacement_i
    fractional_disp = scipy.linalg.solve(lattice, cartesian_disp)
    displaced["positions"][:, atom_i] += fractional_disp

    # Displace second atom
    cartesian_disp[:] = 0.0
    cartesian_disp[coord_j] = displacement_j
    fractional_disp = scipy.linalg.solve(lattice, cartesian_disp)
    displaced["positions"][:, atom_j] += fractional_disp

    return displaced


def prepare_fc3_calculation(
    structure: StructureData,
    supercell_dims: Tuple[int, int, int],
    cutoff: str,
    symprec: float = SYMPREC_DEFAULT,
) -> Tuple[
    dict,
    dict,
    SymmetryOperations,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Optional[int],
]:
    """Prepare data structures for third-order force constant calculation.

    Args:
        structure: Primitive cell structure (StructureData object).
        supercell_dims: Supercell dimensions (na, nb, nc).
        cutoff: Cutoff value string (negative for neighbors, positive for distance in nm).
        symprec: Symmetry precision tolerance. Defaults to 1e-5.

    Returns:
        Tuple containing:
            - primitive_dict: Primitive cell dictionary
            - supercell_dict: Supercell dictionary
            - symmetry_ops: Symmetry operations object
            - min_distances: Minimum distances array
            - n_equivalent: Number of equivalent pairs
            - cell_shifts: Cell shift indices
            - force_range: Cutoff distance in nm
            - n_neighbors: Number of neighbors (or None if using distance cutoff)
    """
    na, nb, nc = supercell_dims

    # Parse cutoff: negative = nth neighbor, positive = distance in nm
    if cutoff.startswith("-"):
        n_neighbors = -int(cutoff)
        force_range = None
    else:
        n_neighbors = None
        force_range = float(cutoff)

    primitive_dict = structure.to_dict()

    typer.echo("Analyzing the symmetries")
    symmetry_ops = SymmetryOperations(
        primitive_dict["lattvec"],
        primitive_dict["types"],
        primitive_dict["positions"].T,
        symprec,
    )
    typer.echo(f"Symmetry group {symmetry_ops.symbol} detected")
    typer.echo(f"{symmetry_ops.translations.shape[0]} symmetry operations")

    typer.echo("Creating the supercell")
    supercell_dict = structure.to_supercell_dict(na, nb, nc)

    typer.echo("Computing all distances in the supercell")
    min_distances, n_equivalent, cell_shifts = compute_supercell_distances(
        supercell_dict
    )

    if n_neighbors is not None:
        force_range = compute_neighbor_cutoff(
            primitive_dict, supercell_dict, n_neighbors, min_distances
        )
        typer.echo(f"Automatic cutoff: {force_range} nm")
    else:
        typer.echo(f"User-defined cutoff: {force_range} nm")

    typer.echo("Looking for an irreducible set of third-order IFCs")

    return (
        primitive_dict,
        supercell_dict,
        symmetry_ops,
        min_distances,
        n_equivalent,
        cell_shifts,
        force_range,
        n_neighbors,
    )


def calculate_fc3(
    structure: StructureData,
    calc: Calculator,
    supercell_matrix: Tuple[int, int, int],
    cutoff: str,
    output_format: str = "text",
    save_intermediate: bool = False,
    h: float = H_DEFAULT,
    symprec: float = SYMPREC_DEFAULT,
) -> None:
    """Calculate third-order force constants using thirdorder algorithm.

    Args:
        structure: Primitive cell structure (StructureData object).
        calc: Calculator for computing forces.
        supercell_matrix: Supercell expansion (na, nb, nc).
        cutoff: Cutoff value (negative for nearest neighbors, positive for distance).
        save_intermediate: Whether to save intermediate files. Defaults to False.
        h: Magnitude of finite displacements in nm. Defaults to 1e-3.
        symprec: Symmetry precision tolerance. Defaults to 1e-5.

    Returns:
        None: Force constants are saved to FORCE_CONSTANTS_3RD file.
    """
    na, nb, nc = supercell_matrix

    # Prepare calculation data structures
    (
        primitive_dict,
        supercell_dict,
        symmetry_ops,
        min_distances,
        n_equivalent,
        cell_shifts,
        force_range,
        n_neighbors,
    ) = prepare_fc3_calculation(structure, supercell_matrix, cutoff, symprec)

    n_atoms = len(primitive_dict["types"])
    n_total = n_atoms * na * nb * nc

    # Build wedge and list of triplets
    wedge = thirdorder_core.TripletWedge(
        primitive_dict,
        supercell_dict,
        symmetry_ops,
        min_distances,
        n_equivalent,
        cell_shifts,
        force_range,
    )
    typer.echo(f"Found {wedge.nlist} triplet equivalence classes")

    triplet_list = wedge.build_list4()
    n_irreducible = len(triplet_list)
    n_runs = 4 * n_irreducible
    typer.echo(f"Total DFT runs needed: {n_runs}")

    # Build supercell and compute reference forces
    supercell_structure = StructureData.from_dict(supercell_dict)
    supercell_atoms = supercell_structure.to_atoms()
    supercell_atoms.calc = calc
    supercell_atoms.get_forces()
    supercell_atoms.write("3RD.SPOSCAR.xyz", format="extxyz")

    # Setup for displaced structures
    width = len(str(4 * (len(triplet_list) + 1)))
    name_pattern = f"3RD.POSCAR.{{:0{width}d}}.xyz"

    typer.echo("Computing an irreducible set of anharmonic force constants")
    phi_partial = np.zeros((3, n_irreducible, n_total))

    with Progress() as progress:
        task = progress.add_task("Processing triplets", total=len(triplet_list))
        for i, triplet in enumerate(triplet_list):
            for n in range(4):
                sign_i = (-1) ** (n // 2)
                sign_j = -((-1) ** (n % 2))
                run_number = n_irreducible * n + i + 1

                displaced_dict = create_displaced_structure(
                    supercell_dict,
                    triplet[1],
                    triplet[3],
                    sign_i * h,
                    triplet[0],
                    triplet[2],
                    sign_j * h,
                )

                displaced_structure = StructureData.from_dict(displaced_dict)
                displaced_atoms = displaced_structure.to_atoms()
                displaced_atoms.calc = calc
                forces = displaced_atoms.get_forces()

                # Accumulate into phi_partial
                phi_partial[:, i, :] -= sign_i * sign_j * forces.T

                if save_intermediate:
                    filename = name_pattern.format(run_number)
                    displaced_atoms.write(filename, format="extxyz")
            progress.update(task, advance=1)

    phi_partial /= 400.0 * h * h

    typer.echo("Reconstructing the full array")
    phi_full = thirdorder_core.reconstruct_ifcs(
        phi_partial, wedge, triplet_list, primitive_dict, supercell_dict
    )

    typer.echo("Writing the constants to FORCE_CONSTANTS_3RD")
    write_ifcs3(
        phi_full,
        primitive_dict,
        supercell_dict,
        min_distances,
        n_equivalent,
        cell_shifts,
        force_range,
        "FORCE_CONSTANTS_3RD",
    )

    if output_format == "hdf5":
        typer.echo("Converting output to fc3.hdf5 (using hiphive)")
        # Convert internal dicts to ASE Atoms for hiphive
        primitive_atoms = StructureData.from_dict(primitive_dict).to_atoms()
        supercell_atoms = StructureData.from_dict(supercell_dict).to_atoms()

        write_fc3_hdf5(
            primitive_atoms, supercell_atoms, "FORCE_CONSTANTS_3RD", "fc3.hdf5"
        )


def fc3(
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
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-k",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    save_intermediate: bool = typer.Option(
        False,
        "--save-intermediate/--no-save-intermediate",
        "-w",
        help="Whether to save intermediate displaced structures",
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
    h: float = typer.Option(
        H_DEFAULT,
        "--h",
        help="Magnitude of finite displacements in nm",
    ),
    symprec: float = typer.Option(
        SYMPREC_DEFAULT,
        "--symprec",
        help="Symmetry precision tolerance",
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        "-f",
        help="Output format: text or hdf5",
    ),
):
    """
    Calculate third-order force constants using any registered calculator.

    Example:
        fc3 2 2 2 --calculator nep --potential model.nep --cutoff -3
        fc3 2 2 2 --calculator dp --potential model.pb --cutoff 0.5
        fc3 2 2 2 --calculator tace --potential model.pt --cutoff -4 --device cuda
    """
    # Read structure using StructureData
    typer.echo(f"Reading structure from {structure_file}")
    structure = StructureData.from_file(structure_file)
    structure_atoms = structure.to_atoms()

    # Build supercell using StructureData's make_supercell method
    supercell_dict = structure.to_supercell_dict(na, nb, nc)
    supercell_structure = structure.from_dict(supercell_dict)
    supercell_atoms = supercell_structure.to_atoms()
    typer.echo(f"Supercell matrix: {na} x {nb} x {nc}")

    # Build calculator arguments
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

    # Run third-order force constants calculation
    typer.echo("Starting third-order force constants calculation...")
    try:
        calculate_fc3(
            structure=structure,
            calc=calc,
            supercell_matrix=(na, nb, nc),
            cutoff=cutoff,
            output_format=output_format,
            save_intermediate=save_intermediate,
            h=h,
            symprec=symprec,
        )
        typer.secho(
            "✓ Third-order force constants calculation completed successfully",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(f"✗ Force constants calculation failed: {e}", fg=typer.colors.RED)
        sys.exit(1)
