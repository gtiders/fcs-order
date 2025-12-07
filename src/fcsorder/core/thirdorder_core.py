#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions for third-order force constant calculation and reconstruction.

This module provides:
- TripletWedge: Class for symmetry-based IFC reduction (formerly Wedge)
- reconstruct_ifcs: Reconstruct full IFC tensor from irreducible set
- gaussian_elimination: Specialized Gaussian elimination for finding independent basis
- Numba-optimized helper functions for sparse matrix construction
"""

from __future__ import annotations

from math import fabs
from typing import Tuple

import numpy as np
import scipy as sp
import sparse
import typer
from numba import jit, types
from numba.typed import List
from rich.progress import Progress

from fcsorder.core.symmetry import SymmetryOperations


# =============================================================================
# Constants
# =============================================================================

# Tolerance for Gaussian elimination
GAUSSIAN_EPS = 1e-10

# Permutations of triplet indices (order matters for symmetry operations)
TRIPLET_PERMUTATIONS = np.array(
    [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]],
    dtype=np.intc,
)


@jit(nopython=True)
def gaussian_elimination(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find independent IFC basis via Gaussian elimination.

    Performs row reduction to identify dependent and independent columns,
    returning a transformation matrix and list of independent column indices.
    Matrix must be contiguous float64 array.

    Args:
        matrix: 2D coefficient matrix (modified in place during reduction).

    Returns:
        Tuple of (basis_transform, independent_indices).
    """
    n_rows, n_cols = matrix.shape

    dependent = np.empty(n_cols, dtype=np.intc)
    independent = np.empty(n_cols, dtype=np.intc)
    basis_transform = np.zeros((n_cols, n_cols), dtype=np.double)

    current_row = 0
    n_dependent = 0
    n_independent = 0

    for col in range(min(n_rows, n_cols)):
        # Zero out near-zero entries in current column
        col_values = matrix[:, col]
        col_values[np.abs(col_values) < GAUSSIAN_EPS] = 0.0

        # Swap rows to get largest pivot
        if current_row < n_rows:
            for row in range(current_row + 1, n_rows):
                if abs(matrix[row, col]) - abs(matrix[current_row, col]) > GAUSSIAN_EPS:
                    tmp = matrix[current_row, col:n_cols].copy()
                    matrix[current_row, col:n_cols] = matrix[row, col:n_cols]
                    matrix[row, col:n_cols] = tmp

        if current_row < n_rows and abs(matrix[current_row, col]) > GAUSSIAN_EPS:
            # This column is dependent
            dependent[n_dependent] = col
            n_dependent += 1

            # Normalize pivot row
            pivot = matrix[current_row, col]
            if n_cols - 1 > col:
                matrix[current_row, col + 1 : n_cols] /= pivot
            matrix[current_row, col] = 1.0

            # Eliminate other rows
            for row in range(n_rows):
                if row == current_row:
                    continue
                if n_cols - 1 > col:
                    factor = matrix[row, col] / matrix[current_row, col]
                    matrix[row, col + 1 : n_cols] -= (
                        factor * matrix[current_row, col + 1 : n_cols]
                    )
                matrix[row, col] = 0.0

            if current_row < n_rows - 1:
                current_row += 1
        else:
            # This column is independent
            independent[n_independent] = col
            n_independent += 1

    # Build transformation matrix from reduced form
    for j in range(n_independent):
        for i in range(n_dependent):
            basis_transform[dependent[i], j] = -matrix[i, independent[j]]
        basis_transform[independent[j], j] = 1.0

    return basis_transform, independent[:n_independent]


# =============================================================================
# Numba helper functions for Wedge operations
# =============================================================================


@jit(nopython=True)
def _cell_atom_to_index(
    cell: np.ndarray, species: int, grid: np.ndarray, n_species: int
) -> int:
    """Convert cell coordinates and species to supercell atom index.

    Args:
        cell: (3,) array of cell indices.
        species: Atom species index within unit cell.
        grid: (3,) supercell dimensions.
        n_species: Number of atoms per primitive cell.

    Returns:
        Linear index into supercell atom array.
    """
    return (cell[0] + (cell[1] + cell[2] * grid[1]) * grid[0]) * n_species + species


@jit(nopython=True)
def _triplet_in_list(
    triplet: np.ndarray, triplet_list: np.ndarray, n_list: int
) -> bool:
    """Check if triplet exists in the list.

    Args:
        triplet: (3,) array of atom indices.
        triplet_list: (3, N) array of stored triplets.
        n_list: Number of valid triplets in list.

    Returns:
        True if triplet is found in list.
    """
    for i in range(n_list):
        if (
            triplet[0] == triplet_list[0, i]
            and triplet[1] == triplet_list[1, i]
            and triplet[2] == triplet_list[2, i]
        ):
            return True
    return False


@jit(nopython=True)
def _triplets_equal(triplet1: np.ndarray, triplet2: np.ndarray) -> bool:
    """Check if two triplets are identical."""
    for i in range(3):
        if triplet1[i] != triplet2[i]:
            return False
    return True


@jit(nopython=True)
def _index_to_cell_atom(
    grid: np.ndarray, n_species: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create mapping from linear index to (cell, species).

    Args:
        grid: (3,) supercell dimensions.
        n_species: Atoms per primitive cell.

    Returns:
        Tuple of (cell_indices, species_indices) with shapes (3, n_total) and (n_total,).
    """
    n_total = grid[0] * grid[1] * grid[2] * n_species
    cell_indices = np.empty((3, n_total), dtype=np.intc)
    species_indices = np.empty(n_total, dtype=np.intc)

    for idx in range(n_total):
        tmp, species_indices[idx] = divmod(idx, n_species)
        tmp, cell_indices[0, idx] = divmod(tmp, grid[0])
        cell_indices[2, idx], cell_indices[1, idx] = divmod(tmp, grid[1])

    return cell_indices, species_indices


@jit(nopython=True)
def _build_transformation_product(
    transformation: np.ndarray,
    transformation_aux: np.ndarray,
    n_equivalent: np.ndarray,
    n_independent_basis: np.ndarray,
    n_list: int,
    output: np.ndarray,
) -> None:
    """Compute product of transformation matrices for all equivalence classes.

    Computes: output[k, l, j, i] = sum_aux transformation[k, aux, j, i] * transformation_aux[aux, l, i]

    Args:
        transformation: (27, 27, max_equiv, n_list) rotation matrices.
        transformation_aux: (27, 27, n_list) auxiliary transformation.
        n_equivalent: (n_list,) count of equivalent triplets per class.
        n_independent_basis: (n_list,) count of independent basis per class.
        n_list: Number of triplet equivalence classes.
        output: (27, 27, max_equiv, n_list) output array (modified in place).
    """
    for list_idx in range(n_list):
        n_equiv = n_equivalent[list_idx]
        n_indep = n_independent_basis[list_idx]

        for equiv_idx in range(n_equiv):
            for basis_k in range(27):
                for basis_l in range(n_indep):
                    value = 0.0
                    for aux_idx in range(27):
                        value += (
                            transformation[basis_k, aux_idx, equiv_idx, list_idx]
                            * transformation_aux[aux_idx, basis_l, list_idx]
                        )
                    # Zero out very small values
                    if value != 0.0 and abs(value) < 1e-12:
                        value = 0.0
                    output[basis_k, basis_l, equiv_idx, list_idx] = value


# =============================================================================
# TripletWedge class for third-order IFC reduction
# =============================================================================


class TripletWedge:
    """Symmetry-based reduction of third-order interatomic force constants.

    This class identifies irreducible sets of third-order force constants
    by exploiting crystal symmetry and permutation invariance. It stores
    the transformation matrices needed to reconstruct the full IFC tensor.

    Attributes:
        nlist: Number of irreducible triplet equivalence classes.
        nequi: (n_list,) count of equivalent triplets per class.
        llist: (3, n_list) representative triplet for each class.
        allequilist: (3, max_equiv, n_list) all equivalent triplets.
        nindependentbasis: (n_list,) count of independent basis elements.
        independentbasis: (27, n_list) independent basis indices.
        transformationarray: (27, 27, max_equiv, n_list) transformation matrices.
    """

    def __init__(
        self,
        primitive_dict: dict,
        supercell_dict: dict,
        symmetry_ops: SymmetryOperations,
        min_distances: np.ndarray,
        n_equivalent: np.ndarray,
        cell_shifts: np.ndarray,
        force_range: float,
    ):
        """Initialize and compute irreducible triplet set.

        Args:
            primitive_dict: Primitive cell structure dictionary.
            supercell_dict: Supercell structure dictionary with 'na', 'nb', 'nc'.
            symmetry_ops: SymmetryOperations object for the crystal.
            min_distances: (n_atoms, n_total) minimum distance matrix.
            n_equivalent: (n_atoms, n_total) count of equivalent image cells.
            cell_shifts: (n_atoms, n_total, max_equiv) shift indices.
            force_range: Cutoff distance for interactions in nm.
        """
        self.primitive_dict = primitive_dict
        self.supercell_dict = supercell_dict
        self.symmetry_ops = symmetry_ops
        self.min_distances = min_distances
        self.n_equivalent_images = n_equivalent
        self.cell_shifts = cell_shifts
        self.force_range = force_range

        # Initialize dynamic arrays
        self._alloc_size = 0
        self._all_alloc_size = 0
        self._expand_arrays()
        self._expand_all_list()

        # Perform the reduction
        self._reduce()

    def _expand_arrays(self) -> None:
        """Double the size of main storage arrays."""
        if self._alloc_size == 0:
            self._alloc_size = 16
            max_equiv = 6 * self.symmetry_ops.nsyms

            self.nequi = np.empty(self._alloc_size, dtype=np.intc)
            self.allequilist = np.empty((3, max_equiv, self._alloc_size), dtype=np.intc)
            self.transformationarray = np.empty(
                (27, 27, max_equiv, self._alloc_size), dtype=np.double
            )
            self._transformation = np.empty(
                (27, 27, max_equiv, self._alloc_size), dtype=np.double
            )
            self._transformation_aux = np.empty(
                (27, 27, self._alloc_size), dtype=np.double
            )
            self.nindependentbasis = np.empty(self._alloc_size, dtype=np.intc)
            self.independentbasis = np.empty((27, self._alloc_size), dtype=np.intc)
            self.llist = np.empty((3, self._alloc_size), dtype=np.intc)
        else:
            self._alloc_size *= 2
            self.nequi = np.concatenate((self.nequi, self.nequi), axis=-1)
            self.allequilist = np.concatenate(
                (self.allequilist, self.allequilist), axis=-1
            )
            self.transformationarray = np.concatenate(
                (self.transformationarray, self.transformationarray), axis=-1
            )
            self._transformation = np.concatenate(
                (self._transformation, self._transformation), axis=-1
            )
            self._transformation_aux = np.concatenate(
                (self._transformation_aux, self._transformation_aux), axis=-1
            )
            self.nindependentbasis = np.concatenate(
                (self.nindependentbasis, self.nindependentbasis), axis=-1
            )
            self.independentbasis = np.concatenate(
                (self.independentbasis, self.independentbasis), axis=-1
            )
            self.llist = np.concatenate((self.llist, self.llist), axis=-1)

    def _expand_all_list(self) -> None:
        """Double the size of complete triplet list."""
        if self._all_alloc_size == 0:
            self._all_alloc_size = 512
            self._all_list = np.empty((3, self._all_alloc_size), dtype=np.intc)
        else:
            self._all_alloc_size *= 2
            self._all_list = np.concatenate((self._all_list, self._all_list), axis=-1)

    def _reduce(self) -> None:
        """Identify irreducible triplet classes using symmetry reduction."""
        force_range_sq = self.force_range * self.force_range

        # Extract supercell dimensions
        grid = np.array(
            [
                self.supercell_dict["na"],
                self.supercell_dict["nb"],
                self.supercell_dict["nc"],
            ],
            dtype=np.intc,
        )

        n_symmetries = self.symmetry_ops.nsyms
        n_atoms = len(self.primitive_dict["types"])
        n_total = len(self.supercell_dict["types"])

        lattice = self.supercell_dict["lattvec"]
        cart_coords = np.dot(lattice, self.supercell_dict["positions"])
        rotation_tensors = np.transpose(self.symmetry_ops.crotations, (1, 2, 0))

        self.nlist = 0
        self._n_all_list = 0

        # Build 27 neighbor shifts in [-1,0,1]^3
        shifts_27 = np.empty((27, 3), dtype=np.intc)
        idx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shifts_27[idx] = [i, j, k]
                    idx += 1

        # Map supercell atoms to (cell, species) indices
        equiv_atom_map = self.symmetry_ops.map_supercell(self.supercell_dict)
        cell_indices, species_indices = _index_to_cell_atom(grid, n_atoms)

        # Build rotation matrices for third-order tensor indices
        rotation_matrices, rotation_identity_diff, has_nonzero = (
            self._build_rotation_matrices(n_symmetries, rotation_tensors)
        )

        # Temporary arrays
        triplet = np.empty(3, dtype=np.intc)
        triplet_permuted = np.empty(3, dtype=np.intc)
        triplet_symmetric = np.empty(3, dtype=np.intc)
        equiv_triplets = np.empty((3, n_symmetries * 6), dtype=np.intc)
        coeff_matrix = np.empty((6 * n_symmetries * 27, 27), dtype=np.double)
        shift_i = np.empty((3, 27), dtype=np.intc)
        shift_j = np.empty((3, 27), dtype=np.intc)

        # Main loop over atom triplets
        with Progress() as progress:
            atom_task = progress.add_task("Scanning atom triplets", total=n_atoms)

            for atom_i in range(n_atoms):
                for atom_j in range(n_total):
                    dist_ij = self.min_distances[atom_i, atom_j]
                    if dist_ij >= self.force_range:
                        continue

                    n_equiv_ij = self.n_equivalent_images[atom_i, atom_j]
                    for k in range(n_equiv_ij):
                        shift_i[:, k] = shifts_27[self.cell_shifts[atom_i, atom_j, k]]

                    for atom_k in range(n_total):
                        dist_ik = self.min_distances[atom_i, atom_k]
                        if dist_ik >= self.force_range:
                            continue

                        n_equiv_ik = self.n_equivalent_images[atom_i, atom_k]
                        for l in range(n_equiv_ik):
                            shift_j[:, l] = shifts_27[
                                self.cell_shifts[atom_i, atom_k, l]
                            ]

                        # Check if jk distance is within range
                        min_dist_jk_sq = np.inf
                        for idx_i in range(n_equiv_ij):
                            cart_j = (
                                shift_i[:, idx_i : idx_i + 1].T @ lattice.T
                            ).flatten() + cart_coords[:, atom_j]

                            for idx_j in range(n_equiv_ik):
                                cart_k = (
                                    shift_j[:, idx_j : idx_j + 1].T @ lattice.T
                                ).flatten() + cart_coords[:, atom_k]

                                dist_sq = np.sum((cart_k - cart_j) ** 2)
                                if dist_sq < min_dist_jk_sq:
                                    min_dist_jk_sq = dist_sq

                        if min_dist_jk_sq >= force_range_sq:
                            continue

                        # Check if already processed
                        triplet[:] = [atom_i, atom_j, atom_k]
                        if _triplet_in_list(triplet, self._all_list, self._n_all_list):
                            continue

                        # Add new equivalence class
                        self.nlist += 1
                        if self.nlist == self._alloc_size:
                            self._expand_arrays()

                        idx = self.nlist - 1
                        self.llist[:, idx] = triplet
                        self.nequi[idx] = 0
                        coeff_matrix[:] = 0.0
                        n_nonzero = 0

                        # Scan permutations and symmetries
                        for perm_idx in range(6):
                            perm = TRIPLET_PERMUTATIONS[perm_idx]
                            triplet_permuted[:] = triplet[perm]

                            for sym_idx in range(n_symmetries):
                                # Apply symmetry operation
                                for d in range(3):
                                    triplet_symmetric[d] = equiv_atom_map[
                                        sym_idx, triplet_permuted[d]
                                    ]

                                # Translate so first atom is in origin cell
                                vec1 = cell_indices[:, triplet_symmetric[0]].copy()
                                if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                    vec2 = (
                                        cell_indices[:, triplet_symmetric[1]] - vec1
                                    ) % grid
                                    vec3 = (
                                        cell_indices[:, triplet_symmetric[2]] - vec1
                                    ) % grid
                                    vec1[:] = 0

                                    triplet_symmetric[0] = _cell_atom_to_index(
                                        vec1,
                                        species_indices[triplet_symmetric[0]],
                                        grid,
                                        n_atoms,
                                    )
                                    triplet_symmetric[1] = _cell_atom_to_index(
                                        vec2,
                                        species_indices[triplet_symmetric[1]],
                                        grid,
                                        n_atoms,
                                    )
                                    triplet_symmetric[2] = _cell_atom_to_index(
                                        vec3,
                                        species_indices[triplet_symmetric[2]],
                                        grid,
                                        n_atoms,
                                    )

                                # Add if new equivalent triplet
                                if (perm_idx == 0 and sym_idx == 0) or not (
                                    _triplets_equal(triplet_symmetric, triplet)
                                    or _triplet_in_list(
                                        triplet_symmetric,
                                        equiv_triplets,
                                        self.nequi[idx],
                                    )
                                ):
                                    eq = self.nequi[idx]
                                    self.nequi[idx] += 1

                                    equiv_triplets[:, eq] = triplet_symmetric
                                    self.allequilist[:, eq, idx] = triplet_symmetric

                                    self._n_all_list += 1
                                    if self._n_all_list == self._all_alloc_size:
                                        self._expand_all_list()
                                    self._all_list[:, self._n_all_list - 1] = (
                                        triplet_symmetric
                                    )

                                    self._transformation[:, :, eq, idx] = (
                                        rotation_matrices[perm_idx, sym_idx]
                                    )

                                # Add constraint if maps to self
                                if _triplets_equal(triplet_symmetric, triplet):
                                    for basis_idx in range(27):
                                        if has_nonzero[perm_idx, sym_idx, basis_idx]:
                                            coeff_matrix[n_nonzero, :] = (
                                                rotation_identity_diff[
                                                    perm_idx, sym_idx, basis_idx, :
                                                ]
                                            )
                                            n_nonzero += 1

                        # Solve for independent basis
                        coeff_reduced = np.ascontiguousarray(
                            np.zeros((max(n_nonzero, 27), 27), dtype=np.double)
                        )
                        coeff_reduced[:n_nonzero, :] = coeff_matrix[:n_nonzero, :]

                        basis_transform, independent = gaussian_elimination(
                            coeff_reduced
                        )
                        self._transformation_aux[:, :, idx] = basis_transform
                        self.nindependentbasis[idx] = len(independent)
                        self.independentbasis[: len(independent), idx] = independent

                progress.update(atom_task, advance=1)

        # Build final transformation arrays
        self.transformationarray[:] = 0.0
        _build_transformation_product(
            self._transformation,
            self._transformation_aux,
            self.nequi,
            self.nindependentbasis,
            self.nlist,
            self.transformationarray,
        )

    def _build_rotation_matrices(
        self,
        n_symmetries: int,
        rotation_tensors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build rotation matrices for third-order tensor transformation.

        Returns:
            Tuple of (rotation_matrices, rotation_identity_diff, has_nonzero).
        """
        rotation_matrices = np.empty((6, n_symmetries, 27, 27), dtype=np.double)

        for perm_idx in range(6):
            perm = TRIPLET_PERMUTATIONS[perm_idx]
            for sym_idx in range(n_symmetries):
                for i_prime in range(3):
                    for j_prime in range(3):
                        for k_prime in range(3):
                            idx_prime = (i_prime * 3 + j_prime) * 3 + k_prime
                            for i in range(3):
                                for j in range(3):
                                    for k in range(3):
                                        idx = i * 9 + j * 3 + k
                                        basis = np.array([i, j, k])
                                        i_perm = basis[perm[0]]
                                        j_perm = basis[perm[1]]
                                        k_perm = basis[perm[2]]
                                        rotation_matrices[
                                            perm_idx, sym_idx, idx_prime, idx
                                        ] = (
                                            rotation_tensors[i_prime, i_perm, sym_idx]
                                            * rotation_tensors[j_prime, j_perm, sym_idx]
                                            * rotation_tensors[k_prime, k_perm, sym_idx]
                                        )

        # Compute R - I for constraint equations
        rotation_identity_diff = rotation_matrices.copy()
        has_nonzero = np.zeros((6, n_symmetries, 27), dtype=np.intc)

        for perm_idx in range(6):
            for sym_idx in range(n_symmetries):
                for idx_prime in range(27):
                    rotation_identity_diff[perm_idx, sym_idx, idx_prime, idx_prime] -= (
                        1.0
                    )
                    for idx in range(27):
                        if (
                            fabs(
                                rotation_identity_diff[
                                    perm_idx, sym_idx, idx_prime, idx
                                ]
                            )
                            > 1e-12
                        ):
                            has_nonzero[perm_idx, sym_idx, idx_prime] = 1
                        else:
                            rotation_identity_diff[
                                perm_idx, sym_idx, idx_prime, idx
                            ] = 0.0

        return rotation_matrices, rotation_identity_diff, has_nonzero

    def build_triplet_list(self) -> list:
        """Build list of irreducible triplets for force calculations.

        Returns:
            List of tuples (atom_j, atom_k, coord_l, coord_m) representing
            the irreducible triplets that need force calculations.
        """
        extended_list = []
        for list_idx in range(self.nlist):
            for basis_idx in range(self.nindependentbasis[list_idx]):
                basis_val = self.independentbasis[basis_idx, list_idx]
                coord_l = basis_val // 9
                coord_m = (basis_val % 9) // 3
                coord_n = basis_val % 3
                extended_list.append(
                    (
                        coord_l,
                        self.llist[0, list_idx],
                        coord_m,
                        self.llist[1, list_idx],
                        coord_n,
                        self.llist[2, list_idx],
                    )
                )

        # Remove duplicates while preserving order
        unique_list = []
        for item in extended_list:
            four_tuple = (item[1], item[3], item[0], item[2])
            if four_tuple not in unique_list:
                unique_list.append(four_tuple)

        return unique_list

    # Alias for backward compatibility
    build_list4 = build_triplet_list


# =============================================================================
# Numba-optimized sparse matrix construction helpers
# =============================================================================


@jit(nopython=True)
def _build_sparse_triplets(
    triplet_class_idx: np.ndarray,
    equiv_class_idx: np.ndarray,
    accumulated_independent: np.ndarray,
    n_atoms: int,
    n_total: int,
    n_list: int,
    transformation_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build COO sparse matrix triplets for IFC reconstruction."""
    # Pass 1: Count total elements
    total_count = 0
    for atom_i in range(n_atoms):
        for atom_j in range(n_total):
            for _ in range(27):
                for atom_k in range(n_total):
                    for list_idx in range(n_list):
                        if triplet_class_idx[atom_i, atom_j, atom_k] == list_idx:
                            total_count += (
                                accumulated_independent[list_idx + 1]
                                - accumulated_independent[list_idx]
                            )

    # Allocate arrays
    row_indices = np.empty(total_count, dtype=np.int64)
    col_indices = np.empty(total_count, dtype=np.int64)
    values = np.empty(total_count, dtype=np.float64)

    # Pass 2: Fill arrays
    count = 0
    col_idx = 0
    for atom_i in range(n_atoms):
        for atom_j in range(n_total):
            basis_idx = 0
            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        for atom_k in range(n_total):
                            for list_idx in range(n_list):
                                if (
                                    triplet_class_idx[atom_i, atom_j, atom_k]
                                    == list_idx
                                ):
                                    start = accumulated_independent[list_idx]
                                    end = accumulated_independent[list_idx + 1]
                                    for ss in range(start, end):
                                        offset = ss - start
                                        equiv_idx = equiv_class_idx[
                                            atom_i, atom_j, atom_k
                                        ]
                                        row_indices[count] = ss
                                        col_indices[count] = col_idx
                                        values[count] = transformation_array[
                                            basis_idx, offset, equiv_idx, list_idx
                                        ]
                                        count += 1
                        basis_idx += 1
                        col_idx += 1

    return row_indices, col_indices, values


@jit(nopython=True)
def _build_full_ifc_coordinates(
    n_list: int,
    n_equivalent: np.ndarray,
    n_independent_basis: np.ndarray,
    equiv_list: np.ndarray,
    accumulated_independent: np.ndarray,
    transformation_array: np.ndarray,
    phi_values: np.ndarray,
) -> Tuple:
    """Build coordinate lists for final sparse IFC tensor."""
    coord0_list = List.empty_list(types.int64)
    coord1_list = List.empty_list(types.int64)
    coord2_list = List.empty_list(types.int64)
    atom0_list = List.empty_list(types.int64)
    atom1_list = List.empty_list(types.int64)
    atom2_list = List.empty_list(types.int64)
    value_list = List.empty_list(types.float64)

    for list_idx in range(n_list):
        n_equiv = n_equivalent[list_idx]
        n_indep = n_independent_basis[list_idx]

        for equiv_idx in range(n_equiv):
            equiv_atom0 = equiv_list[0, equiv_idx, list_idx]
            equiv_atom1 = equiv_list[1, equiv_idx, list_idx]
            equiv_atom2 = equiv_list[2, equiv_idx, list_idx]

            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        basis_idx = (coord_l * 3 + coord_m) * 3 + coord_n

                        for indep_idx in range(n_indep):
                            phi_idx = accumulated_independent[list_idx] + indep_idx
                            val = (
                                transformation_array[
                                    basis_idx, indep_idx, equiv_idx, list_idx
                                ]
                                * phi_values[phi_idx]
                            )
                            if val == 0.0:
                                continue

                            coord0_list.append(coord_l)
                            coord1_list.append(coord_m)
                            coord2_list.append(coord_n)
                            atom0_list.append(equiv_atom0)
                            atom1_list.append(equiv_atom1)
                            atom2_list.append(equiv_atom2)
                            value_list.append(val)

    return (
        coord0_list,
        coord1_list,
        coord2_list,
        atom0_list,
        atom1_list,
        atom2_list,
        value_list,
    )


# =============================================================================
# Main reconstruction function
# =============================================================================


def reconstruct_ifcs(
    phi_partial: np.ndarray,
    wedge: TripletWedge,
    triplet_list: list,
    primitive_dict: dict,
    supercell_dict: dict,
) -> sparse.COO:
    """Reconstruct full third-order IFC tensor from irreducible set.

    Args:
        phi_partial: (3, n_irreducible, n_total) computed force constants.
        wedge: TripletWedge object with symmetry information.
        triplet_list: List of irreducible triplets.
        primitive_dict: Primitive cell dictionary.
        supercell_dict: Supercell dictionary.

    Returns:
        Sparse COO tensor of shape (3, 3, 3, n_atoms, n_total, n_total).
    """
    n_list = wedge.nlist
    n_atoms = len(primitive_dict["types"])
    n_total = len(supercell_dict["types"])

    typer.echo("Using sparse method with DOK sparse matrix")

    # Step 1: Store partial results
    result_sparse = sparse.zeros((3, 3, 3, n_atoms, n_total, n_total), format="dok")
    accumulated_independent = np.insert(
        np.cumsum(wedge.nindependentbasis[:n_list], dtype=np.intc),
        0,
        np.zeros(1, dtype=np.intc),
    )
    n_total_independent = accumulated_independent[-1]

    with Progress() as progress:
        task = progress.add_task("Storing partial results", total=len(triplet_list))
        for idx, (atom_j, atom_k, coord_l, coord_m) in enumerate(triplet_list):
            result_sparse[coord_l, coord_m, :, atom_j, atom_k, :] = phi_partial[
                :, idx, :
            ]
            progress.update(task, advance=1)

    result_sparse = result_sparse.to_coo()

    # Step 2: Extract phi values
    phi_values_list = []
    with Progress() as progress:
        task = progress.add_task("Extracting phi values", total=n_list)
        for list_idx in range(n_list):
            for basis_idx in range(wedge.nindependentbasis[list_idx]):
                basis_val = wedge.independentbasis[basis_idx, list_idx]
                coord_l = basis_val // 9
                coord_m = (basis_val % 9) // 3
                coord_n = basis_val % 3

                phi_values_list.append(
                    result_sparse[
                        coord_l,
                        coord_m,
                        coord_n,
                        wedge.llist[0, list_idx],
                        wedge.llist[1, list_idx],
                        wedge.llist[2, list_idx],
                    ]
                )
            progress.update(task, advance=1)

    phi_values = np.array(phi_values_list, dtype=np.double)

    # Step 3: Build index mappings
    triplet_class_idx = -np.ones((n_atoms, n_total, n_total), dtype=np.intc)
    equiv_class_idx = -np.ones((n_atoms, n_total, n_total), dtype=np.intc)
    equiv_list = wedge.allequilist

    for list_idx in range(n_list):
        for equiv_idx in range(wedge.nequi[list_idx]):
            i, j, k = (
                equiv_list[0, equiv_idx, list_idx],
                equiv_list[1, equiv_idx, list_idx],
                equiv_list[2, equiv_idx, list_idx],
            )
            triplet_class_idx[i, j, k] = list_idx
            equiv_class_idx[i, j, k] = equiv_idx

    # Step 4: Build sparse coefficient matrix
    typer.echo("Building sparse coefficient matrix")
    row_indices, col_indices, values = _build_sparse_triplets(
        triplet_class_idx,
        equiv_class_idx,
        accumulated_independent,
        n_atoms,
        n_total,
        n_list,
        wedge.transformationarray,
    )

    coeff_matrix = sp.sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        (n_total_independent, n_atoms * n_total * 27),
    ).tocsr()

    # Step 5: Apply acoustic sum rule
    diag_matrix = sp.sparse.spdiags(
        phi_values, [0], phi_values.size, phi_values.size, format="csr"
    )
    weighted_coeff = diag_matrix.dot(coeff_matrix)
    multiplier = -sp.sparse.linalg.lsqr(weighted_coeff, np.ones_like(phi_values))[0]
    phi_values += diag_matrix.dot(weighted_coeff.dot(multiplier))

    # Step 6: Build final tensor
    typer.echo("Building final full IFC tensor")
    coords = _build_full_ifc_coordinates(
        n_list,
        wedge.nequi,
        wedge.nindependentbasis,
        equiv_list,
        accumulated_independent,
        wedge.transformationarray,
        phi_values,
    )

    final_coords = np.array(
        [np.array(coords[i], dtype=np.intp) for i in range(6)], dtype=np.intp
    )

    return sparse.COO(
        final_coords,
        np.array(coords[6], dtype=np.double),
        shape=(3, 3, 3, n_atoms, n_total, n_total),
        has_duplicates=True,
    )
