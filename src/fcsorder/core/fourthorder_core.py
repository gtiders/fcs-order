#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions for fourth-order force constant calculation and reconstruction.

This module provides:
- QuartetWedge: Class for symmetry-based IFC reduction (formerly Wedge)
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

GAUSSIAN_EPS = 1e-10

# 24 permutations of 4 elements (order matches old Fortran code)
QUARTET_PERMUTATIONS = np.array(
    [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 1, 3, 2],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [0, 2, 3, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=np.int64,
)


# =============================================================================
# Gaussian elimination
# =============================================================================


@jit(nopython=True)
def gaussian_elimination(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find independent IFC basis via Gaussian elimination."""
    n_rows, n_cols = matrix.shape
    dependent = np.empty(n_cols, dtype=np.int64)
    independent = np.empty(n_cols, dtype=np.int64)
    basis_transform = np.zeros((n_cols, n_cols), dtype=np.float64)

    current_row = 0
    n_dependent = 0
    n_independent = 0

    for col in range(min(n_rows, n_cols)):
        col_values = matrix[:, col]
        col_values[np.abs(col_values) < GAUSSIAN_EPS] = 0.0

        if current_row < n_rows:
            for row in range(current_row + 1, n_rows):
                if abs(matrix[row, col]) - abs(matrix[current_row, col]) > GAUSSIAN_EPS:
                    tmp = matrix[current_row, col:n_cols].copy()
                    matrix[current_row, col:n_cols] = matrix[row, col:n_cols]
                    matrix[row, col:n_cols] = tmp

        if current_row < n_rows and abs(matrix[current_row, col]) > GAUSSIAN_EPS:
            dependent[n_dependent] = col
            n_dependent += 1
            pivot = matrix[current_row, col]
            if n_cols - 1 > col:
                matrix[current_row, col + 1 : n_cols] /= pivot
            matrix[current_row, col] = 1.0

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
            independent[n_independent] = col
            n_independent += 1

    for j in range(n_independent):
        for i in range(n_dependent):
            basis_transform[dependent[i], j] = -matrix[i, independent[j]]
        basis_transform[independent[j], j] = 1.0

    return basis_transform, independent[:n_independent]


# =============================================================================
# Numba helper functions
# =============================================================================


@jit(nopython=True)
def _cell_atom_to_index(
    cell: np.ndarray, species: int, grid: np.ndarray, n_species: int
) -> int:
    """Convert cell coordinates and species to supercell atom index."""
    return (cell[0] + (cell[1] + cell[2] * grid[1]) * grid[0]) * n_species + species


@jit(nopython=True)
def _quartet_in_list(
    quartet: np.ndarray, quartet_list: np.ndarray, n_list: int
) -> bool:
    """Check if quartet exists in the list."""
    for i in range(n_list):
        if (
            quartet[0] == quartet_list[0, i]
            and quartet[1] == quartet_list[1, i]
            and quartet[2] == quartet_list[2, i]
            and quartet[3] == quartet_list[3, i]
        ):
            return True
    return False


@jit(nopython=True)
def _quartets_equal(q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if two quartets are identical."""
    for i in range(4):
        if q1[i] != q2[i]:
            return False
    return True


@jit(nopython=True)
def _index_to_cell_atom(
    grid: np.ndarray, n_species: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create mapping from linear index to (cell, species)."""
    n_total = grid[0] * grid[1] * grid[2] * n_species
    cell_indices = np.empty((3, n_total), dtype=np.int64)
    species_indices = np.empty(n_total, dtype=np.int64)
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
    """Compute product of transformation matrices for all equivalence classes."""
    for idx in range(n_list):
        n_equiv = n_equivalent[idx]
        n_indep = n_independent_basis[idx]
        for eq in range(n_equiv):
            for k in range(81):
                for l in range(n_indep):
                    value = 0.0
                    for aux in range(81):
                        value += (
                            transformation[k, aux, eq, idx]
                            * transformation_aux[aux, l, idx]
                        )
                    if value != 0.0 and abs(value) < 1e-15:
                        value = 0.0
                    output[k, l, eq, idx] = value


@jit(nopython=True)
def _build_sparse_quartets(
    quartet_class_idx: np.ndarray,
    equiv_class_idx: np.ndarray,
    accumulated_independent: np.ndarray,
    n_atoms: int,
    n_total: int,
    n_list: int,
    transformation_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build COO sparse matrix triplets for IFC reconstruction."""
    # Count total elements
    total_count = 0
    for i in range(n_atoms):
        for j in range(n_total):
            for _ in range(81):
                for k in range(n_total):
                    for l in range(n_total):
                        for idx in range(n_list):
                            if quartet_class_idx[i, j, k, l] == idx:
                                total_count += (
                                    accumulated_independent[idx + 1]
                                    - accumulated_independent[idx]
                                )

    row_indices = np.empty(total_count, dtype=np.int64)
    col_indices = np.empty(total_count, dtype=np.int64)
    values = np.empty(total_count, dtype=np.float64)

    count = 0
    col_idx = 0
    for i in range(n_atoms):
        for j in range(n_total):
            basis = 0
            for _ in range(81):
                for k in range(n_total):
                    for l in range(n_total):
                        for idx in range(n_list):
                            if quartet_class_idx[i, j, k, l] == idx:
                                start = accumulated_independent[idx]
                                end = accumulated_independent[idx + 1]
                                for s in range(start, end):
                                    offset = s - start
                                    eq = equiv_class_idx[i, j, k, l]
                                    row_indices[count] = s
                                    col_indices[count] = col_idx
                                    values[count] = transformation_array[
                                        basis, offset, eq, idx
                                    ]
                                    count += 1
                basis += 1
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
    c0 = List.empty_list(types.int64)
    c1 = List.empty_list(types.int64)
    c2 = List.empty_list(types.int64)
    c3 = List.empty_list(types.int64)
    a0 = List.empty_list(types.int64)
    a1 = List.empty_list(types.int64)
    a2 = List.empty_list(types.int64)
    a3 = List.empty_list(types.int64)
    vals = List.empty_list(types.float64)

    for idx in range(n_list):
        n_equiv = n_equivalent[idx]
        n_indep = n_independent_basis[idx]
        for eq in range(n_equiv):
            e0, e1, e2, e3 = (
                equiv_list[0, eq, idx],
                equiv_list[1, eq, idx],
                equiv_list[2, eq, idx],
                equiv_list[3, eq, idx],
            )
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        for o in range(3):
                            basis = ((l * 3 + m) * 3 + n) * 3 + o
                            for ind in range(n_indep):
                                phi_idx = accumulated_independent[idx] + ind
                                val = (
                                    transformation_array[basis, ind, eq, idx]
                                    * phi_values[phi_idx]
                                )
                                if val == 0.0:
                                    continue
                                c0.append(l)
                                c1.append(m)
                                c2.append(n)
                                c3.append(o)
                                a0.append(e0)
                                a1.append(e1)
                                a2.append(e2)
                                a3.append(e3)
                                vals.append(val)

    return c0, c1, c2, c3, a0, a1, a2, a3, vals


# =============================================================================
# QuartetWedge class
# =============================================================================


class QuartetWedge:
    """Symmetry-based reduction of fourth-order interatomic force constants."""

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
        self.primitive_dict = primitive_dict
        self.supercell_dict = supercell_dict
        self.symmetry_ops = symmetry_ops
        self.min_distances = min_distances
        self.n_equivalent_images = n_equivalent
        self.cell_shifts = cell_shifts
        self.force_range = force_range

        self._alloc_size = 0
        self._all_alloc_size = 0
        self._expand_arrays()
        self._expand_all_list()
        self._reduce()

    def _expand_arrays(self) -> None:
        if self._alloc_size == 0:
            self._alloc_size = 32
            max_equiv = 24 * self.symmetry_ops.nsyms
            self.nequi = np.empty(self._alloc_size, dtype=np.int64)
            self.allequilist = np.empty(
                (4, max_equiv, self._alloc_size), dtype=np.int64
            )
            self.transformationarray = np.empty(
                (81, 81, max_equiv, self._alloc_size), dtype=np.float64
            )
            self._transformation = np.empty(
                (81, 81, max_equiv, self._alloc_size), dtype=np.float64
            )
            self._transformation_aux = np.empty(
                (81, 81, self._alloc_size), dtype=np.float64
            )
            self.nindependentbasis = np.empty(self._alloc_size, dtype=np.int64)
            self.independentbasis = np.empty((81, self._alloc_size), dtype=np.int64)
            self.llist = np.empty((4, self._alloc_size), dtype=np.int64)
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
        if self._all_alloc_size == 0:
            self._all_alloc_size = 512
            self._all_list = np.empty((4, self._all_alloc_size), dtype=np.int64)
        else:
            self._all_alloc_size *= 2
            self._all_list = np.concatenate((self._all_list, self._all_list), axis=-1)

    def _reduce(self) -> None:
        force_range_sq = self.force_range * self.force_range
        grid = np.array(
            [
                self.supercell_dict["na"],
                self.supercell_dict["nb"],
                self.supercell_dict["nc"],
            ],
            dtype=np.int64,
        )
        n_sym = self.symmetry_ops.nsyms
        n_atoms = len(self.primitive_dict["types"])
        n_total = len(self.supercell_dict["types"])
        lattice = self.supercell_dict["lattvec"]
        cart_coords = np.dot(lattice, self.supercell_dict["positions"])
        rotation_tensors = np.transpose(self.symmetry_ops.crotations, (1, 2, 0))

        self.nlist = 0
        self._n_all_list = 0

        shifts_27 = np.array(
            [
                [i, j, k]
                for i in range(-1, 2)
                for j in range(-1, 2)
                for k in range(-1, 2)
            ],
            dtype=np.int64,
        )
        equiv_atom_map = self.symmetry_ops.map_supercell(self.supercell_dict)
        cell_indices, species_indices = _index_to_cell_atom(grid, n_atoms)
        rotation_matrices, rotation_identity_diff, has_nonzero = (
            self._build_rotation_matrices(n_sym, rotation_tensors)
        )

        quartet = np.empty(4, dtype=np.int64)
        quartet_permuted = np.empty(4, dtype=np.int64)
        quartet_symmetric = np.empty(4, dtype=np.int64)
        equiv_quartets = np.empty((4, n_sym * 24), dtype=np.int64)
        coeff_matrix = np.empty((24 * n_sym * 81, 81), dtype=np.float64)
        shift_j = np.empty((3, 27), dtype=np.int64)
        shift_k = np.empty((3, 27), dtype=np.int64)
        shift_l = np.empty((3, 27), dtype=np.int64)

        with Progress() as progress:
            task = progress.add_task("Scanning atom quartets", total=n_atoms)
            for atom_i in range(n_atoms):
                for atom_j in range(n_total):
                    if self.min_distances[atom_i, atom_j] >= self.force_range:
                        continue
                    n_eq_j = self.n_equivalent_images[atom_i, atom_j]
                    for x in range(n_eq_j):
                        shift_j[:, x] = shifts_27[self.cell_shifts[atom_i, atom_j, x]]

                    for atom_k in range(n_total):
                        if self.min_distances[atom_i, atom_k] >= self.force_range:
                            continue
                        n_eq_k = self.n_equivalent_images[atom_i, atom_k]
                        for x in range(n_eq_k):
                            shift_k[:, x] = shifts_27[
                                self.cell_shifts[atom_i, atom_k, x]
                            ]

                        for atom_l in range(n_total):
                            if self.min_distances[atom_i, atom_l] >= self.force_range:
                                continue
                            n_eq_l = self.n_equivalent_images[atom_i, atom_l]
                            for x in range(n_eq_l):
                                shift_l[:, x] = shifts_27[
                                    self.cell_shifts[atom_i, atom_l, x]
                                ]

                            # Check pairwise distances
                            min_jk_sq = min_jl_sq = min_kl_sq = np.inf
                            for ij in range(n_eq_j):
                                cj = (
                                    shift_j[:, ij : ij + 1].T @ lattice.T
                                ).flatten() + cart_coords[:, atom_j]
                                for ik in range(n_eq_k):
                                    ck = (
                                        shift_k[:, ik : ik + 1].T @ lattice.T
                                    ).flatten() + cart_coords[:, atom_k]
                                    d_jk = np.sum((ck - cj) ** 2)
                                    if d_jk < min_jk_sq:
                                        min_jk_sq = d_jk
                                for il in range(n_eq_l):
                                    cl = (
                                        shift_l[:, il : il + 1].T @ lattice.T
                                    ).flatten() + cart_coords[:, atom_l]
                                    d_jl = np.sum((cl - cj) ** 2)
                                    if d_jl < min_jl_sq:
                                        min_jl_sq = d_jl

                            if (
                                min_jk_sq >= force_range_sq
                                or min_jl_sq >= force_range_sq
                            ):
                                continue

                            quartet[:] = [atom_i, atom_j, atom_k, atom_l]
                            if _quartet_in_list(
                                quartet, self._all_list, self._n_all_list
                            ):
                                continue

                            self.nlist += 1
                            if self.nlist == self._alloc_size:
                                self._expand_arrays()

                            idx = self.nlist - 1
                            self.llist[:, idx] = quartet
                            self.nequi[idx] = 0
                            coeff_matrix[:] = 0.0
                            n_nonzero = 0

                            for perm_idx in range(24):
                                perm = QUARTET_PERMUTATIONS[perm_idx]
                                quartet_permuted[:] = quartet[perm]

                                for sym_idx in range(n_sym):
                                    for d in range(4):
                                        quartet_symmetric[d] = equiv_atom_map[
                                            sym_idx, quartet_permuted[d]
                                        ]

                                    vec1 = cell_indices[:, quartet_symmetric[0]].copy()
                                    if not (
                                        vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0
                                    ):
                                        vec2 = (
                                            cell_indices[:, quartet_symmetric[1]] - vec1
                                        ) % grid
                                        vec3 = (
                                            cell_indices[:, quartet_symmetric[2]] - vec1
                                        ) % grid
                                        vec4 = (
                                            cell_indices[:, quartet_symmetric[3]] - vec1
                                        ) % grid
                                        vec1[:] = 0
                                        quartet_symmetric[0] = _cell_atom_to_index(
                                            vec1,
                                            species_indices[quartet_symmetric[0]],
                                            grid,
                                            n_atoms,
                                        )
                                        quartet_symmetric[1] = _cell_atom_to_index(
                                            vec2,
                                            species_indices[quartet_symmetric[1]],
                                            grid,
                                            n_atoms,
                                        )
                                        quartet_symmetric[2] = _cell_atom_to_index(
                                            vec3,
                                            species_indices[quartet_symmetric[2]],
                                            grid,
                                            n_atoms,
                                        )
                                        quartet_symmetric[3] = _cell_atom_to_index(
                                            vec4,
                                            species_indices[quartet_symmetric[3]],
                                            grid,
                                            n_atoms,
                                        )

                                    if (perm_idx == 0 and sym_idx == 0) or not (
                                        _quartets_equal(quartet_symmetric, quartet)
                                        or _quartet_in_list(
                                            quartet_symmetric,
                                            equiv_quartets,
                                            self.nequi[idx],
                                        )
                                    ):
                                        eq = self.nequi[idx]
                                        self.nequi[idx] += 1
                                        equiv_quartets[:, eq] = quartet_symmetric
                                        self.allequilist[:, eq, idx] = quartet_symmetric
                                        self._n_all_list += 1
                                        if self._n_all_list == self._all_alloc_size:
                                            self._expand_all_list()
                                        self._all_list[:, self._n_all_list - 1] = (
                                            quartet_symmetric
                                        )
                                        self._transformation[:, :, eq, idx] = (
                                            rotation_matrices[perm_idx, sym_idx]
                                        )

                                    if _quartets_equal(quartet_symmetric, quartet):
                                        for basis_idx in range(81):
                                            if has_nonzero[
                                                perm_idx, sym_idx, basis_idx
                                            ]:
                                                coeff_matrix[n_nonzero, :] = (
                                                    rotation_identity_diff[
                                                        perm_idx, sym_idx, basis_idx, :
                                                    ]
                                                )
                                                n_nonzero += 1

                            coeff_reduced = np.ascontiguousarray(
                                np.zeros((max(n_nonzero, 81), 81), dtype=np.float64)
                            )
                            coeff_reduced[:n_nonzero, :] = coeff_matrix[:n_nonzero, :]
                            basis_transform, independent = gaussian_elimination(
                                coeff_reduced
                            )
                            self._transformation_aux[:, :, idx] = basis_transform
                            self.nindependentbasis[idx] = len(independent)
                            self.independentbasis[: len(independent), idx] = independent

                progress.update(task, advance=1)

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
        self, n_sym: int, rotation_tensors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rot = np.empty((24, n_sym, 81, 81), dtype=np.float64)
        for perm_idx in range(24):
            perm = QUARTET_PERMUTATIONS[perm_idx]
            for sym_idx in range(n_sym):
                for i_p in range(3):
                    for j_p in range(3):
                        for k_p in range(3):
                            for l_p in range(3):
                                idx_p = ((i_p * 3 + j_p) * 3 + k_p) * 3 + l_p
                                for i in range(3):
                                    for j in range(3):
                                        for k in range(3):
                                            for l in range(3):
                                                idx = i * 27 + j * 9 + k * 3 + l
                                                basis = np.array([i, j, k, l])
                                                rot[perm_idx, sym_idx, idx_p, idx] = (
                                                    rotation_tensors[
                                                        i_p, basis[perm[0]], sym_idx
                                                    ]
                                                    * rotation_tensors[
                                                        j_p, basis[perm[1]], sym_idx
                                                    ]
                                                    * rotation_tensors[
                                                        k_p, basis[perm[2]], sym_idx
                                                    ]
                                                    * rotation_tensors[
                                                        l_p, basis[perm[3]], sym_idx
                                                    ]
                                                )

        rot_diff = rot.copy()
        has_nonzero = np.zeros((24, n_sym, 81), dtype=np.int64)
        for perm_idx in range(24):
            for sym_idx in range(n_sym):
                for idx_p in range(81):
                    rot_diff[perm_idx, sym_idx, idx_p, idx_p] -= 1.0
                    for idx in range(81):
                        if fabs(rot_diff[perm_idx, sym_idx, idx_p, idx]) > 1e-12:
                            has_nonzero[perm_idx, sym_idx, idx_p] = 1
                        else:
                            rot_diff[perm_idx, sym_idx, idx_p, idx] = 0.0
        return rot, rot_diff, has_nonzero

    def build_quartet_list(self) -> list:
        extended = []
        for idx in range(self.nlist):
            for b in range(self.nindependentbasis[idx]):
                val = self.independentbasis[b, idx]
                c0, c1, c2, c3 = val // 27, (val % 27) // 9, (val % 9) // 3, val % 3
                extended.append(
                    (
                        c0,
                        self.llist[0, idx],
                        c1,
                        self.llist[1, idx],
                        c2,
                        self.llist[2, idx],
                        c3,
                        self.llist[3, idx],
                    )
                )
        unique = []
        for item in extended:
            six_tuple = (item[1], item[3], item[5], item[0], item[2], item[4])
            if six_tuple not in unique:
                unique.append(six_tuple)
        return unique

    build_list4 = build_quartet_list


# =============================================================================
# Main reconstruction function
# =============================================================================


def reconstruct_ifcs(
    phi_partial: np.ndarray,
    wedge: QuartetWedge,
    quartet_list: list,
    primitive_dict: dict,
    supercell_dict: dict,
) -> sparse.COO:
    """Reconstruct full fourth-order IFC tensor from irreducible set."""
    n_list = wedge.nlist
    n_atoms = len(primitive_dict["types"])
    n_total = len(supercell_dict["types"])

    typer.echo("Using sparse method with DOK sparse matrix")
    result_sparse = sparse.zeros(
        (3, 3, 3, 3, n_atoms, n_total, n_total, n_total), format="dok"
    )
    accumulated = np.insert(
        np.cumsum(wedge.nindependentbasis[:n_list], dtype=np.int64), 0, 0
    )
    n_total_indep = accumulated[-1]

    with Progress() as progress:
        task = progress.add_task("Storing partial results", total=len(quartet_list))
        for idx, (e0, e1, e2, e3, e4, e5) in enumerate(quartet_list):
            result_sparse[e3, e4, e5, :, e0, e1, e2, :] = phi_partial[:, idx, :]
            progress.update(task, advance=1)
    result_sparse = result_sparse.to_coo()

    phi_list = []
    with Progress() as progress:
        task = progress.add_task("Extracting phi values", total=n_list)
        for idx in range(n_list):
            for b in range(wedge.nindependentbasis[idx]):
                val = wedge.independentbasis[b, idx]
                c0, c1, c2, c3 = val // 27, (val % 27) // 9, (val % 9) // 3, val % 3
                phi_list.append(
                    result_sparse[
                        c0,
                        c1,
                        c2,
                        c3,
                        wedge.llist[0, idx],
                        wedge.llist[1, idx],
                        wedge.llist[2, idx],
                        wedge.llist[3, idx],
                    ]
                )
            progress.update(task, advance=1)
    phi_values = np.array(phi_list, dtype=np.float64)

    quartet_class_idx = -np.ones((n_atoms, n_total, n_total, n_total), dtype=np.int64)
    equiv_class_idx = -np.ones((n_atoms, n_total, n_total, n_total), dtype=np.int64)
    equiv_list = wedge.allequilist
    for idx in range(n_list):
        for eq in range(wedge.nequi[idx]):
            i, j, k, l = (
                equiv_list[0, eq, idx],
                equiv_list[1, eq, idx],
                equiv_list[2, eq, idx],
                equiv_list[3, eq, idx],
            )
            quartet_class_idx[i, j, k, l] = idx
            equiv_class_idx[i, j, k, l] = eq

    typer.echo("Building sparse coefficient matrix")
    row, col, val = _build_sparse_quartets(
        quartet_class_idx,
        equiv_class_idx,
        accumulated,
        n_atoms,
        n_total,
        n_list,
        wedge.transformationarray,
    )
    coeff = sp.sparse.coo_matrix(
        (val, (row, col)), (n_total_indep, n_atoms * n_total * 81)
    ).tocsr()

    diag = sp.sparse.spdiags(
        phi_values, [0], phi_values.size, phi_values.size, format="csr"
    )
    weighted = diag.dot(coeff)
    mult = -sp.sparse.linalg.lsqr(weighted, np.ones_like(phi_values))[0]
    phi_values += diag.dot(weighted.dot(mult))

    typer.echo("Building final full IFC tensor")
    coords = _build_full_ifc_coordinates(
        n_list,
        wedge.nequi,
        wedge.nindependentbasis,
        equiv_list,
        accumulated,
        wedge.transformationarray,
        phi_values,
    )

    final_coords = np.array(
        [np.array(coords[i], dtype=np.intp) for i in range(8)], dtype=np.intp
    )
    return sparse.COO(
        final_coords,
        np.array(coords[8], dtype=np.float64),
        shape=(3, 3, 3, 3, n_atoms, n_total, n_total, n_total),
        has_duplicates=True,
    )
