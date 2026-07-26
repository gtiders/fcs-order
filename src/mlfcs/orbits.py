from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms

from mlfcs.geometry import SupercellIndex
from mlfcs.symmetry import SymmetryOperations

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True, slots=True)
class OrbitImage:
    cluster: tuple[int, ...]
    transform: np.ndarray


@dataclass(frozen=True, slots=True)
class ClusterOrbit:
    representative: tuple[int, ...]
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[OrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class OrbitSpace:
    order: int
    orbits: tuple[ClusterOrbit, ...]
    cutoff: float

    @property
    def displacement_keys(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for flat_component in orbit.pivots:
                components = np.unravel_index(int(flat_component), (3,) * self.order)
                key = tuple(
                    (orbit.representative[axis], int(components[axis]))
                    for axis in range(self.order - 1)
                )
                keys.add(key)
        return tuple(sorted(keys))


def build_orbit_space(
    supercell: Atoms,
    index: SupercellIndex,
    symmetry: SymmetryOperations,
    *,
    order: int,
    cutoff: float,
    tolerance: float = 1e-9,
) -> OrbitSpace:
    """Enumerate cutoff clusters and reduce them by permutation and space group."""
    if order < 2:
        raise ValueError("order must be at least two")
    distances = supercell.get_all_distances(mic=True)
    neighbors = [
        np.flatnonzero(distances[atom] < cutoff).tolist() for atom in range(index.n_primitive)
    ]
    axis_permutations = tuple(permutations(range(order)))
    # Cache only rotations. Caching rotation x permutation at fourth order
    # retains 48 * 24 * 81^2 float64 values (~58 MiB) before any orbit exists.
    identity_permutation = tuple(range(order))
    rotation_actions = tuple(
        tensor_action_matrix(rotation.T, identity_permutation, order)
        for rotation in symmetry.cartesian_rotations
    )
    seen: set[tuple[int, ...]] = set()
    orbits: list[ClusterOrbit] = []

    for first in range(index.n_primitive):
        for tail in product(neighbors[first], repeat=order - 1):
            cluster = (first, *tail)
            if not _inside_cluster_cutoff(cluster, distances, cutoff):
                continue
            images = _orbit_images(cluster, index, symmetry, axis_permutations, order)
            if cluster in seen:
                continue
            seen.update(images)
            representative = cluster
            action_by_image: dict[tuple[int, ...], np.ndarray] = {}
            identity = np.eye(3**order)
            constraint_gram = np.zeros_like(identity)
            for operation in range(symmetry.size):
                transformed = tuple(
                    int(symmetry.atom_permutations[operation, atom]) for atom in representative
                )
                for axis_permutation in axis_permutations:
                    candidate = index.anchor(tuple(transformed[axis] for axis in axis_permutation))
                    action = permute_tensor_action(
                        rotation_actions[operation], axis_permutation, order
                    )
                    # Two mappings to one image differ by a representative
                    # stabilizer, so either acts identically on the invariant basis.
                    action_by_image.setdefault(candidate, action)
                    if candidate == representative:
                        constraint = action - identity
                        constraint_gram += constraint.T @ constraint
            basis, pivots = _null_space_from_gram(constraint_gram, tolerance)
            if basis.shape[1] == 0:
                continue
            orbit_images = tuple(
                OrbitImage(key, action) for key, action in sorted(action_by_image.items())
            )
            orbits.append(ClusterOrbit(representative, basis, pivots, orbit_images))
    return OrbitSpace(order, tuple(orbits), cutoff)


def _inside_cluster_cutoff(
    cluster: tuple[int, ...],
    distances: np.ndarray,
    cutoff: float,
) -> bool:
    return all(distances[a, b] < cutoff for a, b in permutations(cluster, 2))


def _orbit_images(
    cluster: tuple[int, ...],
    index: SupercellIndex,
    symmetry: SymmetryOperations,
    axis_permutations: tuple[tuple[int, ...], ...],
    order: int,
) -> set[tuple[int, ...]]:
    del order
    images: set[tuple[int, ...]] = set()
    for operation in range(symmetry.size):
        transformed = tuple(int(symmetry.atom_permutations[operation, atom]) for atom in cluster)
        for permutation in axis_permutations:
            images.add(index.anchor(tuple(transformed[axis] for axis in permutation)))
    return images


def tensor_action_matrix(
    rotation: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Return the Cartesian tensor representation of a symmetry action."""
    size = 3**order
    tensors = jnp.eye(size, dtype=jnp.float64).reshape((size,) + (3,) * order)
    rotation_array = jnp.asarray(rotation)
    transformed = jax.vmap(lambda tensor: _rotate_tensor(tensor, rotation_array, order))(tensors)
    transformed = jnp.transpose(transformed, (0,) + tuple(axis + 1 for axis in axis_permutation))
    return np.asarray(transformed.reshape(size, size).T)


def permute_tensor_action(
    action: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Permute output axes of a flattened tensor action."""
    size = 3**order
    by_input = action.T.reshape((size,) + (3,) * order)
    axes = (0,) + tuple(axis + 1 for axis in axis_permutation)
    return np.transpose(by_input, axes).reshape(size, size).T


def _rotate_tensor(tensor: jax.Array, rotation: jax.Array, order: int) -> jax.Array:
    result = tensor
    for axis in range(order):
        result = jnp.tensordot(rotation, result, axes=((1,), (axis,)))
        result = jnp.moveaxis(result, 0, axis)
    return result


def _null_space_from_gram(gram: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Find null(C) from the small Gram matrix C.T @ C.

    The direct constraint matrix may have tens of thousands of rows for a
    fourth-order high-symmetry cluster. Its Gram matrix is always at most
    81x81, preventing LAPACK from allocating a huge left-singular-vector
    matrix.
    """
    matrix = gram.copy()
    n_rows, n_columns = matrix.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    threshold = tolerance * max(float(np.max(np.abs(matrix))), 1.0)
    for column in range(n_columns):
        if pivot_row == n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[candidate, column]) <= threshold:
            continue
        if candidate != pivot_row:
            matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(n_rows):
            if row != pivot_row and abs(matrix[row, column]) > threshold:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        matrix[np.abs(matrix) < threshold] = 0.0
        pivot_columns.append(column)
        pivot_row += 1

    independent = np.asarray(
        [column for column in range(n_columns) if column not in pivot_columns],
        dtype=np.int32,
    )
    basis = np.zeros((n_columns, len(independent)), dtype=float)
    for basis_column, free_column in enumerate(independent):
        basis[free_column, basis_column] = 1.0
        for row, dependent_column in enumerate(pivot_columns):
            basis[dependent_column, basis_column] = -matrix[row, free_column]
    basis[np.abs(basis) < tolerance] = 0.0
    return basis, independent
