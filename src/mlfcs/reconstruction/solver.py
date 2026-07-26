from __future__ import annotations

import numpy as np

from mlfcs.core.geometry import SupercellIndex
from mlfcs.core.orbits import OrbitSpace
from mlfcs.finite_difference.sampling import DisplacementKey
from mlfcs.model import SparseOrderForceConstants
from mlfcs.reconstruction.asr import project_acoustic_sum_rule


def reconstruct_compact(
    orbit_space: OrbitSpace,
    index: SupercellIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
) -> np.ndarray:
    """Reconstruct a compact, translation-reduced IFC tensor orbit by orbit."""
    sparse_result = reconstruct_sparse(
        orbit_space,
        index,
        derivatives,
        enforce_asr=enforce_asr,
    )
    return sparse_result.to_dense(max_bytes=None)


def reconstruct_sparse(
    orbit_space: OrbitSpace,
    index: SupercellIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
) -> SparseOrderForceConstants:
    """Reconstruct only symmetry-generated cluster tensors."""
    order = orbit_space.order
    pivot_values: list[np.ndarray] = []
    for orbit in orbit_space.orbits:
        values: list[float] = []
        for pivot in orbit.pivots:
            components = np.unravel_index(int(pivot), (3,) * order)
            key = tuple(
                (orbit.representative[axis], int(components[axis])) for axis in range(order - 1)
            )
            values.append(derivatives[key][orbit.representative[-1], int(components[-1])])
        pivot_values.append(np.asarray(values))

    if enforce_asr:
        pivot_values = project_acoustic_sum_rule(orbit_space, pivot_values)
    clusters: list[tuple[int, ...]] = []
    tensors: list[np.ndarray] = []
    for orbit, values in zip(orbit_space.orbits, pivot_values, strict=True):
        pivot_basis = orbit.basis[orbit.pivots]
        coefficients = np.linalg.solve(pivot_basis, values)
        representative = orbit.basis @ coefficients
        for image in orbit.images:
            tensor = image.action.apply_flat(representative).reshape((3,) * order)
            clusters.append(image.cluster)
            tensors.append(tensor)
    return SparseOrderForceConstants(
        order=order,
        n_primitive=index.n_primitive,
        n_supercell=len(index.primitive),
        clusters=np.asarray(clusters, dtype=np.int32).reshape((-1, order)),
        tensors=np.asarray(tensors, dtype=float).reshape((-1,) + (3,) * order),
    )
