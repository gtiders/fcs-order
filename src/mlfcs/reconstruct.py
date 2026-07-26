from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from mlfcs.geometry import SupercellIndex
from mlfcs.orbits import OrbitSpace
from mlfcs.sampling import DisplacementKey


def reconstruct_compact(
    orbit_space: OrbitSpace,
    index: SupercellIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
) -> np.ndarray:
    """Reconstruct a compact, translation-reduced IFC tensor orbit by orbit."""
    order = orbit_space.order
    atom_shape = (index.n_primitive,) + (len(index.primitive),) * (order - 1)
    shape = atom_shape + (3,) * order
    result = np.zeros(shape, dtype=float)
    counts = np.zeros(atom_shape, dtype=np.int16)

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
        pivot_values = _project_acoustic_sum_rule(orbit_space, pivot_values)
    for orbit, values in zip(orbit_space.orbits, pivot_values, strict=True):
        pivot_basis = orbit.basis[orbit.pivots]
        coefficients = np.linalg.solve(pivot_basis, values)
        representative = orbit.basis @ coefficients
        for image in orbit.images:
            tensor = (image.transform @ representative).reshape((3,) * order)
            result[image.cluster] += tensor
            counts[image.cluster] += 1

    nonzero = counts > 0
    result[nonzero] /= counts[nonzero].reshape((-1,) + (1,) * order)
    return result


def _project_acoustic_sum_rule(
    orbit_space: OrbitSpace,
    pivot_values: list[np.ndarray],
    tolerance: float = 1e-12,
) -> list[np.ndarray]:
    """Apply the reference relative-weight ASR projection in a small Gram space."""
    offsets = np.cumsum([0] + [len(values) for values in pivot_values])
    n_parameters = int(offsets[-1])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []

    for orbit_index, orbit in enumerate(orbit_space.orbits):
        inverse_pivots = np.linalg.inv(orbit.basis[orbit.pivots])
        representative_from_pivots = orbit.basis @ inverse_pivots
        for image in orbit.images:
            image_from_pivots = image.transform @ representative_from_pivots
            for component in range(3**orbit_space.order):
                directions = np.unravel_index(component, (3,) * orbit_space.order)
                # ASR sums the final atom while retaining every Cartesian index.
                equation_key = image.cluster[:-1] + tuple(int(x) for x in directions)
                equation = equations.setdefault(equation_key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_from_pivots[component]) > tolerance)
                for local_parameter in nonzero:
                    rows.append(equation)
                    columns.append(int(offsets[orbit_index] + local_parameter))
                    data.append(float(image_from_pivots[component, local_parameter]))

    constraints = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), n_parameters),
    ).tocsr()
    parameters = np.concatenate(pivot_values)
    weighted_constraints = sparse.diags(parameters) @ constraints.T
    multiplier = -lsqr(weighted_constraints, np.ones(n_parameters))[0]
    correction = parameters * (weighted_constraints @ multiplier)
    projected = parameters + correction
    return [projected[begin:end] for begin, end in pairwise(offsets)]
