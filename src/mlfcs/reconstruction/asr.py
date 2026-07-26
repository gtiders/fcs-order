from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.core.orbits import OrbitSpace


def build_translational_constraints(
    orbit_space: OrbitSpace,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the order-local acoustic sum-rule matrix ``A``.

    Each row fixes the first ``n - 1`` atom indices and every Cartesian
    component, and sums only the final atom index. Permutation symmetry in
    the orbit space makes the equivalent constraints on other axes redundant.
    """
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []

    for orbit_index, orbit in enumerate(orbit_space.orbits):
        representative_from_pivots = orbit.basis @ np.linalg.inv(orbit.basis[orbit.pivots])
        for image in orbit.images:
            image_from_pivots = image.action.apply_columns(representative_from_pivots)
            for component in range(3**orbit_space.order):
                directions = np.unravel_index(component, (3,) * orbit_space.order)
                equation_key = image.cluster[:-1] + tuple(int(x) for x in directions)
                equation = equations.setdefault(equation_key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_from_pivots[component]) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_from_pivots[component, value]) for value in nonzero)

    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()


def project_acoustic_sum_rule(
    orbit_space: OrbitSpace,
    pivot_values: list[np.ndarray],
    *,
    tolerance: float = 1e-11,
) -> list[np.ndarray]:
    """Orthogonally project independent IFC parameters onto ``null(A)``."""
    offsets = np.cumsum([0] + [len(values) for values in pivot_values])
    parameters = np.concatenate(pivot_values)
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return pivot_values

    residual = constraints @ parameters
    scale = max(float(np.linalg.norm(parameters)), 1.0)
    if float(np.linalg.norm(residual, ord=np.inf)) > tolerance * scale:
        correction = lsmr(
            constraints,
            residual,
            atol=tolerance * 0.1,
            btol=tolerance * 0.1,
            maxiter=max(1000, 4 * constraints.shape[1]),
        )[0]
        parameters = parameters - correction

    final_residual = constraints @ parameters
    if float(np.linalg.norm(final_residual, ord=np.inf)) > tolerance * scale:
        raise RuntimeError(
            "acoustic sum-rule projection did not converge: "
            f"max residual={np.linalg.norm(final_residual, ord=np.inf):.3e}"
        )
    return [parameters[begin:end] for begin, end in pairwise(offsets)]
