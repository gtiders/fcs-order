"""Finite-difference kernels for high-order force constants."""

from __future__ import annotations

from itertools import product as iter_product
from typing import Dict, Tuple

import numpy as np
from ase import Atoms
from hiphive.force_constant_model import ForceConstantModel


def compute_fc_finite_difference(
    atoms: Atoms,
    pos_ideal: np.ndarray,
    cluster: Tuple[int, ...],
    displacement: float,
) -> np.ndarray:
    """Compute force constants using finite difference for arbitrary order."""
    order = len(cluster)
    i = cluster[0]
    disp_atoms = cluster[1:]
    n_disp = order - 1

    fc_shape = (3,) * order
    phi_fd = np.zeros(fc_shape)

    for disp_indices in iter_product(range(3), repeat=n_disp):
        numerator = np.zeros(3)
        for signs in iter_product([1, -1], repeat=n_disp):
            pos = pos_ideal.copy()
            for atom_idx, dir_idx, sign in zip(disp_atoms, disp_indices, signs):
                pos[atom_idx, dir_idx] += sign * displacement

            atoms.set_positions(pos)
            force_all_alpha = atoms.get_forces()[i, :]
            numerator += np.prod(signs) * force_all_alpha

        phi_fd[(slice(None),) + disp_indices] = (
            -numerator / (2 * displacement) ** n_disp
        )

    return phi_fd


def compute_fcs_from_orbits(
    atoms_with_calc: Atoms,
    pos_ideal: np.ndarray,
    fcm: ForceConstantModel,
    displacement: float,
    max_order: int = 2,
    verbose: bool = True,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """Compute force constants for all orbits using finite difference."""
    if verbose:
        print("\n" + "=" * 60)
        print("Starting Finite Difference Force Constant Calculation")
        print("=" * 60)

    fc_dict = {}
    orbit_count = 0
    total_orbits = sum(1 for orb in fcm.orbits if orb.order <= max_order)

    for orbit_idx, orbit in enumerate(fcm.orbits):
        order = orbit.order

        if order > max_order:
            continue

        orbit_count += 1
        prototype_cluster_idx = orbit.prototype_index
        cluster = fcm.cluster_list[prototype_cluster_idx]

        n_eval = 3 ** (order - 1) * 2 ** (order - 1)

        if verbose:
            print(f"\n[{orbit_count}/{total_orbits}] Orbit {orbit_idx} | Order {order}")
            print(f"  Prototype cluster: {cluster}")
            print(f"  Force evaluations: {n_eval}")

        phi_fd = compute_fc_finite_difference(
            atoms_with_calc, pos_ideal, cluster, displacement
        )

        sorted_cluster = tuple(sorted(cluster))
        perm = np.argsort(cluster)
        phi_sorted = phi_fd.transpose(perm)

        fc_dict[sorted_cluster] = phi_sorted

    if verbose:
        print("\n" + "=" * 60)
        print("Finite Difference Calculation Complete!")
        print(f"Total orbits computed: {orbit_count}")
        print("=" * 60)

    return fc_dict
