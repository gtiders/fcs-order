"""Workflow builders for HIFINIT computations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.force_constant_model import ForceConstantModel
from hiphive.force_constants import SortedForceConstants
from hiphive.utilities import extract_parameters
from phonopy import Phonopy


def prepare_simulation(
    primitive: Union[str, Atoms],
    calculator,
    supercell: Optional[Union[str, Atoms]] = None,
    supercell_matrix: Optional[np.ndarray] = None,
    displacement: float = 0.01,
    cutoffs: Optional[List[Optional[float]]] = None,
    verbose: bool = True,
) -> dict:
    """Prepare simulation environment and model objects."""
    if isinstance(primitive, Atoms):
        prim = primitive.copy()
    else:
        prim = read(primitive)
    prim.calc = calculator

    if supercell is not None:
        if isinstance(supercell, Atoms):
            supercell = supercell.copy()
        else:
            supercell = read(supercell)
    elif supercell_matrix is not None:
        phonopy = Phonopy(prim, supercell_matrix)
        supercell = Atoms(
            cell=phonopy.supercell.cell,
            numbers=phonopy.supercell.numbers,
            positions=phonopy.supercell.positions,
        )
    else:
        raise ValueError("Either supercell or supercell_matrix must be provided")

    supercell.calc = calculator

    if verbose:
        print("=" * 60)
        print("Simulation Setup")
        print("=" * 60)
        print(f"Primitive cell atoms: {len(prim)}")
        print(f"Supercell atoms: {len(supercell)}")

    max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
    if verbose:
        print(f"Maximum cutoff: {max_cutoff:.4f} Å")

    if cutoffs is None:
        cutoffs = [max_cutoff]
    else:
        cutoffs = [c if c is not None else max_cutoff for c in cutoffs]

    cs = ClusterSpace(prim, cutoffs, acoustic_sum_rules=True)
    if verbose:
        print("\n" + "-" * 60)
        print("ClusterSpace Information")
        print("-" * 60)
        print(f"Independent parameters (n_dofs): {cs.n_dofs}")
        print(f"Total orbits: {len(cs.orbits)}")
        for order in cs.cutoffs.orders:
            n_orbits = sum(1 for orb in cs.orbits if orb.order == order)
            n_params = sum(
                len(orb.eigentensors) for orb in cs.orbits if orb.order == order
            )
            print(f"  Order {order}: {n_orbits} orbits, {n_params} parameters")

    fcm = ForceConstantModel(supercell, cs)
    if verbose:
        print("\n" + "-" * 60)
        print("ForceConstantModel Information")
        print("-" * 60)
        print(f"Total clusters: {len(fcm.cluster_list)}")
        print(f"Total orbits: {len(fcm.orbits)}")

    atoms_with_calc = supercell.copy()
    atoms_with_calc.calc = calculator
    pos_ideal = atoms_with_calc.get_positions().copy()
    max_order = max(cs.cutoffs.orders)

    return {
        "prim": prim,
        "supercell": supercell,
        "cs": cs,
        "fcm": fcm,
        "atoms_with_calc": atoms_with_calc,
        "pos_ideal": pos_ideal,
        "displacement": displacement,
        "max_order": max_order,
    }


def generate_force_constants(
    fc_dict: Dict[Tuple[int, ...], np.ndarray],
    supercell: Atoms,
    cs: ClusterSpace,
    verbose: bool = True,
) -> Tuple[ForceConstantPotential, SortedForceConstants]:
    """Generate ForceConstantPotential and complete force constants."""
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Force Constants Object")
        print("=" * 60)

    fcs = SortedForceConstants(fc_dict, supercell)
    if verbose:
        print(f"Number of force constant clusters: {len(fcs)}")
        print(f"Force constant orders: {fcs.orders}")
        print("\nExtracting parameters...")

    parameters = extract_parameters(fcs, cs, sanity_check=False, lstsq_method="scipy")

    if verbose:
        print("Creating ForceConstantPotential...")
    fcp = ForceConstantPotential(cs, parameters)

    if verbose:
        print("Generating complete force constants...")
    fcs_final = fcp.get_force_constants(supercell)

    frequencies = fcs_final.compute_gamma_frequencies()
    if verbose:
        print(f"\nGamma point frequencies (first 5, THz): {frequencies[:5]}")

    return fcp, fcs_final
