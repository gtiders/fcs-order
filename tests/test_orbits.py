import numpy as np
from ase import Atoms
from ase.build import bulk

from mlfcs.core.geometry import make_supercell, resolve_cutoff
from mlfcs.core.orbits import (
    _legacy_cluster_geometry,
    build_orbit_space,
    tensor_action_matrix,
)
from mlfcs.core.symmetry import SymmetryOperations


def test_identity_tensor_action():
    action = tensor_action_matrix(np.eye(3), (0, 1, 2), 3)
    np.testing.assert_allclose(action, np.eye(27))


def test_si_third_order_orbits_are_nonempty():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    cutoff = resolve_cutoff(supercell, index, -1)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=3,
        cutoff=cutoff,
    )
    assert space.orbits
    assert space.displacement_keys


def test_joint_periodic_images_match_legacy_cluster_cutoff():
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    supercell, _ = make_supercell(primitive, (3, 1, 1))
    distances, compatible = _legacy_cluster_geometry(supercell, 1, 1.1)

    # Every pair has an independent MIC distance of 1, but atoms 1 and 2
    # occupy opposite minimum images relative to anchor 0. They cannot form
    # one legacy-compatible periodic triangle within the 1.1 cutoff.
    np.testing.assert_allclose(distances[0], [0, 1, 1])
    assert not compatible[0, 1, 2]
