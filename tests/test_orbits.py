from ase.build import bulk

from mlfcs.geometry import make_supercell, resolve_cutoff
from mlfcs.orbits import build_orbit_space, tensor_action_matrix
from mlfcs.symmetry import SymmetryOperations


def test_identity_tensor_action():
    import numpy as np

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
