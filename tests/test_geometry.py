import numpy as np
from ase.build import bulk

from mlfcs.core.geometry import _unique_distances, make_supercell, resolve_cutoff
from mlfcs.core.symmetry import SymmetryOperations


def test_supercell_index_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    assert len(supercell) == 16
    for atom in range(len(supercell)):
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom


def test_negative_cutoff_and_symmetry_mapping():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    assert resolve_cutoff(supercell, index, -2) > 0
    operations = SymmetryOperations.from_atoms(primitive, supercell)
    assert operations.size > 1
    for permutation in operations.atom_permutations:
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(supercell)))


def test_neighbor_shells_merge_small_relaxation_splittings():
    shells = _unique_distances(np.array([0.0, 2.9997637, 2.9997681, 3.66109]))
    assert shells == [2.9997637, 3.66109]
