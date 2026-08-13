import numpy as np
import pytest
from ase import Atoms

phonopy = pytest.importorskip("phonopy")
from phonopy.structure.atoms import PhonopyAtoms

from mlfcs.core.geometry import make_supercell
from mlfcs.sscha.ensemble import HarmonicEnsemble

pytestmark = pytest.mark.reference


def test_qspace_frequencies_and_quantum_covariance_match_phonopy():
    primitive = Atoms("Al", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    supercell, _ = make_supercell(primitive, (2, 2, 2))
    n_atoms = len(supercell)
    spring = 2.0
    full = np.zeros((n_atoms, n_atoms, 3, 3))
    for axis in range(3):
        full[:, :, axis, axis] = spring * (np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / n_atoms)

    actual = HarmonicEnsemble(primitive, supercell, full[:1], temperature=300)
    ph = phonopy.Phonopy(
        PhonopyAtoms(
            symbols=primitive.get_chemical_symbols(),
            cell=primitive.cell,
            scaled_positions=primitive.get_scaled_positions(),
        ),
        supercell_matrix=np.diag([2, 2, 2]),
    )
    ph.force_constants = full
    ph.init_random_displacements()
    reference = ph.random_displacements
    assert reference is not None

    actual_frequencies = np.sort(np.concatenate(actual.frequencies))
    reference_frequencies = np.sort(reference.frequencies[np.abs(reference.frequencies) > 0.01])
    np.testing.assert_allclose(actual_frequencies, reference_frequencies, rtol=2e-7)

    reference.run_correlation_matrix(300)
    assert reference.uu is not None
    displacement = actual.sample(100_000, random_seed=19)
    empirical_variance = np.var(displacement[:, 0], axis=0)
    np.testing.assert_allclose(empirical_variance, np.diag(reference.uu[0, 0]), rtol=0.02)
