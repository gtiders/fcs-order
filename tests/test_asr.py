import numpy as np
from ase.build import bulk

from mlfcs.geometry import make_supercell, resolve_cutoff
from mlfcs.orbits import build_orbit_space
from mlfcs.reconstruct import reconstruct_compact
from mlfcs.symmetry import SymmetryOperations


def test_acoustic_sum_rule_projection_reduces_residual():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=3,
        cutoff=resolve_cutoff(supercell, index, -1),
    )
    rng = np.random.default_rng(4)
    derivatives = {key: rng.normal(size=(len(supercell), 3)) for key in space.displacement_keys}
    raw = reconstruct_compact(space, index, derivatives, enforce_asr=False)
    projected = reconstruct_compact(space, index, derivatives, enforce_asr=True)
    raw_residual = np.linalg.norm(raw.sum(axis=2))
    projected_residual = np.linalg.norm(projected.sum(axis=2))
    assert projected_residual < raw_residual
