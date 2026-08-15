from __future__ import annotations

import numpy as np
import pytest

from mlfcs.io.export import build_export_view
from mlfcs.io.hdf5 import read_hdf5
from tests.reference.phono3py.K4As4Pt2_FC23.case import calculation_and_reference


@pytest.mark.reference
def test_K4As4Pt2_native_export_accepts_reordered_primitive_and_supercell(tmp_path):
    """A multicomponent physical IFC survives an independent target relabelling."""
    data, calculation, _ = calculation_and_reference(2)
    result = calculation.reap(data["fc2_forces"], acoustic_sum_rule=False)

    primitive_order = np.roll(np.arange(len(result.relation.primitive)), 3)
    supercell_order = np.random.default_rng(19).permutation(len(result.supercell))
    target_primitive = result.relation.primitive[primitive_order]
    target_supercell = result.supercell[supercell_order]
    target = tmp_path / "K4As4Pt2-reordered.hdf5"

    result.write(
        target,
        format="hdf5",
        primitive=target_primitive,
        supercell=target_supercell,
    )
    restored = read_hdf5(target)

    np.testing.assert_array_equal(restored.relation.primitive.numbers, target_primitive.numbers)
    np.testing.assert_array_equal(restored.supercell.numbers, target_supercell.numbers)
    source = result.sparse[2]
    actual = build_export_view(
        restored,
        primitive=result.relation.primitive,
        supercell=result.supercell,
    ).force_constants.sparse[2]
    np.testing.assert_array_equal(actual.sites, source.sites)
    np.testing.assert_array_equal(
        actual.translation_representatives, source.translation_representatives
    )
    np.testing.assert_allclose(actual.tensors, source.tensors)
