from __future__ import annotations

import numpy as np
import pytest
from hiphive.input_output.shengBTE import _raw_to_fancy, _read_raw_sheng

from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.model import ForceConstants
from tests.reference.phono3py.K4As4Pt2_FC23.case import calculation_and_reference


@pytest.mark.reference
def test_K4As4Pt2_raw_FC3_shengbte_uses_joint_minimum_images(tmp_path):
    _, calculation, sparse = calculation_and_reference(3)
    result = ForceConstants(
        {},
        calculation.supercell,
        metadata={"cutoff_angstrom": calculation.cutoff, "acoustic_sum_rule": False},
        sparse={3: sparse},
    )
    target = tmp_path / "FORCE_CONSTANTS_3RD"
    result.write(target, format="shengbte")

    primitive = calculation.primitive
    entries = _raw_to_fancy(_read_raw_sheng(target), primitive.cell)
    tensors = {
        tuple(int(value) for value in cluster): tensor
        for cluster, tensor in zip(sparse.clusters, sparse.tensors, strict=True)
    }
    geometry = PeriodicGeometry(calculation.supercell.cell)
    assert entries
    for entry in entries:
        basis = primitive.positions
        positions = np.asarray(
            [
                basis[entry.site_1] - basis[entry.site_0] + entry.pos_1,
                basis[entry.site_2] - basis[entry.site_0] + entry.pos_2,
            ]
        )
        for vector in (positions[0], positions[1], positions[1] - positions[0]):
            assert np.isclose(np.linalg.norm(vector), geometry.minimum_length(vector), atol=1e-8)
        cluster = (
            calculation.index.representative(entry.site_0),
            calculation.index.atom(entry.site_1, entry.offset_1),
            calculation.index.atom(entry.site_2, entry.offset_2),
        )
        np.testing.assert_allclose(entry.fc, tensors[cluster], atol=1e-8, rtol=0.0)
