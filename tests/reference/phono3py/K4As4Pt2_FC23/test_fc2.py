from __future__ import annotations

import numpy as np
import pytest

from tests.reference.phono3py.K4As4Pt2_FC23.case import (
    assert_matches_phono3py,
    calculation_and_reference,
)


@pytest.mark.reference
def test_K4As4Pt2_FC2_matches_phono3py_traditional_solver():
    data, calculation, sparse = calculation_and_reference(2)
    assert (
        calculation.plan.hash == "a7ac802fa62a1701f89cd8d215b109f456c4112204e0625ca5f6ece420e0af8c"
    )
    assert len(calculation.plan) == 24
    assert str(data["cutoff_mode"]) == "maximum_supercell_mic"
    assert np.isclose(calculation.cutoff, 12.646150266897997, atol=1e-12, rtol=0)
    assert np.array_equal(sparse.clusters, data["fc2_clusters"])
    assert_matches_phono3py(sparse.tensors, data["phono3py_raw_fc2_tensors"], order=2)


@pytest.mark.reference
def test_K4As4Pt2_FC2_ASR_matches_symfc_full_space_projection():
    data, calculation, _ = calculation_and_reference(2)
    result = calculation.reap(
        data["fc2_forces"],
        plan_hash=str(data["fc2_plan_hash"]),
        acoustic_sum_rule=True,
    )
    actual = result.sparse[2]
    dense = actual.to_dense(max_bytes=None)
    assert np.max(np.abs(np.sum(dense, axis=1))) < 1e-10

    expected = data["symfc_fc2_tensors"]
    difference = actual.tensors - expected
    assert np.max(np.abs(difference)) < 2.1e-3
    assert np.sqrt(np.mean(difference**2)) < 6e-5
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 1.3e-4
    assert np.corrcoef(actual.tensors.ravel(), expected.ravel())[0, 1] > 0.9999999
