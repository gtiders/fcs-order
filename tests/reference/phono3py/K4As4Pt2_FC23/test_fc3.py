from __future__ import annotations

import numpy as np
import pytest

from tests.reference.phono3py.K4As4Pt2_FC23.case import (
    assert_matches_phono3py,
    calculation_and_reference,
)


@pytest.mark.reference
def test_K4As4Pt2_FC3_matches_phono3py_traditional_solver():
    data, calculation, sparse = calculation_and_reference(3)
    assert len(calculation.plan) == 4244
    assert str(data["cutoff_mode"]) == "maximum_supercell_mic"
    assert np.isclose(calculation.cutoff, 12.646150266897997, atol=1e-12, rtol=0)
    assert np.array_equal(sparse.clusters, data["fc3_clusters"])
    assert_matches_phono3py(sparse.tensors, data["phono3py_raw_fc3_tensors"], order=3)


@pytest.mark.reference
def test_K4As4Pt2_FC3_ASR_is_strict_and_compared_with_symfc():
    data, calculation, raw = calculation_and_reference(3)
    result = calculation.reap(
        data["fc3_forces"],
        acoustic_sum_rule=True,
    )
    actual = result.sparse[3]
    dense = actual.to_dense(max_bytes=None)
    assert np.max(np.abs(np.sum(dense, axis=1))) < 1e-10
    assert np.max(np.abs(np.sum(dense, axis=2))) < 1e-10

    correction = actual.tensors - raw.tensors
    assert np.linalg.norm(correction) / np.linalg.norm(raw.tensors) < 1.4e-4

    # symfc projects the redundant full FC3 with a different basis and metric.
    # Both results obey ASR, but they are not expected to be the same minimizer.
    expected = data["symfc_fc3_tensors"]
    difference = actual.tensors - expected
    assert np.max(np.abs(difference)) < 1.49
    assert np.sqrt(np.mean(difference**2)) < 1.0e-2
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 8.7e-2
    assert np.corrcoef(actual.tensors.ravel(), expected.ravel())[0, 1] > 0.997
