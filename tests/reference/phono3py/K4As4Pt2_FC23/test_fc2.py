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
    assert np.array_equal(sparse.clusters, data["fc2_clusters"])
    assert_matches_phono3py(sparse.tensors, data["phono3py_raw_fc2_tensors"], order=2)
