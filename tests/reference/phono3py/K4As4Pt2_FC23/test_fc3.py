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
    assert (
        calculation.plan.hash == "0f90240455f0874e4e5f722124969770bc554b8615ad5ca2068869130fc42fd4"
    )
    assert len(calculation.plan) == 2328
    assert np.array_equal(sparse.clusters, data["fc3_clusters"])
    assert_matches_phono3py(sparse.tensors, data["phono3py_raw_fc3_tensors"], order=3)
