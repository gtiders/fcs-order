from __future__ import annotations

import numpy as np
import pytest
from phono3py.file_IO import read_fc3_from_hdf5
from phonopy.file_IO import read_force_constants_hdf5

from mlfcs.model import ForceConstants
from tests.reference.phono3py.K4As4Pt2_FC23.case import calculation_and_reference


@pytest.mark.reference
@pytest.mark.parametrize(
    ("order", "format_name"),
    [(2, "phonopy_hdf5"), (3, "phono3py_hdf5")],
)
def test_K4As4Pt2_external_hdf5_roundtrip(tmp_path, order, format_name):
    _, calculation, sparse = calculation_and_reference(order)
    result = ForceConstants({}, calculation.supercell, sparse={order: sparse})
    target = tmp_path / f"fc{order}.hdf5"

    result.write(target, format=format_name)
    full = read_force_constants_hdf5(target) if order == 2 else read_fc3_from_hdf5(target)

    expected = sparse.to_dense(max_bytes=None)[tuple(sparse.clusters.T)]
    actual = full[tuple(sparse.clusters.T)]
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
