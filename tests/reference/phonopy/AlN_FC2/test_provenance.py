from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REFERENCE = Path(__file__).parent / "data" / "reference.npz"
EXPECTED_SHA256 = "677f95b8fa8018fa3b5d43add18b1b11ed2f33643f962c66d0e3dcb36ae8c45c"


@pytest.mark.reference
def test_AlN_FC2_reference_matches_recorded_hash():
    assert REFERENCE.is_file()
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_SHA256
