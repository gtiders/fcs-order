from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).parent / "data" / "reference.npz"
EXPECTED_SHA256 = "a6b6f36e416145dbd59d93dd30e0f8105f96065bf453e24d228bd82c4846b44f"


@pytest.mark.reference
def test_Si_FC3_reference_matches_recorded_provenance():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_SHA256
    with np.load(REFERENCE) as data:
        assert str(data["mlfcs_vasprun_combined_sha256"]) == (
            "9d4bb8edec81510d07337a7a53f1aa48805f92e3d4d037b2d8a54c30df5e4c51"
        )
        assert str(data["thirdorder_file_sha256"]) == (
            "9f2b7ff8a2128c9d8e9351f76c0f5142a3285787a1b77c545dc2b480ed223b29"
        )
