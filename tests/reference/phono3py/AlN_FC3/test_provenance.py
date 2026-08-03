from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"

EXPECTED_SHA256 = {
    DATA / "reference.npz": "310d6b3bd3a19f57af86ff2385870dbc28703a6b1509e2567e7eebbf8b2405ba",
    DATA / "training" / "phonopy_params_mp-661.yaml.xz": (
        "de153514ace4f0828d4111228b20f67fde02dd8bcac7e6c49ad52f24f958007e"
    ),
    DATA / "training" / "polymlp.yaml": (
        "cb81eb864fdc29e6f725d6ac9ec41b043beeadc073416d42fb75e3728ce415ec"
    ),
}


@pytest.mark.reference
def test_AlN_reference_artifacts_match_recorded_hashes():
    for path, expected in EXPECTED_SHA256.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
