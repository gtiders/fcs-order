from __future__ import annotations

import hashlib

import pytest

from tests.reference.phono3py.K4As4Pt2_FC23.case import DATA

EXPECTED_SHA256 = {
    DATA / "POSCAR": "6f097324431fe0c781a119071fbda730d3ca511aad5b1b46c15795f36391e49c",
    DATA / "polymlp.yaml": "f3f88866f39b6ca1549446a2801094815db7062ae391a132e2cf13d728e5bf14",
    DATA / "reference.npz": "274ffea910263f9722682e0b766b406175475d891b715794b3c3a6f0c37305b5",
}


@pytest.mark.reference
def test_K4As4Pt2_reference_artifacts_match_recorded_hashes():
    for path, expected in EXPECTED_SHA256.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
