from __future__ import annotations

import hashlib

import pytest

from tests.reference.sscha.KCl_phonopy.case import DATA

EXPECTED_SHA256 = {
    DATA / "polymlp.yaml": "da1da7ded0e6b9fdbc79f0d6773bd86b6cd682f1c211abab91c12a0f4894dd85",
    DATA / "phonopy_KCl.yaml": "94de84ebeaea5ae9370da5ae2eb17d187dee4cab81c4ff8ec13539aaa3e40eaa",
    DATA / "phonopy_sscha_fc_JPCM2022.yaml.xz": (
        "49010c569ddfec8702158be6bcce651022a85f5c0f21721d22369f278c5ac77d"
    ),
}


@pytest.mark.reference
def test_upstream_kcl_artifacts_match_recorded_hashes():
    for path, expected in EXPECTED_SHA256.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
