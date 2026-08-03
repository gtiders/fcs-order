from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ase.io import read

from mlfcs import ForceConstantCalculation

STRUCTURES = Path(__file__).parent / "structures"


@pytest.mark.reference
def test_Si_FC3_external_sow_order_is_frozen():
    manifest = json.loads((STRUCTURES / "sow-plan.json").read_text(encoding="utf-8"))
    calculation = ForceConstantCalculation(
        read(STRUCTURES / "POSCAR-unitcell"),
        order=3,
        supercell=(3, 3, 3),
        cutoff=-6,
        displacement=0.01,
        jax_platform="cpu",
        report_cutoff=False,
    )

    assert manifest["atom_order"] == "grouped"
    assert manifest["plan_hash"] == calculation.plan.hash
    assert len(manifest["configurations"]) == len(calculation.plan) == 168
    for expected_id, record in enumerate(manifest["configurations"]):
        path = STRUCTURES / f"POSCAR-{expected_id + 1:03d}"
        assert record["configuration_id"] == expected_id
        assert record["filename"] == path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
