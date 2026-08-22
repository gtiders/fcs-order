"""Fit the 101-frame 300 K NVE trajectory with hiPhive and compare MLFCS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from hiphive.calculators import ForceConstantCalculator
from hiphive.cutoffs import Cutoffs
from hiphive.input_output.phonopy import read_phonopy_fc2
from hiphive.utilities import get_displacements
from trainstation import Optimizer

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
PRIMITIVE = ROOT / "input" / "reference.vasp"
REFERENCE = read(PRIMITIVE).repeat((2, 2, 2))
SNAPSHOTS = read(CASE / "nve.extxyz", index=":")
OUTPUT = CASE / "hiphive"


def main() -> None:
    if len(SNAPSHOTS) != 101:
        raise ValueError(f"expected the complete 101-frame NVE trajectory, got {len(SNAPSHOTS)}")
    cutoffs = Cutoffs([[5.4, 4.35]])
    cs = ClusterSpace(REFERENCE, cutoffs, symprec=1e-4)
    container = StructureContainer(cs)
    for snapshot in SNAPSHOTS:
        prepared = REFERENCE.copy()
        prepared.new_array("displacements", get_displacements(snapshot, REFERENCE))
        prepared.new_array("forces", snapshot.get_forces())
        container.add_structure(prepared)

    optimizer = Optimizer(container.get_fit_data(), train_size=1.0, check_condition=False)
    optimizer.train()
    fcp = ForceConstantPotential(cs, optimizer.parameters, metadata=optimizer.summary)
    fcs = fcp.get_force_constants(REFERENCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fcs.write_to_phonopy(str(OUTPUT / "FORCE_CONSTANTS_2ND"), format="text")
    fcs.write_to_shengBTE(
        str(OUTPUT / "FORCE_CONSTANTS_3RD"),
        read(PRIMITIVE),
        cutoff=4.35,
        symprec=1e-4,
    )

    calculator = ForceConstantCalculator(fcs)
    predicted = []
    for snapshot in SNAPSHOTS:
        probe = snapshot.copy()
        probe.calc = calculator
        predicted.append(probe.get_forces())
    predicted = np.asarray(predicted)
    actual = np.asarray([snapshot.get_forces() for snapshot in SNAPSHOTS])
    residual = predicted - actual
    hiphive_fc2 = read_phonopy_fc2(str(OUTPUT / "FORCE_CONSTANTS_2ND"), format="text")
    mlfcs_fc2 = read_phonopy_fc2(str(CASE / "mlfcs" / "FORCE_CONSTANTS_2ND"), format="text")
    summary = {
        "frames": len(SNAPSHOTS),
        "cutoffs_angstrom": [5.4, 4.35],
        "max_body_order": 2,
        "n_dofs": int(cs.n_dofs),
        "n_dofs_by_order": {
            str(order): int(cs.get_n_dofs_by_order(order=order)) for order in (2, 3)
        },
        "force_rmse_eV_per_angstrom": float(np.sqrt(np.mean(residual**2))),
        "force_relative_error": float(np.linalg.norm(residual) / np.linalg.norm(actual)),
        "fc2_text_roundtrip_maximum": float(
            np.max(np.abs(hiphive_fc2 - fcs.get_fc_array(order=2)))
        ),
        "fc2_vs_mlfcs_text_maximum": float(np.max(np.abs(hiphive_fc2 - mlfcs_fc2))),
        "fc2_vs_mlfcs_text_relative": float(
            np.linalg.norm(hiphive_fc2 - mlfcs_fc2)
            / max(np.linalg.norm(mlfcs_fc2), np.finfo(float).tiny)
        ),
        "optimizer": optimizer.summary,
        "mlfcs_metrics": json.loads((CASE / "mlfcs" / "metrics.json").read_text()),
    }
    encoded = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    (OUTPUT / "metrics.json").write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
