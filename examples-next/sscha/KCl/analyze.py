"""Combine and summarize regenerated KCl SSCHA results."""

from __future__ import annotations

import json

from common import ITERATIONS, RESULTS, SEED, SNAPSHOTS, TEMPERATURE


def main() -> None:
    phonopy = json.loads((RESULTS / "free_energy_phonopy.json").read_text())
    mlfcs = json.loads((RESULTS / "free_energy_mlfcs.json").read_text())
    combined = {"phonopy": phonopy, "MLFCS": mlfcs}
    (RESULTS / "free_energy_convergence.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="ascii"
    )
    metadata = {
        "temperature_K": TEMPERATURE,
        "snapshots": SNAPSHOTS,
        "iterations": ITERATIONS,
        "random_seed": SEED,
        "phonopy_harmonic": "reference/phonopy_fc222_JPCM2022.yaml.xz",
        "working_cell": "phonopy.unitcell (shared with MLFCS)",
        "supercell_atoms": 64,
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print("combined regenerated free-energy histories and metadata")


if __name__ == "__main__":
    main()
