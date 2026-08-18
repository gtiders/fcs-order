"""Run the phonopy MLPSSCHA calculation for KCl."""

from __future__ import annotations

import argparse
import json

import numpy as np
from common import (
    ITERATIONS,
    POTENTIAL_PATH,
    RESULTS,
    SEED,
    SNAPSHOTS,
    TEMPERATURE,
    harmonic_phonopy,
)
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import MLPSSCHA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    phonon = harmonic_phonopy()
    phonopy_mlp = PhonopyMLP().load(POTENTIAL_PATH)
    sscha = MLPSSCHA(
        phonon,
        phonopy_mlp,
        temperature=TEMPERATURE,
        number_of_snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=SEED,
    )
    history = {"iteration": [], "free_energy_eV_per_atom": [], "error_eV_per_atom": []}
    for iteration in sscha:
        sscha.calculate_free_energy()
        free_energy = float(sscha.free_energy) / 2.0
        error = float(getattr(sscha, "free_energy_error", float("nan"))) / 2.0
        history["iteration"].append(iteration)
        history["free_energy_eV_per_atom"].append(free_energy)
        history["error_eV_per_atom"].append(error)
        print(
            f"FREE_ENERGY software=phonopy iteration={iteration} "
            f"eV_per_atom={free_energy:.12e} error_eV_per_atom={error:.12e}",
            flush=True,
        )
    np.save(RESULTS / "phonopy_sscha_final_fc2.npy", np.asarray(sscha.force_constants))
    write_FORCE_CONSTANTS(
        np.asarray(sscha.force_constants),
        filename=RESULTS / "FORCE_CONSTANTS_PHONOPY_SSCHA",
    )
    (RESULTS / "free_energy_phonopy.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="ascii"
    )
    print(f"wrote phonopy SSCHA outputs under {RESULTS}")


if __name__ == "__main__":
    main()
