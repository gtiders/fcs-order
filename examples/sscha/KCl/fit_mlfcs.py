"""Run the native MLFCS SSCHA calculation for KCl."""

from __future__ import annotations

from mlfcs import write_force_constants
import argparse
import json

import numpy as np
from common import (
    ITERATIONS,
    MLFCS_CUTOFF,
    RESULTS,
    SEED,
    SNAPSHOTS,
    TEMPERATURE,
    mlfcs_calculator,
    mlfcs_working_cells,
)

from mlfcs.physics.sscha.solver import SSCHA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    _, primitive, reference = mlfcs_working_cells()
    calculator = mlfcs_calculator()
    sscha = SSCHA(
        primitive,
        reference=reference,
        cutoff=MLFCS_CUTOFF,
        temperature=TEMPERATURE,
        snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=SEED,
        imaginary_modes="absolute",
    )
    for _ in range(args.iterations):
        sscha.step(calculator, calculate_free_energy=True)
    history = {"iteration": [], "free_energy_eV_per_atom": [], "error_eV_per_atom": []}
    for iteration in sscha.history:
        if iteration.free_energy is None:
            continue
        free_energy = float(iteration.free_energy) / 8.0
        error = float(iteration.free_energy_error) / 8.0
        history["iteration"].append(iteration.index)
        history["free_energy_eV_per_atom"].append(free_energy)
        history["error_eV_per_atom"].append(error)
        print(
            f"FREE_ENERGY software=MLFCS iteration={iteration.index} "
            f"eV_per_atom={free_energy:.12e} error_eV_per_atom={error:.12e}",
            flush=True,
        )
    final = sscha.force_constants
    if final is None:
        raise RuntimeError("MLFCS SSCHA produced no effective FC2")
    np.savez_compressed(
        RESULTS / "mlfcs_sscha_fc2.npz",
        force_constants=final.materialize(2, max_bytes=None),
    )
    write_force_constants(final, RESULTS / "mlfcs_sscha.h5", format="hdf5")
    write_force_constants(final, RESULTS / "FORCE_CONSTANTS_MLFCS_SSCHA", format="phonopy", order=2)
    (RESULTS / "free_energy_mlfcs.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="ascii"
    )
    print(f"wrote MLFCS SSCHA outputs under {RESULTS}")


if __name__ == "__main__":
    main()
