"""Run a small, fixed-parameter K4As4Pt2 SSCHA check.

This script intentionally has no command-line interface.  Edit the constants
below and run it from the repository with ``uv run python run_sscha_simple.py``.
Results are written below ``results/manual`` so the regular case outputs are
never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import read_hdf5
from mlfcs.anharmonic.sscha import SSCHA

# Edit these values directly when testing.
TEMPERATURE_K = 300.0
SNAPSHOTS = 100
ITERATIONS = 1  # one Cartesian random fit; no self-consistent update
RANDOM_SEED = 42
USE_INITIAL_FORCE_CONSTANTS = False

CASE = Path(__file__).resolve().parent
INPUT = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "input"
FINITE_RESULTS = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "results"
OUTPUT = CASE / "results" / "manual-random"


 
def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primitive = read(INPUT / "primitive.vasp")
    harmonic = read_hdf5(FINITE_RESULTS / "harmonic" / "mlfcs.h5")
    reference = harmonic.relation.reference.copy()
    initial = harmonic if USE_INITIAL_FORCE_CONSTANTS else None
    options = {}
    if initial is not None:
        options["initial_force_constants"] = initial
    sscha = SSCHA(
        primitive,
        reference=reference,
        cutoff=-1,
        temperature=TEMPERATURE_K,
        snapshots=SNAPSHOTS,
        max_iterations=ITERATIONS,
        random_seed=RANDOM_SEED,
        imaginary_modes="absolute",
        log_level=1,
        **options,
    )
    calculator = PolymlpASECalculator(pot=INPUT / "polymlp.yaml")
    for _ in range(ITERATIONS):
        sscha.step(calculator, calculate_free_energy=True)

    effective = sscha.force_constants
    if effective is None:
        raise RuntimeError("SSCHA produced no effective force constants")
    np.savez_compressed(
        OUTPUT / "sscha_fc2.npz", force_constants=effective.materialize(2, max_bytes=None)
    )
    history = {
        "temperature_K": TEMPERATURE_K,
        "snapshots": SNAPSHOTS,
        "iterations": ITERATIONS,
        "random_seed": RANDOM_SEED,
        "free_energy_eV_per_primitive_cell": [x.free_energy for x in sscha.history],
        "free_energy_error_eV_per_primitive_cell": [
            x.free_energy_error for x in sscha.history
        ],
        "relative_force_constant_change": [
            x.relative_force_constant_change for x in sscha.history
        ],
        "fitting_relative_force_error": [
            x.fitting_relative_force_error for x in sscha.history
        ],
    }
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    effective.write(OUTPUT / "sscha.h5", format="hdf5", order=2)
    effective.write(OUTPUT / "FORCE_CONSTANTS_SSCHA", format="phonopy", order=2)
    effective.write(OUTPUT / "force_constants.xml", format="alamode", order=2)
    print(f"wrote SSCHA results to {OUTPUT}")


if __name__ == "__main__":
    main()
