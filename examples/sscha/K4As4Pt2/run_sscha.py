"""Generate a 300 K native MLFCS SSCHA result for K4As4Pt2."""

from __future__ import annotations

from mlfcs import write_force_constants
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import read_hdf5
from mlfcs.physics.sscha.solver import SSCHA

CASE = Path(__file__).resolve().parent
INPUT = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "input"
FINITE = CASE.parent.parent / "finite-difference" / "K4As4Pt2" / "results"
RESULTS = CASE / "results"
TEMPERATURE = 300.0
SNAPSHOTS = 100
ITERATIONS = 5
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument(
        "--mixing",
        type=float,
        default=1.0,
        help="linear FC2 update fraction in (0, 1]; 1 keeps direct replacement",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS,
        help="directory for generated results",
    )
    args = parser.parse_args()
    results = args.output.resolve()
    results.mkdir(parents=True, exist_ok=True)
    primitive = read(INPUT / "primitive.vasp")
    harmonic = read_hdf5(FINITE / "harmonic" / "mlfcs.h5")
    if harmonic.relation is None:
        raise ValueError("harmonic result has no structure relation")
    # SSCHA must sample the explicit finite-difference reference supercell,
    # not the compact primitive-only relation stored in the FC2 result.
    reference = read(INPUT / "supercell.vasp")
    sscha = SSCHA(
        primitive,
        reference=reference,
        cutoff=6.0,
        temperature=TEMPERATURE,
        snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=SEED,
        initial_force_constants=harmonic,
        imaginary_modes="absolute",
        mixing=args.mixing,
        log_level=1,
    )
    calculator = PolymlpASECalculator(pot=INPUT / "polymlp.yaml")
    # The case parameter ``iterations`` means the number of updates after the
    # initial harmonic state.  ``SSCHA.run`` intentionally includes iteration
    # zero and therefore performs max_iterations + 1 updates; keep the case's
    # historical five-update trajectory explicit here.
    for _ in range(args.iterations):
        sscha.step(calculator, calculate_free_energy=True)
    effective = sscha.force_constants
    if effective is None:
        raise RuntimeError("SSCHA produced no effective force constants")
    np.savez_compressed(
        results / "sscha_fc2.npz", force_constants=effective.materialize(2, max_bytes=None)
    )
    history = {
        "temperature_K": TEMPERATURE,
        "snapshots": args.snapshots,
        "iterations": args.iterations,
        "mixing": args.mixing,
        "random_seed": SEED,
        "free_energy_eV_per_primitive_cell": [item.free_energy for item in sscha.history],
        "free_energy_error_eV_per_primitive_cell": [
            item.free_energy_error for item in sscha.history
        ],
        "relative_force_constant_change": [
            item.relative_force_constant_change for item in sscha.history
        ],
        "raw_relative_force_constant_change": [
            item.raw_relative_force_constant_change for item in sscha.history
        ],
        "fitting_relative_force_error": [
            item.fitting_relative_force_error for item in sscha.history
        ],
    }
    (results / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="ascii")
    write_force_constants(effective, results / "sscha.h5", format="hdf5", order=2)
    write_force_constants(effective, results / "FORCE_CONSTANTS_SSCHA", format="phonopy", order=2)
    write_force_constants(effective, results / "force_constants.xml", format="alamode", order=2)
    print(f"wrote fresh 300 K SSCHA result after {len(sscha.history)} iterations")


if __name__ == "__main__":
    main()
