"""Generate a 300 K native MLFCS SSCHA result for K4As4Pt2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import read_hdf5
from mlfcs.ifc.model import SparseOrderForceConstants
from mlfcs.anharmonic.sscha import SSCHA

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
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    primitive = read(INPUT / "primitive.vasp")
    harmonic = read_hdf5(FINITE / "harmonic" / "mlfcs.h5")
    if harmonic.relation is None:
        raise ValueError("harmonic result has no structure relation")
    reference = harmonic.relation.reference.copy()
    initial = harmonic.materialize(2, max_bytes=None)
    sscha = SSCHA(
        primitive,
        reference=reference,
        temperature=TEMPERATURE,
        snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=SEED,
        initial_force_constants=initial,
        imaginary_modes="absolute",
        log_level=1,
    )
    calculator = PolymlpASECalculator(pot=INPUT / "polymlp.yaml")
    # The case parameter ``iterations`` means the number of updates after the
    # initial harmonic state.  ``SSCHA.run`` intentionally includes iteration
    # zero and therefore performs max_iterations + 1 updates; keep the case's
    # historical five-update trajectory explicit here.
    for _ in range(args.iterations):
        sscha.step(calculator, calculate_free_energy=True)
    final = sscha.force_constants
    if final is None:
        raise RuntimeError("SSCHA produced no effective force constants")
    np.savez_compressed(RESULTS / "sscha_fc2.npz", force_constants=final)
    history = {
        "temperature_K": TEMPERATURE,
        "snapshots": args.snapshots,
        "iterations": args.iterations,
        "random_seed": SEED,
        "free_energy_eV_per_primitive_cell": [item.free_energy for item in sscha.history],
        "free_energy_error_eV_per_primitive_cell": [
            item.free_energy_error for item in sscha.history
        ],
        "relative_force_constant_change": [
            item.relative_force_constant_change for item in sscha.history
        ],
        "fitting_relative_force_error": [
            item.fitting_relative_force_error for item in sscha.history
        ],
    }
    (RESULTS / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="ascii")
    from mlfcs.anharmonic.common.fc2 import compact_fc2

    effective = read_hdf5(FINITE / "harmonic" / "mlfcs.h5")
    compact = compact_fc2(final, reference)
    # Replace the sparse tensors as well as the dense view.  Writers build an
    # export view from ``sparse`` when a structure relation is present; merely
    # assigning ``arrays`` would therefore silently export the old harmonic
    # tensors.
    source = effective.sparse[2]
    assert source.sites is not None
    assert source.translation_representatives is not None
    index = effective.relation.index
    tensors = np.empty_like(source.tensors)
    for row, sites in enumerate(source.sites):
        second = index.atom(
            int(sites[1]), source.translation_representatives[row, 0]
        )
        tensors[row] = compact[int(sites[0]), second]
    effective.sparse[2] = SparseOrderForceConstants(
        2,
        source.n_primitive,
        source.n_supercell,
        source.clusters.copy(),
        tensors,
        source.sites.copy(),
        source.translation_representatives.copy(),
    )
    effective.arrays = {2: compact}
    effective.write(RESULTS / "sscha.h5", format="hdf5", order=2)
    effective.write(RESULTS / "FORCE_CONSTANTS_SSCHA", format="phonopy", order=2)
    print(f"wrote fresh 300 K SSCHA result after {len(sscha.history)} iterations")


if __name__ == "__main__":
    main()
