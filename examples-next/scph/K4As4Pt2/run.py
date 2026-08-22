"""Run quartic loop-SCPH for the three-body-FC4 K4As4Pt2 fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlfcs import LoopSCPH, harmonic_frequencies, read_hdf5

CASE = Path(__file__).resolve().parent
SOURCE = CASE.parent.parent / "fitting" / "K4As4Pt2" / "results" / "three-body" / "mlfcs.h5"
OUTPUT = CASE / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperatures", nargs="+", type=float, default=[300, 600, 900])
    parser.add_argument("--interpolation-mesh", nargs=3, type=int, default=(3, 3, 3))
    parser.add_argument("--scph-mesh", nargs=3, type=int, default=(6, 6, 6))
    parser.add_argument("--mixing", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    harmonic_q, harmonic = harmonic_frequencies(read_hdf5(SOURCE), tuple(args.interpolation_mesh))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "harmonic.npz", qpoints=harmonic_q, frequencies=harmonic)

    warm_start = None
    for temperature in sorted(args.temperatures):
        label = f"T{int(temperature)}"
        target = OUTPUT / label
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite")
        target.mkdir(parents=True, exist_ok=True)
        fc2 = read_hdf5(SOURCE)
        fc4 = read_hdf5(SOURCE)
        result = LoopSCPH(
            fc2=fc2,
            fc4=fc4,
            temperature=temperature,
            interpolation_mesh=tuple(args.interpolation_mesh),
            scph_mesh=tuple(args.scph_mesh),
            mixing=args.mixing,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            warm_start=warm_start,
        ).run()
        warm_start = result.effective_force_constants
        result.effective_force_constants.write(target / "mlfcs.h5", format="hdf5")
        result.effective_force_constants.write(
            target / "FORCE_CONSTANTS_2ND", format="phonopy", order=2
        )
        np.savez_compressed(
            target / "frequencies.npz",
            qpoints=result.qpoints,
            frequencies=result.frequencies,
            loop_correction=result.loop_correction.materialize(2),
        )
        (target / "history.json").write_text(
            json.dumps(
                {
                    "temperature": temperature,
                    "interpolation_mesh": list(args.interpolation_mesh),
                    "scph_mesh": list(args.scph_mesh),
                    "converged": result.converged,
                    "iterations": [
                        {
                            "index": item.index,
                            "frequency_change_thz": item.frequency_change_thz,
                        }
                        for item in result.history
                    ],
                },
                indent=2,
            )
        )
        print(f"{label}: {len(result.history)} iterations, converged={result.converged}")


if __name__ == "__main__":
    main()
