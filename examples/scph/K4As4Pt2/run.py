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
    parser.add_argument("--temperature-step", type=float, default=None)
    parser.add_argument("--interpolation-multiplier", type=int, default=1)
    parser.add_argument("--scph-multiplier", type=int, default=2)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--qpoint-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    harmonic_q, harmonic = harmonic_frequencies(read_hdf5(SOURCE), args.interpolation_multiplier)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "harmonic.npz", qpoints=harmonic_q, frequencies=harmonic)

    temperatures = sorted(args.temperatures)
    if args.temperature_step is not None:
        if args.temperature_step <= 0:
            raise ValueError("--temperature-step must be positive")
        temperatures = list(np.arange(temperatures[0], temperatures[-1] + args.temperature_step / 2, args.temperature_step))
    calculation = LoopSCPH(
        fc2=read_hdf5(SOURCE),
        fc4=read_hdf5(SOURCE),
        temperature=temperatures,
        interpolation_multiplier=args.interpolation_multiplier,
        scph_multiplier=args.scph_multiplier,
        mixing=args.mixing,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        qpoint_workers=args.qpoint_workers,
    )
    schedule_result = calculation.run()
    results = (
        schedule_result.results
        if hasattr(schedule_result, "results")
        else (schedule_result,)
    )
    for result in results:
        temperature = result.temperature
        label = f"T{int(temperature)}"
        target = OUTPUT / label
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite")
        target.mkdir(parents=True, exist_ok=True)
        result.force_constants.write(target / "mlfcs.h5", format="hdf5")
        result.force_constants.write(
            target / "FORCE_CONSTANTS_2ND", format="phonopy", order=2
        )
        np.savez_compressed(
            target / "frequencies.npz",
            qpoints=result.qpoints,
            frequencies=result.frequencies,
        )
        (target / "history.json").write_text(
            json.dumps(
                {
                    "temperature": temperature,
                    "interpolation_multiplier": args.interpolation_multiplier,
                    "scph_multiplier": args.scph_multiplier,
                    "converged": result.converged,
                    "iterations": [
                        {
                            "index": item.index,
                            "frequency_change_thz": item.frequency_change_thz,
                            "correction_norm": item.correction_norm,
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
