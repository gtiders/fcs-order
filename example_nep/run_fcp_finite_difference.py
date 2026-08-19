#!/usr/bin/env python3
"""Generate MLFCS FC2/FC3 from the published hiPhive FCPs.

This is a finite-difference reproduction of the published FCP models.  The
FCP is used as the ASE force calculator; no training data are reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import write
from hiphive import ForceConstantPotential
from hiphive.calculators import ForceConstantCalculator

from mlfcs import ForceConstantCalculation


ROOT = Path(__file__).resolve().parent / "dataset"
CASES = {
    "BaGaGe": {
        "fcp": "fcp_ols_original_model_5.fcp",
        "md_repeat": (2, 2, 2),
        "fd_repeat": (2, 2, 2),
        "cutoff": {2: 5.4, 3: 4.7},
        "max_body_order": {2: 2, 3: 2},
    },
    "MoS2": {
        "fcp": "fcp_order6.fcp",
        "md_repeat": (16, 16, 1),
        "fd_repeat": (4, 4, 1),
        "cutoff": {2: 10.4, 3: 5.6},
        "max_body_order": {2: 2, 3: 3},
    },
    "SnSe": {
        "fcp": "fcp_cm16_rfe-ridge_nf-3000_alpha-1.0.pickle",
        "md_repeat": (4, 11, 11),
        "fd_repeat": (2, 4, 4),
        "cutoff": {2: 8.0, 3: 4.5},
        "max_body_order": {2: 2, 3: 3},
    },
}
DISPLACEMENT = 0.01


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def build_calculation(primitive, reference, order: int, settings):
    return ForceConstantCalculation(
        primitive,
        reference=reference,
        order=order,
        cutoff=settings["cutoff"][order],
        max_body_order=settings["max_body_order"][order],
        displacement=DISPLACEMENT,
        symprec=1e-4,
        verbose=True,
    )


def evaluate_with_cache(calculation, calculator, archive: Path) -> np.ndarray:
    expected = (len(calculation.plan), len(calculation.supercell), 3)
    completed = 0
    values: list[np.ndarray] = []
    if archive.is_file():
        cached = np.load(archive)
        values = [np.asarray(item) for item in cached["forces"]]
        completed = len(values)
        if np.asarray(cached["forces"]).shape[1:] != expected[1:]:
            values = []
            completed = 0
        elif completed == expected[0]:
            print(f"Using cached forces: {archive}")
            return np.asarray(values)
        else:
            print(f"Resuming cached forces: {completed}/{expected[0]}")

    for zero_based_index, atoms in enumerate(calculation.plan):
        if zero_based_index < completed:
            continue
        index = zero_based_index + 1
        atoms.calc = calculator
        values.append(np.asarray(atoms.get_forces(), dtype=float))
        if index % 50 == 0 or index == expected[0]:
            current = np.asarray(values)
            np.savez_compressed(
                archive,
                forces=current,
                configuration_ids=np.arange(index, dtype=int),
                atom_order=np.asarray("reference"),
            )
            print(f"ASE force evaluations: {index}/{expected[0]}", flush=True)
    result = np.asarray(values)
    if result.shape != expected:
        raise ValueError(f"force cache has shape {result.shape}, expected {expected}")
    return result


def run_case(case: str, orders: tuple[int, ...], overwrite: bool) -> None:
    settings = CASES[case]
    root = ROOT / case
    fcp_path = root / settings["fcp"]
    fcp = ForceConstantPotential.read(str(fcp_path))
    primitive = fcp.primitive_structure.copy()
    md_reference = primitive.repeat(settings["md_repeat"])
    reference = primitive.repeat(settings["fd_repeat"])

    original = root / "original"
    original.mkdir(parents=True, exist_ok=True)
    write(original / "primitive.vasp", primitive, format="vasp", direct=True, sort=False)
    write(original / "supercell.vasp", md_reference, format="vasp", direct=True, sort=False)
    fd_root = root / "mlfcs"
    fd_root.mkdir(parents=True, exist_ok=True)
    write(fd_root / "fd_supercell.vasp", reference, format="vasp", direct=True, sort=False)
    (original / "source.json").write_text(
        json.dumps(
            {
                "source_fcp": fcp_path.name,
                "source_fcp_sha256": sha256(fcp_path),
                "supercell_matrix": np.diag(settings["md_repeat"]).tolist(),
                "finite_difference_supercell_matrix": np.diag(settings["fd_repeat"]).tolist(),
                "primitive_atoms": len(primitive),
                "supercell_atoms": len(md_reference),
                "finite_difference_atoms": len(reference),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    for order in orders:
        output = fd_root / f"fc{order}"
        output.mkdir(parents=True, exist_ok=True)
        archive = output / "forces.npz"
        if overwrite and archive.exists():
            archive.unlink()
        calculation = build_calculation(primitive, reference, order, settings)
        calculator = ForceConstantCalculator(fcp.get_force_constants(reference))
        forces = evaluate_with_cache(calculation, calculator, archive)
        result = calculation.reap(forces, acoustic_sum_rule=(order == 2))
        result.write(output / "mlfcs.h5", format="hdf5")
        if order == 2:
            result.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
            result.write(output / "fc2.h5", format="phonopy_hdf5", order=2)
        else:
            result.write(output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
            result.write(output / "fc3.h5", format="phono3py_hdf5", order=3)
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "method": "mlfcs-finite-difference-ase-hiphive-fcp",
                    "order": order,
                    "source_fcp": fcp_path.name,
                    "source_fcp_sha256": sha256(fcp_path),
                    "cutoff_angstrom": settings["cutoff"][order],
                    "max_body_order": settings["max_body_order"][order],
                    "displacement_angstrom": DISPLACEMENT,
                    "supercell_matrix": np.diag(settings["fd_repeat"]).tolist(),
                    "configuration_count": len(calculation.plan),
                    "acoustic_sum_rule": order == 2,
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )
        print(f"Wrote {case} order-{order} outputs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=sorted(CASES), default=None)
    parser.add_argument("--order", type=int, choices=(2, 3), nargs="+", default=(2, 3))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for case in args.cases or sorted(CASES):
        run_case(case, tuple(args.order), args.overwrite)


if __name__ == "__main__":
    main()
