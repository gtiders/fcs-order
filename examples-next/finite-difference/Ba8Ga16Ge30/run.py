"""Generate Ba8Ga16Ge30 IFCs by finite differences of the public hiPhive FCP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read
from hiphive import ForceConstantPotential
from hiphive.calculators import ForceConstantCalculator

from mlfcs import ForceConstantCalculation

CASE = Path(__file__).resolve().parent
INPUT = CASE / "input"
RESULTS = CASE / "results"
PRIMITIVE = INPUT / "reference.vasp"
FCP_PATH = INPUT / "fcp_2body-5.4_4.35_4.35_least-squares.fcp"
REPEATS = (2, 2, 2)
DISPLACEMENT = 0.01
CONFIG = {
    2: {"cutoff": 5.4, "name": "harmonic", "max_body_order": 2},
    3: {"cutoff": 4.35, "name": "three-phonon", "max_body_order": 2},
}


def _fcp_calculator(fcp: ForceConstantPotential, reference):
    force_constants = fcp.get_force_constants(reference)
    return ForceConstantCalculator(force_constants)


def _sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _evaluate(calculation, calculator, archive: Path, overwrite: bool) -> np.ndarray:
    expected = (len(calculation.plan), len(calculation.supercell), 3)
    if archive.is_file() and not overwrite:
        forces = np.load(archive)["forces"]
        if forces.shape == expected:
            print(f"Using cached forces: {archive}")
            return forces

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % max(1, total // 10) == 0:
            print(f"ASE force evaluations: {done}/{total}", flush=True)

    forces = calculation.evaluate(calculator, progress=progress)
    np.savez_compressed(archive, forces=forces)
    return forces


def run_order(order: int, fcp: ForceConstantPotential, overwrite: bool) -> None:
    settings = CONFIG[order]
    output = RESULTS / settings["name"]
    output.mkdir(parents=True, exist_ok=True)
    primitive = read(PRIMITIVE)
    reference = primitive.repeat(REPEATS)
    calculation = ForceConstantCalculation(
        primitive,
        reference=reference,
        order=order,
        cutoff=settings["cutoff"],
        max_body_order=settings["max_body_order"],
        displacement=DISPLACEMENT,
        symprec=1e-4,
        verbose=True,
    )
    forces = _evaluate(
        calculation,
        _fcp_calculator(fcp, reference),
        output / "forces.npz",
        overwrite,
    )
    result = calculation.reap(forces, acoustic_sum_rule=(order == 2))
    result.write(output / "mlfcs.h5", format="hdf5")
    if order == 2:
        result.write(output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    else:
        result.write(output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    metadata = {
        "method": "mlfcs-finite-difference-ase-hiphive-fcp",
        "order": order,
        "cutoff_angstrom": settings["cutoff"],
        "max_body_order": settings["max_body_order"],
        "displacement_angstrom": DISPLACEMENT,
        "supercell_matrix": np.diag(REPEATS).tolist(),
        "configuration_count": len(calculation.plan),
        "fcp": FCP_PATH.name,
        "fcp_sha256": _sha256(FCP_PATH),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"Wrote order-{order} outputs under {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(2, 3), nargs="+", default=(2, 3))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    fcp = ForceConstantPotential.read(str(FCP_PATH))
    for order in args.order:
        run_order(order, fcp, args.overwrite)


if __name__ == "__main__":
    main()
