#!/usr/bin/env python3
"""Generate 300 K SnSe fitting snapshots with the published hiPhive FCP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from hiphive import ForceConstantPotential
from hiphive.calculators import ForceConstantCalculator

CASE = Path(__file__).resolve().parents[1]
FCP = CASE / "input" / "fcp_cm16_rfe-ridge_nf-3000_alpha-1.0.pickle"
REFERENCE = CASE / "input" / "reference.vasp"
OUTPUT = CASE / "md" / "T300K"
TEMPERATURE = 300
TIMESTEP_FS = 1.0
NVT_STEPS = 10_000
NVT_INTERVAL = 25
NVE_STEPS = 5_000
NVE_INTERVAL = 25
SEED = 20260819


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _calculator(fcp: ForceConstantPotential, reference):
    return ForceConstantCalculator(fcp.get_force_constants(reference))


def _prepare_output(overwrite: bool) -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{OUTPUT} is not empty; pass --overwrite")
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)


def _write_extxyz(source: Path, target: Path) -> int:
    frames = read(source, index=":")
    for frame in frames:
        frame.new_array("forces", frame.get_forces())
        frame.calc = None
    write(target, frames, format="extxyz")
    return len(frames)


def run(overwrite: bool) -> None:
    if not FCP.is_file():
        raise FileNotFoundError(FCP)
    if not REFERENCE.is_file():
        raise FileNotFoundError(
            f"{REFERENCE} is missing; run run_fcp_finite_difference.py SnSe first"
        )
    _prepare_output(overwrite)

    fcp = ForceConstantPotential.read(str(FCP))
    reference = read(REFERENCE)
    atoms = reference.copy()
    atoms.calc = _calculator(fcp, reference)
    rng = np.random.default_rng(SEED)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1.5 * TEMPERATURE, rng=rng)

    nvt_path = OUTPUT / "nvt.traj"
    nvt = Langevin(
        atoms,
        TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE,
        friction=0.005,
        rng=rng,
    )
    with Trajectory(nvt_path, "w", atoms) as trajectory:
        logger = MDLogger(nvt, atoms, OUTPUT / "nvt.log", header=True, peratom=True, mode="w")
        nvt.attach(logger, interval=NVT_INTERVAL)
        nvt.attach(trajectory.write, interval=NVT_INTERVAL)
        nvt.run(NVT_STEPS)

    atoms = read(nvt_path, index=-1)
    atoms.calc = _calculator(fcp, reference)
    nve_path = OUTPUT / "nve.traj"
    nve = VelocityVerlet(atoms, TIMESTEP_FS * units.fs)
    with Trajectory(nve_path, "w", atoms) as trajectory:
        logger = MDLogger(nve, atoms, OUTPUT / "nve.log", header=True, peratom=True, mode="w")
        nve.attach(logger, interval=NVE_INTERVAL)
        nve.attach(trajectory.write, interval=NVE_INTERVAL)
        nve.run(NVE_STEPS)

    frames = _write_extxyz(nve_path, OUTPUT / "nve.extxyz")
    expected = NVE_STEPS // NVE_INTERVAL + 1
    if frames != expected:
        raise RuntimeError(f"expected {expected} NVE frames, wrote {frames}")
    metadata = {
        "temperature_K": TEMPERATURE,
        "reference": str(REFERENCE.relative_to(CASE)),
        "primitive": str((CASE / "input" / "primitive.vasp").relative_to(CASE)),
        "atoms": len(reference),
        "supercell": "2x4x4 force-constant reference",
        "initialization": "Maxwell-Boltzmann at 1.5 * target temperature",
        "seed": SEED,
        "nvt": {"steps": NVT_STEPS, "interval": NVT_INTERVAL, "thermostat": "Langevin"},
        "nve": {"steps": NVE_STEPS, "interval": NVE_INTERVAL, "frames": frames},
        "force_constant_potential": FCP.name,
        "fcp_sha256": _sha256(FCP),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"wrote {frames} SnSe NVE frames to {OUTPUT / 'nve.extxyz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.overwrite)


if __name__ == "__main__":
    main()
