#!/usr/bin/env python3
"""Generate Ba8Ga16Ge30 NVE fitting data with the upstream hiPhive FCP.

The dynamics settings deliberately match the thermal-conductivity example in
the upstream hiPhive repository: a 2x2x2 cell, 1 fs time step, 10,000 Langevin
steps, followed by 5,000 NVE steps written every 50 steps.
"""

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

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
MD_ROOT = ROOT / "results"
REFERENCE = INPUT / "reference.vasp"
FCP = INPUT / "fcp_2body-5.4_4.35_4.35_least-squares.fcp"

PRIMITIVE = INPUT / "primitive.vasp"
TIMESTEP_FS = 1
NVT_STEPS = 10_000
NVT_INTERVAL = 25
NVE_STEPS = 5_000
NVE_INTERVAL = 50
DEFAULT_TEMPERATURES = (300, 400, 500, 600)


def _force_calculator(fcp: ForceConstantPotential, reference):
    """Build a calculator tied to the exact reference frame."""
    return ForceConstantCalculator(fcp.get_force_constants(reference))


def _write_extxyz(trajectory: Path, output: Path) -> int:
    frames = read(trajectory, index=":")
    for frame in frames:
        forces = frame.get_forces()
        frame.new_array("forces", forces)
        frame.calc = None
    write(output, frames, format="extxyz")
    return len(frames)


def _prepare_directory(path: Path, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} already contains output; pass --overwrite to replace this temperature."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_temperature(temperature: int, fcp: ForceConstantPotential, overwrite: bool) -> None:
    destination = MD_ROOT / f"T{temperature}K"
    _prepare_directory(destination, overwrite)

    reference = read(REFERENCE)
    atoms = reference.copy()
    atoms.calc = _force_calculator(fcp, reference)
    rng = np.random.default_rng(20260817 + temperature)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1.5 * temperature, rng=rng)

    nvt_trajectory = destination / "nvt.traj"
    nvt_dynamics = Langevin(
        atoms,
        TIMESTEP_FS * units.fs,
        temperature_K=temperature,
        friction=0.005,
        rng=rng,
    )
    with Trajectory(nvt_trajectory, "w", atoms) as writer:
        logger = MDLogger(
            nvt_dynamics,
            atoms,
            destination / "nvt.log",
            header=True,
            stress=False,
            peratom=True,
            mode="w",
        )
        nvt_dynamics.attach(logger, interval=NVT_INTERVAL)
        nvt_dynamics.attach(writer.write, interval=NVT_INTERVAL)
        nvt_dynamics.run(NVT_STEPS)

    _run_nve(temperature, fcp, destination, reference)


def _run_nve(
    temperature: int,
    fcp: ForceConstantPotential,
    destination: Path,
    reference,
) -> None:
    nvt_trajectory = destination / "nvt.traj"
    if not nvt_trajectory.is_file():
        raise FileNotFoundError(f"missing NVT trajectory: {nvt_trajectory}")
    atoms = read(nvt_trajectory, index=-1)
    atoms.calc = _force_calculator(fcp, reference)
    nve_trajectory = destination / "nve.traj"
    nve_dynamics = VelocityVerlet(atoms, TIMESTEP_FS * units.fs)
    with Trajectory(nve_trajectory, "w", atoms) as writer:
        logger = MDLogger(
            nve_dynamics,
            atoms,
            destination / "nve.log",
            header=True,
            stress=False,
            peratom=True,
            mode="w",
        )
        nve_dynamics.attach(logger, interval=NVE_INTERVAL)
        nve_dynamics.attach(writer.write, interval=NVE_INTERVAL)
        nve_dynamics.run(NVE_STEPS)

    nve_frames = len(read(nve_trajectory, index=":"))
    # ASE writes the NVE initial configuration plus every 50-step sample.
    expected_frames = NVE_STEPS // NVE_INTERVAL + 1
    if nve_frames != expected_frames:
        raise RuntimeError(f"expected {expected_frames} NVE frames, wrote {nve_frames}")
    metadata = {
        "temperature_K": temperature,
        "reference": REFERENCE.name,
        "atoms": len(atoms),
        "initialization": "Maxwell-Boltzmann at 1.5 * target temperature",
        "seed": 20260817 + temperature,
        "nvt": {
            "steps": NVT_STEPS,
            "interval": NVT_INTERVAL,
            "thermostat": "Langevin",
            "friction": 0.005,
        },
        "nve": {"steps": NVE_STEPS, "interval": NVE_INTERVAL, "frames": nve_frames},
        "force_constant_potential": FCP.name,
        "fcp_sha256": _fcp_sha256(),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    _write_extxyz(nve_trajectory, destination / "nve.extxyz")
    print(f"T={temperature} K: wrote {nve_frames} NVE frames to {destination / 'nve.extxyz'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=int, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing temperature directory"
    )
    parser.add_argument(
        "--nve-only",
        action="store_true",
        help="run only NVE from an existing completed NVT trajectory",
    )
    return parser.parse_args()


def _fcp_sha256() -> str:
    with FCP.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    args = parse_args()
    unsupported = sorted(set(args.temperatures) - set(DEFAULT_TEMPERATURES))
    if unsupported:
        raise ValueError(f"this case is scoped to 300, 400, 500, and 600 K; got {unsupported}")
    fcp = ForceConstantPotential.read(str(FCP))
    for temperature in args.temperatures:
        if args.nve_only:
            destination = MD_ROOT / f"T{temperature}K"
            _run_nve(temperature, fcp, destination, read(REFERENCE))
        else:
            run_temperature(temperature, fcp, args.overwrite)


if __name__ == "__main__":
    main()
