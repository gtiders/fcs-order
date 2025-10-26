"""
This script carries a series of molecular dynamics simulations at different
temperatures and stores snapshots to file.
"""

import os
import sys

import click

import numpy as np

from ase import units
from ase.io import write, read
from ase.build import make_supercell
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger

from hiphive import ForceConstantPotential, ClusterSpace, StructureContainer
from trainstation import Optimizer


def run_md(
    temperatures, atoms_ideal, outdir, calc, neq, nprod, dt, dump, friction=0.02
):
    """Run molecular dynamics simulation at a given temperature."""
    for temperature in temperatures:
        print(f"Temperature: {temperature}")

        # set up molecular dynamics simulation
        atoms = atoms_ideal.copy()
        atoms.calc = calc
        dyn = Langevin(
            atoms, dt * units.fs, temperature_K=temperature, friction=friction
        )

        # equilibration run
        rs = np.random.RandomState(2020)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rs)
        dyn.run(neq)

        # production run
        log_file = f"{outdir}/T{temperature}.log"
        traj_file = f"{outdir}/T{temperature}.traj"
        logger = MDLogger(
            dyn, atoms, log_file, header=True, stress=False, peratom=True, mode="w"
        )
        traj_writer = Trajectory(traj_file, "w", atoms)
        dyn.attach(logger, interval=dump)
        dyn.attach(traj_writer.write, interval=dump)
        dyn.run(nprod)

        # prepare snapshots for later use
        frames = []
        for atoms in read(traj_file, ":"):
            forces = atoms.get_forces()
            displacements = atoms.positions - atoms_ideal.get_positions()
            atoms.positions = atoms_ideal.get_positions()
            atoms.new_array("displacements", displacements)
            atoms.new_array("forces", forces)
            frames.append(atoms.copy())
        print(f" Number of snapshots: {len(frames)}")
        write(f"{outdir}/snapshots_T{temperature}.xyz", frames, format="extxyz")


@click.command()
@click.argument("na", type=int)
@click.argument("nb", type=int)
@click.argument("nc", type=int)
@click.option(
    "--prim",
    type=click.Path(exists=True),
    default="POSCAR",
    help="Path to primitive structure file.",
)
@click.option(
    "--calc",
    type=click.Choice(["nep", "dp", "hiphive", "ploymp"], case_sensitive=False),
    default=None,
    help="Calculator to use (nep or dp or hiphive or ploymp)",
)
@click.option(
    "--potential",
    type=click.Path(exists=True),
    default=None,
    help="Potential file to use (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml')",
)
@click.option(
    "--temperatures",
    type=str,
    default="300",
    help="Temperatures to use (e.g. '2000,1000,300' for 2000 K, 1000 K, and 300 K)",
)
@click.option(
    "--neq",
    type=int,
    default=3000,
    help="Number of equilibration steps to use.",
)
@click.option(
    "--nprod",
    type=int,
    default=3000,
    help="Number of production steps to use.",
)
@click.option(
    "--dt",
    type=float,
    default=1.0,
    help="Time step to use (in fs).",
)
@click.option(
    "--dump",
    type=int,
    default=100,
    help="Dump interval to use.",
)
@click.option(
    "--friction",
    type=float,
    default=0.02,
    help="Friction coefficient for Langevin thermostat (in ps^-1).",
)
@click.option(
    "--cutoff", type=float, default=6.0, help="Cutoff distance to use (in Angstrom)."
)
@click.option(
    "--outdir", type=click.Path(), default="md_runs", help="Output directory to use."
)
def get_fc(
    na,
    nb,
    nc,
    prim="POSCAR",
    calc=None,
    potential=None,
    temperatures="300",
    neq=3000,
    nprod=5000,
    dt=1.0,
    dump=100,
    friction=0.02,
    cutoff=6.0,
    outdir="md_runs",
):
    """Get force constants from some temperature."""
    # Validate that calc and potential must be provided together
    if (calc is not None and potential is None) or (
        calc is None and potential is not None
    ):
        raise click.BadParameter("--calc and --potential must be provided together")

    temps = [int(t.strip()) for t in temperatures.split(",")]
    cutoffs = [cutoff]
    atoms_ideal_prim = read(prim)
    atoms_idea = make_supercell(
        atoms_ideal_prim, np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    )
    # If calculator type and potential file are specified, set up the calculator
    if calc is not None and potential is not None:
        if calc.lower() == "nep":
            # Add NEP calculator initialization code here
            print(f"Using NEP calculator with potential: {potential}")
            try:
                from calorine.calculators import CPUNEP

                calc = CPUNEP(potential)
            except ImportError:
                print("calorine not found, please install it first")
                sys.exit(1)
        elif calc.lower() == "dp":
            # Add DP calculator initialization code here
            print(f"Using DP calculator with potential: {potential}")
            try:
                from deepmd.calculator import DP

                calc = DP(model=potential)
            except ImportError:
                print("deepmd not found, please install it first")
                sys.exit(1)
        elif calc.lower() == "hiphive":
            # Add hiphive calculator initialization code here
            print(f"Using hiphive calculator with potential: {potential}")
            try:
                from hiphive import ForceConstantPotential
                from hiphive.calculators import ForceConstantCalculator

                fcp = ForceConstantPotential.read(potential)
                fcs = fcp.get_force_constants(atoms_idea)
                calc = ForceConstantCalculator(fcs)

            except ImportError:
                print("hiphive not found, please install it first")
                sys.exit(1)
        elif calc.lower() == "ploymp":
            # Add ploymp calculator initialization code here
            print(f"Using ploymp calculator with potential: {potential}")
            try:
                from pypolymlp.calculator.utils.ase_calculator import (
                    PolymlpASECalculator,
                )

                calc = PolymlpASECalculator(pot=potential)
            except ImportError:
                print("pypolymlp not found, please install it first")
                sys.exit(1)
    else:
        print("No calculator provided")
        sys.exit(1)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(f"{outdir}/fcps", exist_ok=True)
        os.makedirs(f"{outdir}/2nd", exist_ok=True)
    run_md(temps, atoms_idea, outdir, calc, neq, nprod, dt, dump, friction)

    for T in temps:
        structures = read(f"{outdir}/snapshots_T{T}.xyz", index=":")

        cs = ClusterSpace(structures[0], cutoffs)
        sc = StructureContainer(cs)
        for s in structures:
            sc.add_structure(s)

        opt = Optimizer(sc.get_fit_data(), train_size=1.0)
        opt.train()
        print(opt)
        fcp = ForceConstantPotential(cs, opt.parameters)
        fcs.write_to_phonopy(f"{outdir}/2nd/T{T}_FORCE_CONSTANTS", format="text")
        fcp.write(f"{outdir}/fcps/T{T}.fcp")
