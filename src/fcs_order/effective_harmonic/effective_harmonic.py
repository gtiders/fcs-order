"""
This script carries a series of molecular dynamics simulations at different
temperatures and stores snapshots to file.
"""

import os
import sys

import click
import numpy as np
from typing import List

from ase import units
from ase.io import write, read
from ase.build import make_supercell
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger


# hiphive and trainstation will be imported only when needed
def _check_hiphive_imports():
    """Check if hiphive and trainstation are available, raise error if not."""
    try:
        from hiphive import ForceConstantPotential, ClusterSpace, StructureContainer
        from hiphive.calculators import ForceConstantCalculator
        from hiphive.utilities import prepare_structures
        from trainstation import Optimizer

        return (
            ForceConstantPotential,
            ClusterSpace,
            StructureContainer,
            ForceConstantCalculator,
            Optimizer,
            prepare_structures,
        )
    except ImportError as e:
        print("hiphive and trainstation are required for hiphive calculator")
        print(f"Import error: {e}")
        sys.exit(1)


def _run_md(
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


def initialize_calculator(calc_type: str, potential_file: str, atoms):
    """
    Initialize ASE calculator based on calculator type and potential file.

    Parameters
    ----------
    calc_type : str
        Calculator type ('nep', 'dp', 'hiphive', 'ploymp')
    potential_file : str
        Path to potential file
    atoms : ase.Atoms
        Atoms object for hiphive calculator

    Returns
    -------
    calculator : ase.calculators.calculator.Calculator
        Initialized ASE calculator

    Raises
    ------
    SystemExit
        If required package is not installed
    """
    calc_type = calc_type.lower()

    if calc_type == "nep":
        print(f"Using NEP calculator with potential: {potential_file}")
        try:
            from calorine.calculators import CPUNEP

            return CPUNEP(potential_file)
        except ImportError:
            print("calorine not found, please install it first")
            sys.exit(1)

    elif calc_type == "dp":
        print(f"Using DP calculator with potential: {potential_file}")
        try:
            from deepmd.calculator import DP

            return DP(model=potential_file)
        except ImportError:
            print("deepmd not found, please install it first")
            sys.exit(1)

    elif calc_type == "hiphive":
        print(f"Using hiphive calculator with potential: {potential_file}")
        (
            ForceConstantPotential,
            ClusterSpace,
            StructureContainer,
            ForceConstantCalculator,
            Optimizer,
            _,
        ) = _check_hiphive_imports()
        try:
            fcp = ForceConstantPotential.read(potential_file)
            fcs = fcp.get_force_constants(atoms)
            return ForceConstantCalculator(fcs)
        except ImportError:
            print("hiphive not found, please install it first")
            sys.exit(1)

    elif calc_type == "ploymp":
        print(f"Using ploymp calculator with potential: {potential_file}")
        try:
            from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

            return PolymlpASECalculator(pot=potential_file)
        except ImportError:
            print("pypolymlp not found, please install it first")
            sys.exit(1)
    else:
        print(f"Unknown calculator type: {calc_type}")
        sys.exit(1)


@click.group()
def main():
    pass


@main.command()
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
    "--sposcar",
    type=click.Path(exists=True),
    default="SPOSCAR",
    help="Path to supercell structure file.",
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
    sposcar="SPOSCAR",
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
    """Get effective harmonic force constants from some temperature for a given supercell.
    na, nb, nc is the supercell size for training which does not need to match the size of sposcar.
    prim is the primitive structure file.
    sposcar is the supercell structure file.
    calc is the calculator type (nep or dp or hiphive or ploymp).
    potential is the potential file path (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml').
    temperatures is the temperature list (e.g. '2000,1000,300' for 2000 K, 1000 K, and 300 K).
    neq is the number of equilibration steps.
    nprod is the number of production steps.
    dt is the time step (in fs).
    dump is the dump interval.
    friction is the friction coefficient of Langevin thermostat (in ps^-1).
    cutoff is the cutoff distance (in Angstrom).
    """
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
        calc = initialize_calculator(calc, potential, atoms_idea)
    else:
        print("No calculator provided")
        sys.exit(1)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(f"{outdir}/fcps", exist_ok=True)
        os.makedirs(f"{outdir}/2nd", exist_ok=True)
    _run_md(temps, atoms_idea, outdir, calc, neq, nprod, dt, dump, friction)

    for T in temps:
        structures = read(f"{outdir}/snapshots_T{T}.xyz", index=":")

        if calc.lower() == "hiphive":
            (
                ForceConstantPotential,
                ClusterSpace,
                StructureContainer,
                ForceConstantCalculator,
                Optimizer,
                _,
            ) = _check_hiphive_imports()
            cs = ClusterSpace(atoms_ideal_prim, cutoffs)
            sc = StructureContainer(cs)
            for s in structures:
                sc.add_structure(s)

            opt = Optimizer(sc.get_fit_data(), train_size=1.0)
            opt.train()
            print(opt)
            fcp = ForceConstantPotential(cs, opt.parameters)
            fcp.get_force_constants(read(sposcar)).write_to_phonopy(
                f"{T}_FORCE_CONSTANT", format="text"
            )
            fcp.write(f"{outdir}/fcps/T{T}.fcp")
        else:
            print(
                f"Force constant extraction is currently only supported for hiphive calculator, not for {calc}"
            )
            sys.exit(1)


@main.command()
@click.argument("dft_structures", type=click.Path(exists=True), nargs=-1)
@click.option("--prim", type=click.Path(exists=True), nargs=1)
@click.option("--sposcar", type=click.Path(exists=True), nargs=1)
@click.option("--cutoff", type=float, nargs=1)
def get_effective_force_constants_dfts(
    dft_structures: List[str],
    prim: str,
    sposcar: str,
    cutoff: float,
):
    cutoffs = [cutoff]
    atoms_ideal_prim = read(prim)
    atoms_ideal = read(sposcar)
    dft_structures_ans = []
    for dft_structure in dft_structures:
        dft_structures_ans.extend(read(dft_structure, index=":"))
    (
        ForceConstantPotential,
        ClusterSpace,
        StructureContainer,
        _,
        Optimizer,
        prepare_structures,
    ) = _check_hiphive_imports()
    cs = ClusterSpace(atoms_ideal_prim, cutoffs)
    sc = StructureContainer(cs)
    dft_structures_ans = prepare_structures(dft_structures_ans, atoms_ideal)
    for s in dft_structures_ans:
        sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), train_size=1.0)
    opt.train()
    print(opt)
    fcp = ForceConstantPotential(cs, opt.parameters)
    fcp.get_force_constants(atoms_ideal).write_to_phonopy(
        f"{cutoff}_FORCE_CONSTANT", format="text"
    )
    fcp.write(f"{cutoff}.fcp")
