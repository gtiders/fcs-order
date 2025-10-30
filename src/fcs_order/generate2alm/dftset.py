"""Process DFTSETS files for alamode software data preparation."""

import sys
import click
from ase.io import read
from ase import Atoms
from .extxyz import write_extxyz
import numpy as np

# Physical constants: conversion factors
EV_TO_RYD = 1 / 13.60569253  # 1 eV = 1/13.60569253 Rydberg
ANGSTROM_TO_BOHR = 1 / 0.529177210903  # 1 Å = 1/0.529177210903 bohr
FORCE_CONV = EV_TO_RYD / ANGSTROM_TO_BOHR  # eV/Å -> Ryd/bohr


def calculate_properties(atoms: Atoms, sposcar: Atoms, if_correct_force: bool = False):
    """
    Calculate displacements and forces from reference structure to current structure

    Parameters:
        atoms: Current structure (ASE Atoms object)
        sposcar: Reference structure (ASE Atoms object)
        if_correct_force: Whether to correct forces based on sposcar

    Returns:
        tuple: (displacements, forces, potential_energy)
            displacements: Atomic displacement array (in bohr units)
            forces: Atomic force array (in Ryd/bohr units)
            potential_energy: Potential energy (in Rydberg units)
    """

    displacements = (
        atoms.get_positions() * ANGSTROM_TO_BOHR
        - sposcar.get_positions() * ANGSTROM_TO_BOHR
    )

    if if_correct_force:
        forces = (atoms.get_forces() - sposcar.get_forces()) * FORCE_CONV
    else:
        forces = atoms.get_forces() * FORCE_CONV
    potential_energy = atoms.get_potential_energy() * EV_TO_RYD
    atoms.new_array("alm_disps", displacements)
    atoms.new_array("alm_forces", forces)
    atoms.info["alm_potential_energy"] = potential_energy
    symbols_not = [" " for i in range(len(sposcar))]
    atoms.new_array("symbols-not", np.array(symbols_not))
    return atoms


@click.command()
@click.argument(
    "sposcar",
    type=click.Path(exists=True),
    required=True,
)
@click.argument(
    "disps",
    nargs=-1,
    type=click.Path(exists=True),
    required=True,
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
    "--delta",
    required=False,
    help="Sampling interval, default value is 1",
    default=1,
    type=int,
)
@click.option(
    "--output",
    default="DFSETs",
    help="Output filename",
    type=click.Path(),
)
@click.option(
    "--run_with_potential",
    is_flag=True,
    default=False,
    help="Whether to use the given potential file to calculate forces and energies",
)
@click.option(
    "--correct_force",
    is_flag=True,
    default=False,
    help="Whether to correct forces based on sposcar",
)
def gen_dftset(
    sposcar, disps, calc, potential, delta, output, run_with_potential, correct_force
):
    """
    Generate DFTSETS files for alamode software data preparation. The sposcar and disps below can provide completed calculation files such as vasprun.xml to directly extract results. If only structures are provided, then a machine learning potential is needed to calculate forces and energies.
    sposcar: Reference structure file path
    disps: All structure file paths, each file contains multiple structures
    calc: Calculator type, optional values are nep, dp, hiphive, ploymp
    potential: Potential file path, corresponds to different file formats based on calc type
    delta: Sampling interval, default value is 1
    output: Output filename, default value is DFSETs
    run_with_potential: Whether to use the given potential file to calculate forces and energies, default value is False
    correct_force: Whether to correct forces based on sposcar, default value is False
    """
    # Read reference structure
    sposcar: Atoms = read(sposcar)

    # Read all structures
    atoms_list = []
    for disp in disps:
        atoms = read(disp, index=":")
        atoms_list.extend(atoms)

    # Need to run calculations using potential to get forces and energies
    if run_with_potential:
        if (calc is not None and potential is None) or (
            calc is None and potential is not None
        ):
            raise click.BadParameter("--calc and --potential must be provided together")
        # If calculator type and potential file are specified, set up the calculator
        if calc is not None and potential is not None:
            if calc.lower() == "nep":
                # Add NEP calculator initialization code here
                print(f"Using NEP calculator with potential: {potential}")
                try:
                    from calorine.calculators import CPUNEP

                    calculation = CPUNEP(potential)
                except ImportError:
                    print("calorine not found, please install it first")
                    sys.exit(1)
            elif calc.lower() == "dp":
                # Add DP calculator initialization code here
                print(f"Using DP calculator with potential: {potential}")
                try:
                    from deepmd.calculator import DP

                    calculation = DP(model=potential)
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
                    fcs = fcp.get_force_constants(sposcar)
                    calculation = ForceConstantCalculator(fcs)

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

                    calculation = PolymlpASECalculator(pot=potential)
                except ImportError:
                    print("pypolymlp not found, please install it first")
                    sys.exit(1)
        else:
            print("No calculator provided")
            sys.exit(1)
    frames = []
    if run_with_potential:
        for atoms in atoms_list:
            atoms.calc = calculation
            atoms = calculate_properties(atoms, sposcar)
            frames.append(atoms)
    else:
        for atoms in atoms_list:
            atoms = calculate_properties(atoms, sposcar)
            frames.append(atoms)
    write_extxyz(
        output,
        frames[::delta],
        columns=["symbols-not", "alm_disps", "alm_forces"],
        if_no_atom_count=True,
    )


if __name__ == "__main__":
    gen_dftset()
