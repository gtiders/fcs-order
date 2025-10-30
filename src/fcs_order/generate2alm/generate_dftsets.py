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


def calculate_properties(
    atoms: Atoms, reference_structure: Atoms, correct_forces: bool = False
):
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
        - reference_structure.get_positions() * ANGSTROM_TO_BOHR
    )

    if correct_forces:
        forces = (atoms.get_forces() - reference_structure.get_forces()) * FORCE_CONV
    else:
        forces = atoms.get_forces() * FORCE_CONV
    potential_energy = atoms.get_potential_energy() * EV_TO_RYD
    atoms.new_array("alm_disps", displacements)
    atoms.new_array("alm_forces", forces)
    atoms.info["alm_potential_energy"] = potential_energy
    placeholder_symbols = [" " for i in range(len(reference_structure))]
    atoms.new_array("symbols-not", np.array(placeholder_symbols))
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
    "--correct_forces",
    is_flag=True,
    default=False,
    help="Whether to correct forces based on sposcar",
)
def main(
    reference_structure_file,
    structure_files,
    calculator_type,
    potential_file,
    sampling_interval,
    output_file,
    run_with_potential,
    correct_forces,
):
    """
    Generate DFTSETS files for alamode software data preparation. The reference_structure_file and structure_files below can provide completed calculation files such as vasprun.xml to directly extract results. If only structures are provided, then a machine learning potential is needed to calculate forces and energies.
    reference_structure_file: Reference structure file path
    structure_files: All structure file paths, each file contains multiple structures
    calculator_type: Calculator type, optional values are nep, dp, hiphive, ploymp
    potential_file: Potential file path, corresponds to different file formats based on calculator_type
    sampling_interval: Sampling interval, default value is 1
    output_file: Output filename, default value is DFSETs
    run_with_potential: Whether to use the given potential file to calculate forces and energies, default value is False
    correct_force: Whether to correct forces based on reference_structure_file, default value is False
    """
    # Read reference structure
    reference_structure: Atoms = read(reference_structure_file)

    # Read all structures
    atoms_list = []
    for structure_file in structure_files:
        atoms = read(structure_file, index=":")
        atoms_list.extend(atoms)

    # Need to run calculations using potential to get forces and energies
    if run_with_potential:
        if (calculator_type is not None and potential_file is None) or (
            calculator_type is None and potential_file is not None
        ):
            raise click.BadParameter("--calc and --potential must be provided together")
        # If calculator type and potential file are specified, set up the calculator
        if calculator_type is not None and potential_file is not None:
            if calculator_type.lower() == "nep":
                # Add NEP calculator initialization code here
                print(f"Using NEP calculator with potential: {potential_file}")
                try:
                    from calorine.calculators import CPUNEP

                    calculator_instance = CPUNEP(potential_file)
                except ImportError:
                    print("calorine not found, please install it first")
                    sys.exit(1)
            elif calculator_type.lower() == "dp":
                # Add DP calculator initialization code here
                print(f"Using DP calculator with potential: {potential_file}")
                try:
                    from deepmd.calculator import DP

                    calculator_instance = DP(model=potential_file)
                except ImportError:
                    print("deepmd not found, please install it first")
                    sys.exit(1)
            elif calculator_type.lower() == "hiphive":
                # Add hiphive calculator initialization code here
                print(f"Using hiphive calculator with potential: {potential_file}")
                try:
                    from hiphive import ForceConstantPotential
                    from hiphive.calculators import ForceConstantCalculator

                    fcp = ForceConstantPotential.read(potential_file)
                    fcs = fcp.get_force_constants(reference_structure)
                    calculator_instance = ForceConstantCalculator(fcs)

                except ImportError:
                    print("hiphive not found, please install it first")
                    sys.exit(1)
            elif calculator_type.lower() == "ploymp":
                # Add ploymp calculator initialization code here
                print(f"Using ploymp calculator with potential: {potential_file}")
                try:
                    from pypolymlp.calculator.utils.ase_calculator import (
                        PolymlpASECalculator,
                    )

                    calculator_instance = PolymlpASECalculator(pot=potential_file)
                except ImportError:
                    print("pypolymlp not found, please install it first")
                    sys.exit(1)
        else:
            print("No calculator provided")
            sys.exit(1)
    processed_structures = []
    if run_with_potential:
        for atoms in atoms_list:
            atoms.calc = calculator_instance
            atoms = calculate_properties(atoms, reference_structure, correct_forces)
            processed_structures.append(atoms)
    else:
        for atoms in atoms_list:
            atoms = calculate_properties(atoms, reference_structure, correct_forces)
            processed_structures.append(atoms)
    write_extxyz(
        output_file,
        processed_structures[::sampling_interval],
        columns=["symbols-not", "alm_disps", "alm_forces"],
        if_no_atom_count=True,
    )