"""Process DFTSETS files for alamode software data preparation."""

import numpy as np
from ..monkey_write import write_xyz

# Physical constants: conversion factors
EV_TO_RYD = 1 / 13.60569253  # 1 eV = 1/13.60569253 Rydberg
ANGSTROM_TO_BOHR = 1 / 0.529177210903  # 1 Å = 1/0.529177210903 bohr
FORCE_CONV = EV_TO_RYD / ANGSTROM_TO_BOHR  # eV/Å -> Ryd/bohr


def write2alm(
    super_cell, all_atoms, is_correct_with_spuer_cell=False, output_file="DFTSETS"
):
    """
    Write DFTSETS file in the format required by alamode software

    This function processes atomic structures and forces for use with alamode,
    a software package for analyzing lattice anharmonicity and lattice thermal
    conductivity. It converts units from eV/Å to Rydberg/bohr and calculates
    displacements relative to the supercell reference structure.

    Args:
        super_cell: Supercell structure (ASE Atoms object)
        all_atoms: List of atomic structures (ASE Atoms objects)
        is_correct_with_spuer_cell: Whether to correct atomic forces based on
                                   supercell reference (default: False)
        output_file: Output filename (default: "DFTSETS")

    Returns:
        None: Writes output to file

    Note:
        - Forces are converted from eV/Å to Rydberg/bohr units
        - Displacements are calculated relative to supercell positions
        - Displacements are converted from Ångström to bohr units
        - Output format is compatible with alamode DFTSETS requirements
    """
    # Initialize empty symbols array for alamode format
    symbols_alm = np.array([" " for _ in range(len(super_cell))])

    if is_correct_with_spuer_cell:
        # Correct forces by subtracting supercell forces and convert units
        for atoms in all_atoms:
            atoms.new_array("symbols_alm", symbols_alm)
            atoms.new_array(
                "forces", (atoms.get_forces() - super_cell.get_forces()) * FORCE_CONV
            )
            atoms.new_array(
                "displacements",
                (atoms.get_positions() - super_cell.get_positions()) * ANGSTROM_TO_BOHR,
            )
    else:
        # Use absolute forces and calculate displacements relative to supercell
        for atoms in all_atoms:
            atoms.new_array("symbols", symbols_alm)
            atoms.new_array("forces", atoms.get_forces() * FORCE_CONV)
            atoms.new_array(
                "displacements",
                (atoms.get_positions() - super_cell.get_positions()) * ANGSTROM_TO_BOHR,
            )

    # Write output in alamode-compatible format
    write_xyz(
        output_file,
        all_atoms,
        comment="# ",
        write_info=False,
        columns=["symbols_alm", "displacements", "forces"],
        is_index_comment=True,
        is_natoms=False,
    )
