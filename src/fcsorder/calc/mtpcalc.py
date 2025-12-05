#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ASE calculator interface for Moment Tensor Potential."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.cell import Cell


class MTP(Calculator):
    """ASE calculator for Moment Tensor Potential.

    This calculator provides an interface to the MLP (Machine Learning Potential)
    executable for computing energies, forces, and stresses using MTP models.

    Attributes:
        implemented_properties: List of properties that can be calculated.
        nolabel: Whether to use a label for the calculator.

    Args:
        mtp_path: Path to the MTP potential file. Default is "pot.mtp".
        mtp_exe: Path to the MLP executable. Default is "mlp".
        tmp_folder: Temporary folder for intermediate files. Default is system temp directory.
        unique_elements: List of unique element symbols in the system. Required for calculations.
        **kwargs: Additional arguments passed to the parent Calculator class.
    """

    implemented_properties = ["energy", "forces", "stress"]
    nolabel = True

    def __init__(
        self,
        mtp_path: str = "pot.mtp",
        mtp_exe: str = "mlp",
        tmp_folder: Optional[str] = None,
        unique_elements: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize the MTP calculator.

        Args:
            mtp_path: Path to the MTP potential file.
            mtp_exe: Path to the MLP executable.
            tmp_folder: Temporary folder for intermediate files.
            unique_elements: List of unique element symbols.
            **kwargs: Additional arguments for the parent Calculator.
        """
        self._mtp_path = mtp_path
        self._mtp_exe = mtp_exe
        self._tmp_folder = tmp_folder or tempfile.gettempdir()
        self._unique_elements = unique_elements
        self._numbers = None
        Calculator.__init__(self, **kwargs)

    def initialize(self, atoms) -> None:
        """Initialize the calculator with atomic numbers.

        Args:
            atoms: ASE Atoms object.
        """
        self._numbers = atoms.get_atomic_numbers()

    def calculate(
        self,
        atoms=None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """Calculate energy, forces, and stress.

        Args:
            atoms: ASE Atoms object.
            properties: List of properties to calculate.
            system_changes: List of system changes.
        """
        if properties is None:
            properties = ["energy"]
        if system_changes is None:
            system_changes = all_changes
        
        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        if self._unique_elements is None:
            raise ValueError("unique_elements must be specified")

        input_file = os.path.join(self._tmp_folder, "in.cfg")
        output_file = os.path.join(self._tmp_folder, "out.cfg")

        atoms_to_cfg(atoms.copy(), input_file, self._unique_elements)

        subprocess.run(
            [self._mtp_exe, "calc-efs", self._mtp_path, input_file, output_file],
            check=True,
            capture_output=True,
        )

        energy, forces, stress = read_cfg(output_file, self._unique_elements)
        self.results["energy"] = energy
        self.results["forces"] = np.array(forces)
        self.results["stress"] = np.array(stress)


def atoms_to_cfg(atoms, file: str, unique_elements: list[str]) -> None:
    """Convert ASE Atoms object to MTP CFG format.

    Args:
        atoms: ASE Atoms object to convert.
        file: Output file path.
        unique_elements: List of unique element symbols.
    """
    has_forces = True
    has_energy = True

    try:
        energy = atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        has_energy = False

    try:
        forces = atoms.get_forces()
    except (RuntimeError, AttributeError):
        has_forces = False

    with open(file, "w") as f:
        element_dict = {ele.capitalize(): int(i) for i, ele in enumerate(unique_elements)}

        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        num_atoms = len(atoms)
        f.write(f"    {int(num_atoms)}\n")
        f.write(" Supercell\n")
        cell = atoms.get_cell()
        f.write(
            "{0:<9}{1}      {2}      {3}\n".format(
                "", cell[0][0], cell[0][1], cell[0][2]
            )
        )
        f.write(
            "{0:<9}{1}      {2}      {3}\n".format(
                "", cell[1][0], cell[1][1], cell[1][2]
            )
        )
        f.write(
            "{0:<9}{1}      {2}      {3}\n".format(
                "", cell[2][0], cell[2][1], cell[2][2]
            )
        )

        if has_forces:
            f.write(
                " AtomData:  id type       cartes_x      cartes_y"
                "      cartes_z           fx          fy          fz\n"
            )
        else:
            f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z\n")

        positions = atoms.positions
        symbols = atoms.symbols
        for i in range(num_atoms):
            atom_id = int(i + 1)
            atom_type = element_dict[symbols[i]]
            x, y, z = positions[i]
            if has_forces:
                f_x, f_y, f_z = forces[i]
                f.write(
                    "{0:>14}{1:>5}{2:>16.8f}{3:>16.8f}{4:>16.8f}{5:>12.6f}{6:>12.6f}{7:>12.6f}\n".format(
                        atom_id, atom_type, x, y, z, f_x, f_y, f_z
                    )
                )
            else:
                f.write(
                    "{0:>14}{1:>5}{2:>16.8f}{3:>16.8f}{4:>16.8f}\n".format(
                        atom_id, atom_type, x, y, z
                    )
                )

        if has_energy:
            f.write(" Energy\n")
            f.write(f"{energy:16.6f}\n")
        f.write("END_CFG\n")
        f.write("\n")


def read_cfg(file: str, symbols: list[str]) -> tuple[float, list, np.ndarray]:
    """Read MTP CFG format file and extract energy, forces, and stress.

    Adapted from `mlearn` package: https://github.com/materialsvirtuallab/mlearn

    Args:
        file: Path to CFG file.
        symbols: List of unique element symbols (unused but kept for compatibility).

    Returns:
        Tuple of (energy, forces, virial_stress) where:
            - energy: Total energy (float)
            - forces: Atomic forces (list)
            - virial_stress: Virial stress tensor (np.ndarray with 6 components)
    """
    with open(file, "r") as f:
        content = f.read()

    block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
    lattice_pattern = re.compile("SuperCell\n(.*?)\n AtomData", re.S | re.I)
    position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
    energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
    stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)

    def parse_floats(string: str) -> list[float]:
        """Parse space-separated floats from string."""
        return [float(s) for s in string.split()]

    for block in block_pattern.findall(content):
        lattice_str = lattice_pattern.findall(block)[0]
        lattice = np.array(list(map(parse_floats, lattice_str.split("\n"))))
        cell = Cell(lattice)
        cell_volume = cell.volume

        position_str = position_pattern.findall(block)[0]
        position_data = np.array(list(map(parse_floats, position_str.split("\n"))))
        forces = position_data[:, 5:8].tolist()

        energy_str = energy_pattern.findall(block)[0]
        energy = float(energy_str.lstrip())

        stress_str = stress_pattern.findall(block)[0]
        virial_stress = (
            -np.array(list(map(parse_floats, stress_str.split()))).reshape(
                6,
            )
            / cell_volume
        )

    return energy, forces, virial_stress
