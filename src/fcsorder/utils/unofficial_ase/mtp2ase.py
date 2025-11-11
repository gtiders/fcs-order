"""ASE calculator interface for Moment Tensor Potential."""

import os
import re
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.cell import Cell


class MTP(Calculator):
    """ASE calculator for Moment Tensor Potential.

    Parameters
    ----------
    mtp_path : str, optional
        Path to the MTP potential file. Default is "pot.mtp".
    mtp_exe : str, optional
        Path to the MLP executable. Default is "mlp".
    tmp_folder : str, optional
        Temporary folder for intermediate files. Default is system temp directory.
    unique_elements : list, optional
        List of unique element symbols in the system. Required for calculations.
    """

    implemented_properties = ["energy", "forces", "stress"]
    nolabel = True

    def __init__(
        self,
        mtp_path="pot.mtp",
        mtp_exe="mlp",
        tmp_folder=None,
        unique_elements=None,
        **kwargs,
    ):
        self.mtp_path = mtp_path
        self.mtp_exe = mtp_exe
        self.tmp_folder = tmp_folder or tempfile.gettempdir()
        self.unique_elements = unique_elements
        Calculator.__init__(self, **kwargs)

    def initialize(self, atoms):
        self.numbers = atoms.get_atomic_numbers()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        if self.unique_elements is None:
            raise ValueError("unique_elements must be specified")

        input_file = os.path.join(self.tmp_folder, "in.cfg")
        output_file = os.path.join(self.tmp_folder, "out.cfg")

        atoms_to_cfg(atoms.copy(), input_file, self.unique_elements)

        subprocess.run(
            [self.mtp_exe, "calc-efs", self.mtp_path, input_file, output_file],
            check=True,
            capture_output=True,
        )

        energy, forces, stress = read_cfg(output_file, self.unique_elements)
        self.results["energy"] = energy
        self.results["forces"] = np.array(forces)
        self.results["stress"] = np.array(stress)


def atoms_to_cfg(atoms, file, unique_elements):
    """Convert ASE Atoms object to MTP CFG format.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to convert.
    file : str
        Output file path.
    unique_elements : list
        List of unique element symbols.
    """
    write_f, write_e = True, True

    try:
        e = atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        write_e = False

    try:
        fs = atoms.get_forces()
    except (RuntimeError, AttributeError):
        write_f = False

    with open(file, "w") as f:
        ele_dict = {ele.capitalize(): int(i) for i, ele in enumerate(unique_elements)}

        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        size = len(atoms)
        f.write(f"    {int(size)}\n")
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

        if write_f:
            f.write(
                " AtomData:  id type       cartes_x      cartes_y"
                "      cartes_z           fx          fy          fz\n"
            )
        else:
            f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z\n")

        pos = atoms.positions
        symbols = atoms.symbols
        for i in range(size):
            aid = int(i + 1)
            atype = ele_dict[symbols[i]]
            x, y, z = pos[i]
            if write_f:
                f_x, f_y, f_z = fs[i]
                f.write(
                    "{0:>14}{1:>5}{2:>16.8f}{3:>16.8f}{4:>16.8f}{5:>12.6f}{6:>12.6f}{7:>12.6f}\n".format(
                        aid, atype, x, y, z, f_x, f_y, f_z
                    )
                )
            else:
                f.write(
                    "{0:>14}{1:>5}{2:>16.8f}{3:>16.8f}{4:>16.8f}\n".format(
                        aid, atype, x, y, z
                    )
                )

        if write_e:
            f.write(" Energy\n")
            f.write(f"{e:16.6f}\n")
        f.write("END_CFG\n")
        f.write("\n")


def read_cfg(file, symbols):
    """Read MTP CFG format file and extract energy, forces, and stress.

    Adapted from `mlearn` package: https://github.com/materialsvirtuallab/mlearn

    Parameters
    ----------
    file : str
        Path to CFG file.
    symbols : list
        List of unique element symbols (unused but kept for compatibility).

    Returns
    -------
    energy : float
        Total energy.
    forces : list
        Atomic forces.
    virial_stress : np.ndarray
        Virial stress tensor (6 components).
    """
    with open(file, "r") as f:
        lines = f.read()

    block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
    lattice_pattern = re.compile("SuperCell\n(.*?)\n AtomData", re.S | re.I)
    position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
    energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
    stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)

    def formatify(string):
        return [float(s) for s in string.split()]

    for block in block_pattern.findall(lines):
        lattice_str = lattice_pattern.findall(block)[0]
        lattice = np.array(list(map(formatify, lattice_str.split("\n"))))
        cell = Cell(lattice)
        volume = cell.volume

        position_str = position_pattern.findall(block)[0]
        position = np.array(list(map(formatify, position_str.split("\n"))))
        forces = position[:, 5:8].tolist()

        energy_str = energy_pattern.findall(block)[0]
        energy = float(energy_str.lstrip())

        stress_str = stress_pattern.findall(block)[0]
        virial_stress = (
            -np.array(list(map(formatify, stress_str.split()))).reshape(
                6,
            )
            / volume
        )

    return energy, forces, virial_stress
