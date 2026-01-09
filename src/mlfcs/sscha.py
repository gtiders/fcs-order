"""SSCHA calculation."""

# Copyright (C) 2024 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import copy
from typing import Literal

import numpy as np

from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS
from ase import Atoms
from ase.io import read
from pathlib import Path


def phonopy_to_ase(atoms, **kwargs):
    """Convert PhonopyAtoms to ASE Atoms (from calorine logic)."""
    return Atoms(
        cell=atoms.cell,
        numbers=atoms.numbers,
        positions=atoms.positions,
        pbc=True,
        **kwargs,
    )


def ase_to_phonopy(atoms):
    """Convert ASE Atoms to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(), cell=atoms.cell, positions=atoms.positions
    )


class MLPSSCHA:
    """Iterative approach SSCHA using ASE Calculator (replacing PhonopyMLP)."""

    def __init__(
        self,
        unitcell: Phonopy | Atoms | str | Path,
        calculator,  # Changed from mlp: PhonopyMLP to calculator (ASE)
        supercell_matrix: list | np.ndarray | None = None,
        temperature: float | None = None,
        number_of_snapshots: int | Literal["auto"] | None = None,
        max_iterations: int | None = None,
        distance: float | None = None,
        fc_calculator: str | None = None,
        fc_output: str = "FORCE_CONSTANTS",
        avg_n_last_steps: int | None = None,  # New parameter
        log_level: int = 0,
    ):
        """Init method.

        unitcell : Phonopy | Atoms | str | Path
            Phonopy instance OR ASE Atoms object OR path to structure file.
        calculator : ASE Calculator
            ASE Calculator instance (e.g. CPUNEP, Mace, etc.)
        supercell_matrix : array_like, optional
            Supercell matrix. Required if unitcell is ASE Atoms.
        temperature : float, optional
            Temperature in K, by default 300.0.
        number_of_snapshots : int, optional
            Number of snapshots, by default 1000.
        max_iterations : int, optional
            Maximum number of iterations, by default 10.
        distance : float, optional
            Distance of displacements, by default is None, which gives 0.01.
        fc_calculator : str, optional
            Force constants calculator. The default is None, which means "symfc".
        fc_output : str, optional
            Filename to write final force constants. Default "FORCE_CONSTANTS".
        avg_n_last_steps : int, optional
            Number of last iterations to average force constants over. Default None (no averaging).
        log_level : int, optional
            Log level, by default 0.

        """
        if calculator is None:
            raise ValueError("ASE Calculator is not provided.")

        self._calculator = calculator
        self._fc_output = fc_output
        self._avg_n_last_steps = avg_n_last_steps

        # If unitcell is a path, read it
        if isinstance(unitcell, (str, Path)):
            unitcell = read(unitcell)

        # Initialize Phonopy object
        if isinstance(unitcell, Phonopy):
            self._ph = unitcell.copy()
        elif isinstance(unitcell, Atoms):
            if supercell_matrix is None:
                raise ValueError(
                    "supercell_matrix is required when input is ASE Atoms."
                )
            unitcell_ph = ase_to_phonopy(unitcell)
            self._ph = Phonopy(unitcell_ph, supercell_matrix=supercell_matrix)
        else:
            raise TypeError(
                "unitcell must be Phonopy object, ASE Atoms object, or file path."
            )

        if temperature is None:
            self._temperature = 300.0
        else:
            self._temperature = temperature
        if number_of_snapshots is None:
            self._number_of_snapshots = 1000
        else:
            self._number_of_snapshots = number_of_snapshots
        if max_iterations is None:
            self._max_iterations = 10
        else:
            self._max_iterations = max_iterations
        self._max_iterations = max_iterations
        if distance is None:
            self._distance = 0.01
        else:
            self._distance = distance
        if fc_calculator is None:
            self._fc_calculator = "symfc"
        else:
            self._fc_calculator = fc_calculator
        self._log_level = log_level

        # self._ph.mlp = PhonopyMLP(mlp=mlp.mlp) # Removed dependency
        self._ph.nac_params = copy.deepcopy(self._ph.nac_params)

        # Calculate supercell energy without displacements
        self._ph.generate_displacements(distance=0, number_of_snapshots=1)
        self.evaluate_structure()

        self._supercell_energy = float(self._ph.supercell_energies[0])
        self._ph.dataset = None

        if self._ph.force_constants is None:
            self._iter_counter = 0
        else:
            if log_level:
                print("Use provided force constants.")
                print("")
            self._iter_counter = 1

    @property
    def phonopy(self) -> Phonopy:
        """Return Phonopy instance."""
        return self._ph

    @property
    def free_energy(self) -> float:
        """Return free energy in eV."""
        return self._free_energy

    @property
    def force_constants(self) -> np.ndarray:
        """Return force constants."""
        return self._ph.force_constants

    @property
    def harmonic_potential_energy(self) -> float:
        """Return supercell energies."""
        d = self._ph.displacements
        pe = np.einsum("ijkl,mik,mjl", self.force_constants, d, d) / len(d) / 2
        return pe

    @property
    def potential_energy(self) -> float:
        """Return potential energy."""
        return np.average(self._ph.supercell_energies - self._supercell_energy)

    def calculate_free_energy(self, mesh: float = 100.0) -> float:
        """Calculate SSCHA free energy."""
        self._ph.run_mesh(mesh=mesh)
        self._ph.run_thermal_properties(temperatures=[self._temperature])
        hfe = (
            self._ph.get_thermal_properties_dict()["free_energy"][0]
            / get_physical_units().EvTokJmol
        )
        n_cell = len(self._ph.supercell) / len(self._ph.primitive)
        pe = self.potential_energy / n_cell
        hpe = self.harmonic_potential_energy / n_cell
        self._free_energy = hfe + pe - hpe

    def run(self) -> "MLPSSCHA":
        """Run through all iterations."""
        GREEN = "\033[32m"
        RESET = "\033[0m"

        free_energies = []
        fc_history = []  # Store FCs for averaging

        for i in self:
            # Calculate and print Free Energy
            self.calculate_free_energy()
            fe = float(self.free_energy)
            free_energies.append(fe)
            print(f"{GREEN}[SSCHA] Iteration {i + 1}: Free Energy = {fe:.6f} eV{RESET}")

            # Store current FC
            fc_history.append(self._ph.force_constants.copy())

            if self._log_level:
                print("")

        print(f"\n{GREEN}[SSCHA] Final Free Energy History:{RESET}")
        print(f"{GREEN}{free_energies}{RESET}\n")

        # Perform Averaging if requested
        if self._avg_n_last_steps and self._avg_n_last_steps > 0:
            n_avg = min(self._avg_n_last_steps, len(fc_history))
            if n_avg > 0:
                print(
                    f"{GREEN}[SSCHA] Averaging Force Constants over last {n_avg} steps...{RESET}"
                )
                fcs_slice = fc_history[-n_avg:]
                fc_avg = np.mean(fcs_slice, axis=0)
                self._ph.force_constants = fc_avg

        # Save final force constants
        if self._fc_output:
            if self._log_level:
                print(
                    f"Writing final force constants (possibly averaged) to {self._fc_output}..."
                )
            write_FORCE_CONSTANTS(self._ph.force_constants, filename=self._fc_output)

        return self

    def __iter__(self) -> "MLPSSCHA":
        """Iterate over force constants calculations."""
        return self

    def __next__(self) -> int:
        """Calculate next force constants."""
        if self._iter_counter == self._max_iterations + 1:
            self._iter_counter = 0
            raise StopIteration
        self._run()
        self._iter_counter += 1
        return self._iter_counter - 1

    def _run(self) -> Phonopy:
        if self._log_level and self._iter_counter == 0:
            print(
                f"[ SSCHA initialization (rd={self._distance}, "
                f"n_supercells={self._number_of_snapshots}) ]",
                flush=True,
            )
        if self._log_level and self._iter_counter > 0:
            print(f"[ SSCHA iteration {self._iter_counter} / {self._max_iterations} ]")
            print(
                f"Generate {self._number_of_snapshots} supercells with displacements "
                f"at {self._temperature} K",
                flush=True,
            )

        if self._iter_counter == 0:
            self._ph.generate_displacements(
                distance=self._distance, number_of_snapshots=self._number_of_snapshots
            )
        else:
            self._ph.generate_displacements(
                number_of_snapshots=self._number_of_snapshots,
                temperature=self._temperature,
            )
            hist, bin_edges = np.histogram(
                np.linalg.norm(self._ph.displacements, axis=2), bins=10
            )

            if self._log_level:
                size = np.prod(self._ph.displacements.shape[0:2])
                for i, h in enumerate(hist):
                    length = round(h / size * 100)
                    print(
                        f"  [{bin_edges[i]:4.3f}, {bin_edges[i + 1]:4.3f}] "
                        + "*" * length
                    )

        if self._log_level:
            print("Evaluate MLP (via ASE Calculator) to obtain forces...", flush=True)

        self.evaluate_structure()

        if self._log_level:
            print("Calculate force constants using symfc", flush=True)
        self._ph.produce_force_constants(
            fc_calculator="symfc",
            fc_calculator_log_level=self._log_level if self._log_level > 1 else 0,
            calculate_full_force_constants=True,
            show_drift=False,
        )

    def evaluate_structure(self):
        """Evaluate structures using ASE Calculator.

        Calculates supercell energies and forces for the displacements in
        self._ph.supercells_with_displacements using the provided ASE calculator.
        Results are stored into self._ph.supercell_energies and self._ph.dataset["forces"].
        """
        if self._calculator is None:
            raise RuntimeError("ASE Calculator is not set.")

        supercells = self._ph.supercells_with_displacements
        if supercells is None:
            raise RuntimeError("Displacements are not set. Run generate_displacements.")

        energies = []
        forces = []

        for i, ph_atoms in enumerate(supercells):
            if ph_atoms is None:
                continue

            # Convert PhonopyAtoms -> ASE Atoms
            atoms = phonopy_to_ase(ph_atoms)

            # Attach calculator and compute
            atoms.calc = self._calculator

            e = atoms.get_potential_energy()
            f = atoms.get_forces()

            energies.append(e)
            forces.append(f)

        # Update Phonopy object with results
        self._ph.supercell_energies = np.array(energies)

        # Phonopy expects forces in dataset dict
        if self._ph.dataset is None:
            self._ph.dataset = {}

        self._ph.dataset["forces"] = np.array(forces)
