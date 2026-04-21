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
from ase import Atoms
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
from phonopy.physical_units import get_physical_units

from mlfcs.interface import AsePhonopyMLP, ase_to_phonopy_atoms


class MLPSSCHA:
    """Iterative approach SSCHA using ASE calculator via MLP adapter."""

    def __init__(
        self,
        unitcell: Atoms,
        calculator,
        supercell_matrix: list | np.ndarray | None = None,
        temperature: float | None = None,
        number_of_snapshots: int | Literal["auto"] | None = None,
        max_iterations: int | None = None,
        distance: float | None = None,
        fc_calculator: str | None = None,
        fc_output: str = "FORCE_CONSTANTS",
        fc_output_format: Literal["text", "hdf5", "auto"] = "auto",
        avg_n_last_steps: int | None = None,
        log_level: int = 0,
    ):
        """Init method.

        unitcell : Atoms
            ASE Atoms object representing primitive cell.
        calculator : ASE Calculator
            ASE Calculator instance (e.g. CPUNEP, MACE, etc.)
        supercell_matrix : array_like, optional
            Supercell matrix. Required for Atoms input.
        """
        if calculator is None:
            raise ValueError("ASE Calculator is not provided.")

        # ===== MLFCS MOD BEGIN =====
        # Keep library API in ASE style and build Phonopy internally.
        if not isinstance(unitcell, Atoms):
            raise TypeError("unitcell must be ASE Atoms.")
        if supercell_matrix is None:
            raise ValueError("supercell_matrix is required when input is ASE Atoms.")
        unitcell_ph = ase_to_phonopy_atoms(unitcell, include_masses=True)
        self._ph = Phonopy(unitcell_ph, supercell_matrix=supercell_matrix)
        self._ph.mlp = AsePhonopyMLP(calculator)
        # ===== MLFCS MOD END =====

        # ===== MLFCS MOD BEGIN =====
        # Optional output controls retained for backward compatibility in mlfcs.
        self._fc_output = fc_output
        self._fc_output_format = fc_output_format
        self._avg_n_last_steps = avg_n_last_steps
        # ===== MLFCS MOD END =====

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
        if distance is None:
            self._distance = 0.01
        else:
            self._distance = distance
        if fc_calculator is None:
            self._fc_calculator = "symfc"
        else:
            self._fc_calculator = fc_calculator
        self._log_level = log_level

        self._ph.nac_params = copy.deepcopy(self._ph.nac_params)

        # Calculate supercell energy without displacements
        self._ph.generate_displacements(distance=0, number_of_snapshots=1)
        self._ph.evaluate_mlp()
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

    def calculate_free_energy(self, mesh: float = 100.0) -> None:
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
        # ===== MLFCS MOD BEGIN =====
        # Keep free-energy history print and optional FC averaging/output behavior.
        green = "\033[32m"
        reset = "\033[0m"
        free_energies = []
        fc_history = []
        for i in self:
            self.calculate_free_energy()
            fe = float(self.free_energy)
            free_energies.append(fe)
            print(f"{green}[SSCHA] Iteration {i + 1}: Free Energy = {fe:.6f} eV{reset}")
            if self._ph.force_constants is not None:
                fc_history.append(self._ph.force_constants.copy())
            if self._log_level:
                print("")

        print(f"\n{green}[SSCHA] Final Free Energy History:{reset}")
        print(f"{green}{free_energies}{reset}\n")

        if self._avg_n_last_steps and self._avg_n_last_steps > 0 and fc_history:
            n_avg = min(self._avg_n_last_steps, len(fc_history))
            print(
                f"{green}[SSCHA] Averaging Force Constants over last {n_avg} steps...{reset}"
            )
            self._ph.force_constants = np.mean(fc_history[-n_avg:], axis=0)

        if self._fc_output:
            if self._log_level:
                print(f"Writing final force constants to {self._fc_output}...")
            out_fmt = self._fc_output_format
            if out_fmt == "auto":
                out_fmt = (
                    "hdf5"
                    if self._fc_output.lower().endswith((".hdf5", ".h5"))
                    else "text"
                )
            if out_fmt == "text":
                write_FORCE_CONSTANTS(self._ph.force_constants, filename=self._fc_output)
            elif out_fmt == "hdf5":
                write_force_constants_to_hdf5(
                    self._ph.force_constants, filename=self._fc_output
                )
            else:
                raise ValueError(
                    f"Unsupported fc_output_format='{self._fc_output_format}'. "
                    "Use 'text', 'hdf5', or 'auto'."
                )
        # ===== MLFCS MOD END =====
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

    def _run(self) -> None:
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
            print("Evaluate MLP to obtain forces", flush=True)
        self._ph.evaluate_mlp()

        if self._log_level:
            print("Calculate force constants using symfc", flush=True)
        self._ph.produce_force_constants(
            fc_calculator="symfc",
            fc_calculator_log_level=self._log_level if self._log_level > 1 else 0,
            calculate_full_force_constants=True,
            show_drift=False,
        )
