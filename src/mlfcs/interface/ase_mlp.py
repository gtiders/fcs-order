"""ASE calculator adapter implementing phonopy MLP evaluate protocol."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlfcs.interface.phonopy_bridge import phonopy_to_ase_atoms


class AsePhonopyMLP:
    """Expose ASE calculator through a PhonopyMLP-like evaluate() interface."""

    def __init__(self, calculator):
        self._calculator = calculator

    @property
    def mlp(self):
        """Compatibility property used by some phonopy call sites."""
        return self

    def evaluate(
        self, supercells_with_displacements: Sequence
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        energies = []
        forces = []
        natoms = None

        for ph_atoms in supercells_with_displacements:
            if ph_atoms is None:
                raise RuntimeError("Encountered None supercell in MLP evaluation.")

            atoms = phonopy_to_ase_atoms(ph_atoms)
            atoms.calc = self._calculator

            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            if natoms is None:
                natoms = len(atoms)

        if natoms is None:
            raise RuntimeError("No supercells to evaluate in MLP adapter.")

        stresses = np.zeros((len(energies), 6), dtype=float)
        return (
            np.asarray(energies, dtype=float),
            np.asarray(forces, dtype=float),
            stresses,
        )
