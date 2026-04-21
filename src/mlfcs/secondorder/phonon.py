"""Phonon calculation using ASE Calculator."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
    phonopy_exists = True
except ModuleNotFoundError:
    phonopy_exists = False
    Phonopy = None

from ase import Atoms
from mlfcs.interface import ase_to_phonopy_atoms, phonopy_to_ase_atoms


class MLPHONON:
    """Harmonic phonon calculation using ASE Calculator."""

    def __init__(
        self,
        structure: Atoms,
        calculator,
        supercell_matrix: list | np.ndarray,
        kwargs_phonopy: Dict[str, Any] | None = None,
        kwargs_generate_displacements: Dict[str, Any] | None = None,
    ):
        """Init method.

        Parameters
        ----------
        structure : Atoms
            Structure for which to compute the phonon properties;
            usually this is a primitive cell.
        calculator : ASE Calculator
            ASE calculator to use for the calculation of forces.
        supercell_matrix : array_like
            Specification of supercell size handed over to phonopy;
            should be a tuple of three values or a 3x3 matrix.
        kwargs_phonopy : dict, optional
            Keyword arguments used when initializing the Phonopy object;
            includes, e.g., `symprec` for symmetry tolerance,
            `nac_params` for non-analytical corrections.
        kwargs_generate_displacements : dict, optional
            Keyword arguments passed to generate_displacements method;
            includes, e.g., `distance` for displacement magnitude.

        """
        if not phonopy_exists:
            raise ModuleNotFoundError(
                "phonopy (https://pypi.org/project/phonopy/) is "
                "required in order to use the functionality in the phonon module."
            )

        self._structure = structure
        self._calculator = calculator
        self._supercell_matrix = supercell_matrix

        if kwargs_phonopy is None:
            kwargs_phonopy = {}
        if kwargs_generate_displacements is None:
            kwargs_generate_displacements = {}

        self._kwargs_phonopy = kwargs_phonopy
        self._kwargs_generate_displacements = kwargs_generate_displacements

        self._phonon = None

    @property
    def phonopy(self) -> Phonopy:
        """Return Phonopy instance."""
        return self._phonon

    @property
    def force_constants(self) -> np.ndarray:
        """Return force constants."""
        if self._phonon is None:
            raise RuntimeError("Run calculation first.")
        return self._phonon.force_constants

    def run(self) -> "MLPHONON":
        """Run phonon calculation.

        Generates displacements, evaluates forces using the ASE calculator,
        and computes the force constant matrix.

        Returns
        -------
        MLPHONON
            Self for method chaining.

        """
        structure_ph = ase_to_phonopy_atoms(self._structure, include_masses=True)

        self._phonon = Phonopy(
            structure_ph, self._supercell_matrix, **self._kwargs_phonopy
        )
        self._phonon.generate_displacements(**self._kwargs_generate_displacements)

        forces = []
        for ph_atoms in self._phonon.supercells_with_displacements:
            atoms = phonopy_to_ase_atoms(ph_atoms)
            atoms.calc = self._calculator
            forces.append(atoms.get_forces().copy())

        self._phonon.forces = forces
        self._phonon.produce_force_constants()

        return self

    def write(self, filename: str = "FORCE_CONSTANTS", fmt: str = "auto"):
        """Write force constants to file.

        Parameters
        ----------
        filename : str
            Output filename. Default is "FORCE_CONSTANTS".
        fmt : str
            Output format: "text", "hdf5", or "auto" (infer from filename).
            Default is "auto".

        """
        if self._phonon is None:
            raise RuntimeError("Run calculation first.")

        resolved = fmt.lower()
        if resolved == "auto":
            resolved = "hdf5" if filename.lower().endswith((".hdf5", ".h5")) else "text"

        if resolved == "text":
            write_FORCE_CONSTANTS(self._phonon.force_constants, filename=filename)
            return
        if resolved == "hdf5":
            write_force_constants_to_hdf5(self._phonon.force_constants, filename=filename)
            return

        raise ValueError(f"Unsupported fmt='{fmt}'. Use 'text', 'hdf5', or 'auto'.")
