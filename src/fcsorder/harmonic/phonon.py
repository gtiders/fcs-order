#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports

# Third-party imports
import numpy as np
import typer
from ase.calculators.calculator import Calculator

from ase import Atoms

from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS

from fcsorder.calc.calculators import CalculatorFactory
from fcsorder.io.reader import StructureData


def get_2fcs(
    structure: StructureData,
    supercell_structure:StructureData,
    calculator: Calculator,
    supercell_matrix: np.ndarray,
) -> Phonopy:

    # prepare supercells
    phonon = Phonopy(structure)
    phonon.generate_displacements(**kwargs_generate_displacements)

    # compute force constant matrix
    forces = []
    for structure_ph in phonon.supercells_with_displacements:
        structure_ase = phonopy_to_ase(structure_ph)
        structure_ase.calc = calculator
        forces.append(structure_ase.get_forces().copy())

    phonon.forces = forces
    phonon.produce_force_constants()

    return phonon
