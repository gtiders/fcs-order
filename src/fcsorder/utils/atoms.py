#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


def get_atoms(poscar: dict, calc: Optional[Calculator] = None) -> Atoms:
    """
    Construct an ASE Atoms from an internal POSCAR-like dict used by fcsorder.

    The dict is expected to contain keys: 'elements', 'numbers', 'positions', 'lattvec'.
    Positions are assumed fractional (scaled); lattice vectors in nm, converted to Å by *10.
    This function mirrors existing logic in mlp3.py/mlp4.py.
    """
    symbols = np.repeat(poscar["elements"], poscar["numbers"]).tolist()
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=poscar["positions"].T,
        cell=poscar["lattvec"].T * 10,
        pbc=True,
    )
    if calc is not None:
        atoms.calc = calc
    return atoms
