#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write


def to_internal(atoms: Atoms) -> dict:
    """
    Convert ASE Atoms to the internal structure dict used by fcs-order.
    Keys: lattvec(3x3, nm), elements(list[str] order by groups), numbers(np.array[int]),
          positions(3xN fractional), types(list[int])
    """
    nruter: dict = {}
    # fcs-order uses nm units (order_common.read_POSCAR multiplies by 0.1),
    # ASE uses Angstrom for cell; 1 Angstrom = 0.1 nm
    nruter["lattvec"] = 0.1 * atoms.get_cell().T

    chemical_symbols = atoms.get_chemical_symbols()
    # Group consecutive same symbols like POSCAR species blocks
    elements: list[str] = []
    numbers: list[int] = []
    last = None
    count = 0
    for s in chemical_symbols:
        if last is None:
            last = s
            count = 1
        elif s == last:
            count += 1
        else:
            elements.append(last)
            numbers.append(count)
            last = s
            count = 1
    if last is not None:
        elements.append(last)
        numbers.append(count)

    nruter["elements"] = elements
    nruter["numbers"] = np.array(numbers, dtype=np.intc)

    positions = atoms.get_scaled_positions()
    nruter["positions"] = np.asarray(positions).T

    nruter["types"] = np.repeat(
        range(len(nruter["numbers"])), nruter["numbers"]
    ).tolist()

    return nruter


def from_internal(poscar: dict) -> Atoms:
    """
    Convert internal dict to ASE Atoms.
    """
    symbols = np.repeat(poscar["elements"], poscar["numbers"]).tolist()
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=poscar["positions"].T,
        cell=poscar["lattvec"].T * 10.0,  # nm -> Angstrom
        pbc=True,
    )
    return atoms


def read_structure(path: str, in_format: str = "auto") -> dict:
    """
    Read a structure file using ASE (autodetect by extension) and return internal dict.
    """
    fmt = None if in_format == "auto" else in_format
    atoms = ase_read(path, format=fmt)
    return to_internal(atoms)


def write_structure(poscar: dict, filename: str, out_format: str = "vasp") -> None:
    """
    Write internal dict into a target format using ASE when possible.
    Supported out_format:
      - "vasp" (alias: "poscar"): VASP POSCAR, direct coordinates
      - "cif": CIF
      - "xyz": XYZ
    """
    out_format = out_format.lower()
    if out_format in ("vasp", "poscar"):
        # Keep identical behavior with write_POSCAR
        atoms = from_internal(poscar)
        # Ensure direct coordinates for VASP
        ase_write(filename, atoms, format="vasp", direct=True)
        return

    atoms = from_internal(poscar)
    ext = Path(filename).suffix.lower()
    if out_format == "cif" and ext != ".cif":
        filename = f"{filename}.cif" if ext == "" else filename
    if out_format == "xyz" and ext != ".xyz":
        filename = f"{filename}.xyz" if ext == "" else filename

    ase_write(filename, atoms, format=out_format)


def read_atoms(path: str, in_format: str = "auto") -> Atoms:
    """
    Read a structure file and return ASE Atoms directly (single conversion path).
    Equivalent to ase.io.read with optional explicit format, but routed through our
    abstraction to keep behavior consistent.
    """
    fmt = None if in_format == "auto" else in_format
    return ase_read(path, format=fmt)


def get_atoms(poscar: dict, calc: Optional[object] = None) -> Atoms:
    """
    Construct ASE Atoms from an internal dict; optional calculator attachment.
    This mirrors the prior utils.atoms.get_atoms behavior, centralized here.
    """
    atoms = from_internal(poscar)
    if calc is not None:
        atoms.calc = calc
    return atoms


def parse_supercell_matrix(supercell_matrix):
    """
    Parse a 3- or 9-element supercell specification into a 3x3 numpy array of ints.
    """
    if len(supercell_matrix) not in (3, 9):
        raise ValueError("Supercell matrix must have 3 (diagonal) or 9 (3x3) integers")
    if len(supercell_matrix) == 3:
        na, nb, nc = supercell_matrix
        return np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]], dtype=int)
    return np.array(supercell_matrix, dtype=int).reshape(3, 3)
