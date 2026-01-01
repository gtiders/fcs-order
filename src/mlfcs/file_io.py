#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
from ase import Atoms
import ase.io


class StructureData:
    """
    Data structure class to represent crystal structure with conversion between
    ASE Atoms and internal fcs-order format.

    Internal format dict keys:
      - lattvec: 3x3 lattice vectors in nm (transposed from ASE cell)
      - elements: list[str] of element symbols (grouped by consecutive blocks)
      - numbers: np.array[int] of atom counts per element
      - positions: 3xN fractional coordinates (transposed from ASE scaled_positions)
      - types: list[int] mapping each atom to element index
    """

    def __init__(self, atoms: Optional[Atoms] = None, data_dict: Optional[dict] = None):
        """
        Initialize StructureData from either ASE Atoms or internal dict.

        Args:
            atoms: ASE Atoms object
            data_dict: Internal format dict (lattvec, elements, numbers, positions, types)

        Raises:
            ValueError: If neither atoms nor data_dict is provided, or both are provided.
        """
        if atoms is not None and data_dict is not None:
            raise ValueError("Provide either atoms or data_dict, not both")
        if atoms is None and data_dict is None:
            raise ValueError("Must provide either atoms or data_dict")

        if atoms is not None:
            self.atoms = atoms
            self._data_dict = None
        else:
            self.atoms = None
            self._data_dict = data_dict

        self._calc = None

    def to_dict(self) -> dict:
        """
        Convert to internal fcs-order format dict.

        Returns:
            dict with keys: lattvec, elements, numbers, positions, types
        """
        if self._data_dict is not None:
            return self._data_dict.copy()

        # Convert from ASE Atoms to internal format
        atoms = self.atoms
        result: dict = {}

        # fcs-order uses nm units; ASE uses Angstrom (1 Angstrom = 0.1 nm)
        result["lattvec"] = 0.1 * atoms.get_cell().T

        # Group consecutive same symbols like POSCAR species blocks
        chemical_symbols = atoms.get_chemical_symbols()
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

        result["elements"] = elements
        result["numbers"] = np.array(numbers, dtype=np.intc)

        positions = atoms.get_scaled_positions()
        result["positions"] = np.asarray(positions).T

        result["types"] = np.repeat(
            range(len(result["numbers"])), result["numbers"]
        ).tolist()

        return result

    @classmethod
    def from_dict(cls, data_dict: dict) -> StructureData:
        """
        Create StructureData from internal format dict.

        Args:
            data_dict: Internal format dict (lattvec, elements, numbers, positions, types)

        Returns:
            StructureData instance
        """
        # Use types list to correctly map each atom to its element symbol
        # This handles atoms in any order, not just grouped by element type
        types = data_dict["types"]
        elements = data_dict["elements"]
        symbols = [elements[t] for t in types]

        atoms = Atoms(
            symbols=symbols,
            scaled_positions=data_dict["positions"].T,
            cell=data_dict["lattvec"].T * 10.0,  # nm -> Angstrom
            pbc=True,
        )
        return cls(atoms=atoms)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> StructureData:
        """
        Create StructureData from ASE Atoms object.

        Args:
            atoms: ASE Atoms object

        Returns:
            StructureData instance
        """
        return cls(atoms=atoms)

    def to_atoms(self) -> Atoms:
        """
        Get ASE Atoms object. Converts from dict if needed.

        Returns:
            ASE Atoms object
        """
        if self.atoms is not None:
            return self.atoms

        # Convert from dict to Atoms using types list for correct symbol mapping
        types = self._data_dict["types"]
        elements = self._data_dict["elements"]
        symbols = [elements[t] for t in types]

        atoms = Atoms(
            symbols=symbols,
            scaled_positions=self._data_dict["positions"].T,
            cell=self._data_dict["lattvec"].T * 10.0,  # nm -> Angstrom
            pbc=True,
        )
        self.atoms = atoms
        return atoms

    def to_file(self, filename: str, out_format: str = "vasp") -> None:
        """
        Write structure to file using ASE.

        Supported out_format:
          - "vasp" (alias: "poscar"): VASP POSCAR with direct coordinates
          - "cif": CIF format
          - "xyz": XYZ format

        Args:
            filename: Output file path (without extension, will be added automatically)
            out_format: Output format specification (default: "vasp")
        """
        out_format = out_format.lower()
        atoms = self.to_atoms()

        # Map format to file extension
        format_ext_map = {
            "vasp": "vasp",
            "poscar": "vasp",
            "cif": "cif",
            "xyz": "xyz",
        }

        if out_format not in format_ext_map:
            raise ValueError(
                f"Unsupported format '{out_format}'. "
                f"Must be one of: {', '.join(sorted(format_ext_map.keys()))}"
            )

        # Add extension if not already present
        ext = Path(filename).suffix.lower()
        target_ext = f".{format_ext_map[out_format]}"
        if ext != target_ext:
            filename = f"{filename}{target_ext}"

        # Write file
        if out_format in ("vasp", "poscar"):
            # Ensure direct coordinates for VASP
            ase.io.write(filename, atoms, format="vasp", direct=True)
        else:
            ase.io.write(filename, atoms, format=out_format)

    @classmethod
    def from_file(cls, path: str, in_format: str = "auto") -> StructureData:
        """
        Read structure from file using ASE and create StructureData.

        Args:
            path: File path
            in_format: Format specification ("auto" for auto-detection)

        Returns:
            StructureData instance
        """
        fmt = None if in_format == "auto" else in_format
        atoms = ase.io.read(path, format=fmt)
        return cls(atoms=atoms)
