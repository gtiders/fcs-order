"""Explicit phonopy bridge for I/O and object conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


def _require_phonopy():
    try:
        from phonopy.interface.calculator import (  # noqa: F401
            calculator_info,
            get_calc_dataset,
            read_crystal_structure,
            write_crystal_structure,
        )
        from phonopy.structure.atoms import PhonopyAtoms  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ModuleNotFoundError(
            "phonopy is required for this operation. Install with `pip install phonopy`."
        ) from exc


@dataclass(frozen=True)
class PhonopyIOConfig:
    """Explicit interface declaration for structure/forces I/O."""

    structure_interface: str = "vasp"
    forces_interface: str | None = None
    output_interface: str | None = None

    def resolved_forces_interface(self) -> str:
        return self.forces_interface or self.structure_interface

    def resolved_output_interface(self) -> str:
        return self.output_interface or self.structure_interface


def list_supported_phonopy_interfaces() -> list[str]:
    try:
        from phonopy.interface.calculator import calculator_info
    except Exception:
        return []
    return sorted(calculator_info.keys())


def validate_phonopy_interface(interface: str) -> None:
    supported = list_supported_phonopy_interfaces()
    if supported and interface not in supported:
        hint = ", ".join(supported)
        raise ValueError(f"Unsupported interface '{interface}'. Supported: {hint}.")


def ase_to_phonopy_atoms(atoms: Atoms, include_masses: bool = False):
    """Convert ASE Atoms to PhonopyAtoms."""
    _require_phonopy()
    from phonopy.structure.atoms import PhonopyAtoms

    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )
    if include_masses:
        ph_atoms.masses = atoms.get_masses()
    return ph_atoms


def phonopy_to_ase_atoms(atoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms."""
    return Atoms(
        cell=atoms.cell,
        numbers=atoms.numbers,
        positions=atoms.positions,
        pbc=True,
    )


def read_structure_with_interface(path: str, interface: str) -> Atoms:
    """Read structure through explicit phonopy interface."""
    _require_phonopy()
    validate_phonopy_interface(interface)
    from phonopy.interface.calculator import read_crystal_structure

    cell, _ = read_crystal_structure(path, interface_mode=interface)
    if cell is None:
        raise ValueError(
            f"phonopy failed to read structure from {path} with interface='{interface}'"
        )
    return Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


def write_structure_with_interface(atoms: Atoms, path: str, interface: str) -> None:
    """Write structure through explicit phonopy interface."""
    _require_phonopy()
    validate_phonopy_interface(interface)
    from phonopy.interface.calculator import write_crystal_structure

    write_crystal_structure(path, ase_to_phonopy_atoms(atoms), interface_mode=interface)


def parse_forces_with_interface(
    files: list[str], num_atoms: int, interface: str
) -> np.ndarray:
    """Parse force files via explicit phonopy interface."""
    _require_phonopy()
    validate_phonopy_interface(interface)
    from phonopy.interface.calculator import get_calc_dataset

    dataset = get_calc_dataset(
        interface_mode=interface,
        num_atoms=num_atoms,
        force_filenames=files,
        verbose=True,
    )

    if not isinstance(dataset, dict) or "forces" not in dataset:
        raise ValueError(
            f"Phonopy failed to parse forces with interface='{interface}'. "
            "Check force file format and interface declaration."
        )

    forces = np.asarray(dataset["forces"], dtype=float)
    if forces.ndim != 3 or forces.shape[1:] != (num_atoms, 3):
        raise ValueError(
            f"Parsed force array has invalid shape {forces.shape}; "
            f"expected (nfiles, {num_atoms}, 3)."
        )
    return forces
