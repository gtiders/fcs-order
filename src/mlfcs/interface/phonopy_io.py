"""I/O helpers for structure reading and displacement writing."""

from __future__ import annotations

import glob
import os

import numpy as np
from ase import Atoms

from mlfcs.structure import Structure
from mlfcs.interface import (
    list_supported_phonopy_interfaces,
    parse_forces_with_interface,
    read_structure_with_interface,
    validate_phonopy_interface,
    write_structure_with_interface,
)


def _structure_from_ase(path: str, in_format: str) -> Structure:
    return Structure.from_file(path, in_format=in_format)


def list_supported_interfaces() -> list[str]:
    return list_supported_phonopy_interfaces()


def _structure_from_phonopy(path: str, interface_mode: str) -> Structure:
    atoms = read_structure_with_interface(path, interface=interface_mode)
    return Structure.from_atoms(atoms)


def read_structure(path: str, interface: str = "vasp") -> Structure:
    """
    Read structure with explicit interface mode.

    No auto mode is used here. If parsing fails, we emit a guided error that
    includes supported phonopy interfaces.
    """
    supported = list_supported_interfaces()
    try:
        validate_phonopy_interface(interface)
    except ValueError:
        hint = ", ".join(supported)
        raise ValueError(
            f"Unsupported interface '{interface}'. Supported interfaces: {hint}. "
            "Examples: '--interface vasp -i POSCAR', '--interface abacus -i STRU'."
        )

    try:
        return _structure_from_phonopy(path, interface_mode=interface)
    except Exception as ph_err:
        # Keep a strict explicit-interface policy: do not silently switch modes.
        # For vasp only, allow ASE as a compatibility fallback.
        if interface == "vasp":
            try:
                return _structure_from_ase(path, in_format="vasp")
            except Exception as ase_err:
                raise ValueError(
                    f"Failed to read structure '{path}' with interface='vasp'. "
                    f"Phonopy error: {ph_err}. ASE fallback error: {ase_err}. "
                    "Try setting '--interface' explicitly to match your input format."
                ) from ase_err

        hint = ", ".join(supported) if supported else "phonopy unavailable"
        raise ValueError(
            f"Failed to read structure '{path}' with interface='{interface}'. "
            f"Supported interfaces: {hint}. "
            "Example: for ABACUS use '--interface abacus -i STRU'."
        ) from ph_err


def write_structure(structure: Structure, filename: str, out_format: str = "vasp") -> None:
    """Write one structure using VASP writer or phonopy interface writers."""
    if out_format == "vasp":
        structure.to_file(filename, out_format="vasp")
        return

    atoms = structure.to_atoms()
    try:
        write_structure_with_interface(atoms, filename, interface=out_format)
    except Exception as e:
        raise ValueError(
            f"Failed to write structure with interface '{out_format}': {e}. "
            "Try '--format vasp' if this interface needs "
            "extra template information."
        ) from e


def _expand_file_patterns(files_or_patterns: list[str]) -> list[str]:
    all_files: list[str] = []
    for item in files_or_patterns:
        all_files.extend(glob.glob(item))
    all_files.sort()
    return all_files


def load_forces_with_phonopy(
    files_or_patterns: list[str] | str,
    num_atoms: int,
    interface: str,
) -> tuple[list[str], dict[int, np.ndarray]]:
    """
    Load forces from calculator outputs via phonopy interface parser.

    Returns (expanded_files, force_map), where force_map key is 1-based config_id
    in file order.
    """
    if isinstance(files_or_patterns, str):
        files_or_patterns = [files_or_patterns]

    all_files = _expand_file_patterns(files_or_patterns)
    if not all_files:
        raise ValueError("No files matched input force patterns.")

    for fname in all_files:
        lower = fname.lower()
        if lower.endswith(".xyz") or lower.endswith(".extxyz"):
            raise ValueError(
                "CLI reap does not support xyz/extxyz force trajectories. "
                "Use phonopy-supported calculator outputs with --forces-interface."
            )
        if not os.path.isfile(fname):
            raise ValueError(f"{fname} is not a file")

    forces = parse_forces_with_interface(
        all_files, num_atoms=num_atoms, interface=interface
    )

    force_map = {i + 1: forces[i] for i in range(forces.shape[0])}
    return all_files, force_map
