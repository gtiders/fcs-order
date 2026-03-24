"""I/O helpers for structure reading and displacement writing."""

from __future__ import annotations

import glob
import os
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import write

from mlfcs.ase2dict import Structure


def _structure_from_ase(path: str, in_format: str) -> Structure:
    return Structure.from_file(path, in_format=in_format)


def list_supported_interfaces() -> list[str]:
    try:
        from phonopy.interface.calculator import calculator_info
    except Exception:
        return []
    return sorted(calculator_info.keys())


def _structure_from_phonopy(path: str, interface_mode: str) -> Structure:
    from phonopy.interface.calculator import read_crystal_structure

    cell, _ = read_crystal_structure(path, interface_mode=interface_mode)
    if cell is None:
        raise ValueError(
            f"phonopy failed to read structure from {path} with interface='{interface_mode}'"
        )

    atoms = Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )
    return Structure.from_atoms(atoms)


def read_structure(path: str, interface: str = "vasp") -> Structure:
    """
    Read structure with explicit interface mode.

    No auto mode is used here. If parsing fails, we emit a guided error that
    includes supported phonopy interfaces.
    """
    supported = list_supported_interfaces()
    if supported and interface not in supported:
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

    try:
        from phonopy.interface.calculator import write_crystal_structure
        from phonopy.structure.atoms import PhonopyAtoms
    except Exception as e:
        raise ValueError(
            f"Cannot write out_format='{out_format}' because phonopy writer is unavailable: {e}"
        ) from e

    atoms = structure.to_atoms()
    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )
    try:
        write_crystal_structure(filename, ph_atoms, interface_mode=out_format)
    except Exception as e:
        raise ValueError(
            f"Failed to write structure with interface '{out_format}': {e}. "
            "Try '--format vasp' (or '--format xyz') if this interface needs "
            "extra template information."
        ) from e


def write_xyz_trajectory(
    structures_with_id: Iterable[tuple[int, Structure]],
    outfile: str,
) -> int:
    """Write displacement structures to one extxyz file with config_id."""
    atoms_list = []
    for config_id, structure in structures_with_id:
        atoms = structure.to_atoms()
        atoms.info["config_id"] = config_id
        atoms_list.append(atoms)

    if atoms_list:
        write(outfile, atoms_list, format="extxyz")
    return len(atoms_list)


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
                "CLI reap no longer supports xyz/extxyz. "
                "Use phonopy-supported calculator outputs with --forces-interface, "
                "or use Python library workflow for xyz force inputs."
            )
        if not os.path.isfile(fname):
            raise ValueError(f"{fname} is not a file")

    from phonopy.interface.calculator import get_calc_dataset

    dataset = get_calc_dataset(
        interface_mode=interface,
        num_atoms=num_atoms,
        force_filenames=all_files,
        verbose=True,
    )

    if not isinstance(dataset, dict) or "forces" not in dataset:
        raise ValueError(
            f"Phonopy failed to parse forces with interface='{interface}'. "
            "Check --forces-interface and force file format."
        )

    forces = np.asarray(dataset["forces"], dtype=float)
    if forces.ndim != 3 or forces.shape[1] != num_atoms or forces.shape[2] != 3:
        raise ValueError(
            "Parsed force array has invalid shape. "
            f"Got {forces.shape}, expected (nfiles, {num_atoms}, 3)."
        )

    force_map = {i + 1: forces[i] for i in range(forces.shape[0])}
    return all_files, force_map
