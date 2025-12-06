#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Output writers for force constants and other data."""

import io
import itertools

import numpy as np
from hiphive import ForceConstants
from ase import Atoms


def write_ifcs3(
    phi_full: np.ndarray,
    primitive_dict: dict,
    supercell_dict: dict,
    min_distances: np.ndarray,
    n_equivalent: np.ndarray,
    cell_shifts: np.ndarray,
    force_range: float,
    filename: str,
) -> None:
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.

    Args:
        phi_full: Full IFCS tensor.
        primitive_dict: Primitive cell dictionary.
        supercell_dict: Supercell dictionary.
        min_distances: Minimum distance matrix.
        n_equivalent: Number of equivalent images matrix.
        cell_shifts: Cell shifts matrix.
        force_range: Force range cutoff.
        filename: Output filename.
    """
    n_atoms = len(primitive_dict["types"])
    n_total = len(supercell_dict["types"])

    shifts_27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
    force_range_sq = force_range * force_range

    n_blocks = 0
    f = io.StringIO()
    for atom_i, atom_j in itertools.product(range(n_atoms), range(n_total)):
        if min_distances[atom_i, atom_j] >= force_range:
            continue
        j_atom_indices = atom_j % n_atoms
        shifts_ij = [
            shifts_27[i]
            for i in cell_shifts[atom_i, atom_j, : n_equivalent[atom_i, atom_j]]
        ]
        for atom_k in range(n_total):
            if min_distances[atom_i, atom_k] >= force_range:
                continue
            k_atom_indices = atom_k % n_atoms
            shifts_ik = [
                shifts_27[i]
                for i in cell_shifts[atom_i, atom_k, : n_equivalent[atom_i, atom_k]]
            ]
            d2_min = np.inf
            best_2 = None
            best_3 = None
            for shift_2 in shifts_ij:
                car_j = np.dot(
                    supercell_dict["lattvec"],
                    shift_2 + supercell_dict["positions"][:, atom_j],
                )
                for shift_3 in shifts_ik:
                    car_k = np.dot(
                        supercell_dict["lattvec"],
                        shift_3 + supercell_dict["positions"][:, atom_k],
                    )
                    d2 = ((car_j - car_k) ** 2).sum()
                    if d2 < d2_min:
                        best_2 = shift_2
                        best_3 = shift_3
                        d2_min = d2
            if d2_min >= force_range_sq:
                continue
            n_blocks += 1
            R_j = np.dot(
                supercell_dict["lattvec"],
                best_2
                + supercell_dict["positions"][:, atom_j]
                - supercell_dict["positions"][:, j_atom_indices],
            )
            R_k = np.dot(
                supercell_dict["lattvec"],
                best_3
                + supercell_dict["positions"][:, atom_k]
                - supercell_dict["positions"][:, k_atom_indices],
            )
            f.write("\n")
            f.write("{:>5}\n".format(n_blocks))
            f.write(
                "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                    list(10.0 * R_j)
                )
            )
            f.write(
                "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                    list(10.0 * R_k)
                )
            )
            f.write(
                "{:>6d} {:>6d} {:>6d}\n".format(
                    atom_i + 1, j_atom_indices + 1, k_atom_indices + 1
                )
            )
            for a, b, c in itertools.product(range(3), range(3), range(3)):
                f.write(
                    "{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(
                        a + 1, b + 1, c + 1, phi_full[a, b, c, atom_i, atom_j, atom_k]
                    )
                )

    with open(filename, "w") as ffinal:
        ffinal.write("{:>5}\n".format(n_blocks))
        ffinal.write(f.getvalue())
    f.close()


def write_ifcs4(
    phi_full: np.ndarray,
    primitive_dict: dict,
    supercell_dict: dict,
    min_distances: np.ndarray,
    n_equivalent: np.ndarray,
    cell_shifts: np.ndarray,
    force_range: float,
    filename: str,
) -> None:
    """
    Write out the full fourth-order interatomic force constant matrix,
    taking the force cutoff into account.

    Args:
        phi_full: Full IFCS tensor.
        primitive_dict: Primitive cell dictionary.
        supercell_dict: Supercell dictionary.
        min_distances: Minimum distance matrix.
        n_equivalent: Number of equivalent images matrix.
        cell_shifts: Cell shifts matrix.
        force_range: Force range cutoff.
        filename: Output filename.
    """
    n_atoms = len(primitive_dict["types"])
    n_total = len(supercell_dict["types"])

    shifts_27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
    force_range_sq = force_range * force_range

    n_blocks = 0
    f = io.StringIO()
    for atom_i, atom_j in itertools.product(range(n_atoms), range(n_total)):
        if min_distances[atom_i, atom_j] >= force_range:
            continue
        j_atom_indices = atom_j % n_atoms
        shifts_ij = [
            shifts_27[i]
            for i in cell_shifts[atom_i, atom_j, : n_equivalent[atom_i, atom_j]]
        ]
        for atom_k in range(n_total):
            if min_distances[atom_i, atom_k] >= force_range:
                continue
            k_atom_indices = atom_k % n_atoms
            shifts_ik = [
                shifts_27[i]
                for i in cell_shifts[atom_i, atom_k, : n_equivalent[atom_i, atom_k]]
            ]
            for atom_l in range(n_total):
                if min_distances[atom_i, atom_l] >= force_range:
                    continue
                l_atom_indices = atom_l % n_atoms
                shifts_il = [
                    shifts_27[i]
                    for i in cell_shifts[atom_i, atom_l, : n_equivalent[atom_i, atom_l]]
                ]

                d2_min = np.inf
                best_2 = None
                best_3 = None
                best_4 = None

                for shift_2 in shifts_ij:
                    car_j = np.dot(
                        supercell_dict["lattvec"],
                        shift_2 + supercell_dict["positions"][:, atom_j],
                    )
                    for shift_3 in shifts_ik:
                        car_k = np.dot(
                            supercell_dict["lattvec"],
                            shift_3 + supercell_dict["positions"][:, atom_k],
                        )
                        for shift_4 in shifts_il:
                            car_l = np.dot(
                                supercell_dict["lattvec"],
                                shift_4 + supercell_dict["positions"][:, atom_l],
                            )
                            d2_1 = ((car_j - car_k) ** 2).sum()
                            d2_2 = ((car_j - car_l) ** 2).sum()
                            d2_3 = ((car_k - car_l) ** 2).sum()
                            d2 = max(d2_1, d2_2, d2_3)
                            if d2 < d2_min:
                                best_2 = shift_2
                                best_3 = shift_3
                                best_4 = shift_4
                                d2_min = d2
                if d2_min >= force_range_sq:
                    continue
                n_blocks += 1
                R_j = np.dot(
                    supercell_dict["lattvec"],
                    best_2
                    + supercell_dict["positions"][:, atom_j]
                    - supercell_dict["positions"][:, j_atom_indices],
                )
                R_k = np.dot(
                    supercell_dict["lattvec"],
                    best_3
                    + supercell_dict["positions"][:, atom_k]
                    - supercell_dict["positions"][:, k_atom_indices],
                )
                R_l = np.dot(
                    supercell_dict["lattvec"],
                    best_4
                    + supercell_dict["positions"][:, atom_l]
                    - supercell_dict["positions"][:, l_atom_indices],
                )
                f.write("\n")
                f.write("{:>5}\n".format(n_blocks))
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * R_j)
                    )
                )
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * R_k)
                    )
                )
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * R_l)
                    )
                )
                f.write(
                    "{:>6d} {:>6d} {:>6d} {:>6d}\n".format(
                        atom_i + 1,
                        j_atom_indices + 1,
                        k_atom_indices + 1,
                        l_atom_indices + 1,
                    )
                )
                for m, n, o, p in itertools.product(
                    range(3), range(3), range(3), range(3)
                ):
                    f.write(
                        "{:>2d} {:>2d} {:>2d} {:>2d} {:>20.10f}\n".format(
                            m + 1,
                            n + 1,
                            o + 1,
                            p + 1,
                            phi_full[m, n, o, p, atom_i, atom_j, atom_k, atom_l],
                        )
                    )

    with open(filename, "w") as ffinal:
        ffinal.write("{:>5}\n".format(n_blocks))
        ffinal.write(f.getvalue())
    f.close()


def write_fc3_hdf5(
    primitive: Atoms, supercell: Atoms, shengbte_filename: str, hdf5_filename: str
) -> None:
    """
    Write out the full third-order force constant matrix in phono3py HDF5 format
    using hiphive for reading ShengBTE files and writing to HDF5.

    Args:
        primitive: Primitive cell structure (ASE Atoms).
        supercell: Supercell structure (ASE Atoms).
        shengbte_filename: Path to the ShengBTE format force constants file.
        hdf5_filename: Output HDF5 filename.
    """
    fcs = ForceConstants.read_shengBTE(supercell, shengbte_filename, primitive)
    fcs.write_to_phono3py(hdf5_filename)
