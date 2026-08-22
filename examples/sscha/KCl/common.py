"""Local helpers shared by the KCl SSCHA example scripts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from ase import Atoms
from phonopy import load

CASE = Path(__file__).resolve().parent
INPUT = CASE / "input"
REFERENCE = CASE / "reference"
RESULTS = CASE / "results"
FIGURES = CASE / "figures"
HARMONIC_PATH = REFERENCE / "phonopy_fc222_JPCM2022.yaml.xz"
POTENTIAL_PATH = INPUT / "polymlp.yaml"
TEMPERATURE = 600.0
SNAPSHOTS = 100
ITERATIONS = 50
SEED = 42


def harmonic_phonopy():
    return load(HARMONIC_PATH)


def ase_from_phonopy(cell) -> Atoms:
    return Atoms(
        numbers=np.asarray(cell.numbers, dtype=int),
        cell=np.asarray(cell.cell, dtype=float),
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        pbc=True,
    )


def map_reference_to_phonopy(full: np.ndarray, phonon) -> np.ndarray:
    reference = ase_from_phonopy(phonon.unitcell).repeat((2, 2, 2))
    target = phonon.supercell
    if len(reference) != len(target):
        raise ValueError("MLFCS and phonopy supercells have different sizes")
    permutation = []
    for position, number in zip(target.positions, target.numbers, strict=True):
        candidates = np.flatnonzero(reference.numbers == number)
        distances = np.linalg.norm(reference.positions[candidates] - position, axis=1)
        index = int(candidates[np.argmin(distances)])
        if float(np.min(distances)) > 1e-8:
            raise ValueError("cannot map MLFCS KCl supercell onto phonopy order")
        permutation.append(index)
    permutation = np.asarray(permutation, dtype=int)
    full = np.asarray(full)
    if full.shape[0] == len(phonon.primitive):
        return full[:, permutation]
    if full.shape[0] == len(target):
        return full[np.ix_(permutation, permutation)]
    raise ValueError(f"unsupported FC2 shape {full.shape} for KCl supercell")


def paths(phonon):
    import seekpath

    primitive = phonon.primitive
    structure = (
        np.asarray(primitive.cell),
        np.asarray(primitive.scaled_positions),
        np.asarray(primitive.numbers),
    )
    path_data = seekpath.get_path(structure, recipe="hpkot")
    labels = path_data["point_coords"]
    segments = path_data["path"]
    q_paths = [np.linspace(labels[start], labels[end], 101) for start, end in segments]
    connections = [
        index + 1 < len(segments) and end == segments[index + 1][0]
        for index, (_, end) in enumerate(segments)
    ]
    return q_paths, segments, connections


def bands(phonon, force_constants):
    working = copy.deepcopy(phonon)
    working.force_constants = force_constants
    q_paths, labels, connections = paths(working)
    working.run_band_structure(q_paths, path_connections=connections)
    return working.band_structure.distances, working.band_structure.frequencies, labels


def mlfcs_working_cells():
    phonon = harmonic_phonopy()
    primitive = ase_from_phonopy(phonon.unitcell)
    reference = primitive.repeat((2, 2, 2))
    return phonon, primitive, reference


def mlfcs_calculator():
    return mlfcs_calculator_class()(pot=POTENTIAL_PATH)


def mlfcs_calculator_class():
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    return PolymlpASECalculator
