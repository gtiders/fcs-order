#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party imports
from collections import OrderedDict

import ase.units as aunits
import matplotlib.pyplot as plt
import numpy as np
import typer
from ase import Atoms
from ase.io import write

from fcsorder.io.io_abstraction import read_atoms


def _n_BE(T, w_s):
    """
    Bose-Einstein distribution function.

    Parameters
    ---------
    T : float
        Temperature in Kelvin
    w_s: numpy.ndarray
        frequencies in eV (3*N,)

    Returns
    ------
    Bose-Einstein distribution for each energy at a given temperature
    """

    with np.errstate(divide="raise", over="raise"):
        try:
            n = 1 / (np.exp(w_s / (aunits.kB * T)) - 1)
        except Exception:
            n = np.zeros_like(w_s)
    return n


def _phonon_rattle(m_a, T, w2_s, e_sai, QM_statistics):
    """Thermal excitation of phonon modes as described by West and
    Estreicher, Physical Review Letters  **96**, 115504 (2006).

    _s is a mode index
    _i is a Carteesian index
    _a is an atom index

    Parameters
    ----------
    m_a : numpy.ndarray
        masses (N,)
    T : float
        temperature in Kelvin
    w2_s : numpy.ndarray
        the squared frequencies from the eigenvalue problem (3*N,)
    e_sai : numpy.ndarray
        polarizations (3*N, N, 3)
    QM_statistics : bool
        if the amplitude of the quantum harmonic oscillator shoule be used
        instead of the classical amplitude

    Returns
    -------
    displacements : numpy.ndarray
        shape (N, 3)
    """
    n_modes = 3 * len(m_a)

    # skip 3 translational modes
    argsort = np.argsort(np.abs(w2_s))
    e_sai = e_sai[argsort][3:]
    w2_s = w2_s[argsort][3:]

    w_s = np.sqrt(np.abs(w2_s))

    prefactor_a = np.sqrt(1 / m_a).reshape(-1, 1)
    if QM_statistics:
        hbar = aunits._hbar * aunits.J * aunits.s
        frequencyfactor_s = np.sqrt(hbar * (0.5 + _n_BE(T, hbar * w_s)) / w_s)
    else:
        frequencyfactor_s = 1 / w_s
        prefactor_a *= np.sqrt(aunits.kB * T)

    phases_s = np.random.uniform(0, 2 * np.pi, size=n_modes - 3)
    amplitudes_s = np.sqrt(-2 * np.log(1 - np.random.random(n_modes - 3)))

    u_ai = prefactor_a * np.tensordot(
        amplitudes_s * np.cos(phases_s) * frequencyfactor_s, e_sai, (0, 0)
    )
    return u_ai  # displacements


class _PhononRattler:
    """
    Class to be able to conveniently save modes and frequencies needed
    for phonon rattle.

    Parameters
    ----------
    masses : numpy.ndarray
        masses (N,)
    force_constants : numpy.ndarray
        second order force constant matrix, with shape `(3N, 3N)` or
        `(N, N, 3, 3)`. The conversion will be done internally if.
    imag_freq_factor: float
        If a squared frequency, w2, is negative then it is set to
        w2 = imag_freq_factor * np.abs(w2)
    """

    def __init__(self, masses, force_constants, imag_freq_factor=1.0):
        n_atoms = len(masses)
        if len(force_constants.shape) == 4:  # assume shape = (n_atoms, n_atoms, 3, 3)
            force_constants = force_constants.transpose(0, 2, 1, 3)
            force_constants = force_constants.reshape(3 * n_atoms, 3 * n_atoms)
            # Now the fc should have shape = (n_atoms * 3, n_atoms * 3)
        # Construct the dynamical matrix
        inv_root_masses = (1 / np.sqrt(masses)).repeat(3)
        D = np.outer(inv_root_masses, inv_root_masses)
        D *= force_constants
        # find frequnecies and eigenvectors
        w2_s, e_sai = np.linalg.eigh(D)
        # reshape to get atom index and Cartesian index separate
        e_sai = e_sai.T.reshape(-1, n_atoms, 3)

        # The three modes closest to zero are assumed to be zero, ie acoustic sum rules are assumed
        frequency_tol = 1e-6
        argsort = np.argsort(np.abs(w2_s))
        w2_gamma = w2_s[argsort][:3]
        if np.any(np.abs(w2_gamma) > frequency_tol):
            typer.echo(
                f"Acoustic sum rules not enforced, squared frequencies: {w2_gamma}"
            )

        # warning if any imaginary modes
        if np.any(w2_s < -frequency_tol):
            typer.echo("Imaginary modes present")

        # treat imaginary modes as real
        imag_mask = w2_s < -frequency_tol
        w2_s[imag_mask] = imag_freq_factor * np.abs(w2_s[imag_mask])

        self.w2_s = w2_s
        self.e_sai = e_sai
        self.masses = masses

    def __call__(self, atoms, T, QM_statistics):
        """rattle atoms by adding displacements

        Parameters
        ----------
        atoms : ase.Atoms
            Ideal structure to add displacements to.
        T : float
            temperature in Kelvin
        """
        u_ai = _phonon_rattle(self.masses, T, self.w2_s, self.e_sai, QM_statistics)
        atoms.positions += u_ai


# public core APIs


def generate_phonon_rattled_structures(
    atoms: Atoms,
    fc2: np.ndarray,
    n_structures: int,
    temperature: float,
    QM_statistics: bool = False,
    imag_freq_factor: float = 1.0,
) -> list[Atoms]:
    """
    Returns list of phonon-rattled configurations.
    """
    structures = []
    pr = _PhononRattler(atoms.get_masses(), fc2, imag_freq_factor)
    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        pr(atoms_tmp, temperature, QM_statistics)
        structures.append(atoms_tmp)
    return structures


# Distributions and plotting
bins = {
    "displacement": np.linspace(0.0, 1, 200),
    "distance": np.linspace(1.0, 4.5, 200),
}


def get_histogram_data(data, bins=100):
    counts, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = [(bins[i + 1] + bins[i]) / 2.0 for i in range(len(bins) - 1)]
    return bin_centers, counts


def get_distributions(structure_list, ref_pos):
    """Gets distributions of interatomic distances and displacements."""
    distances, displacements = [], []
    for atoms in structure_list:
        distances.extend(atoms.get_all_distances(mic=True).flatten())
        displacements.extend(np.linalg.norm(atoms.positions - ref_pos, axis=1))
    distributions = {}
    distributions["distance"] = get_histogram_data(distances, bins["distance"])
    distributions["displacement"] = get_histogram_data(
        displacements, bins["displacement"]
    )
    return distributions


def parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS", natoms=None):
    """Parse FORCE_CONSTANTS file to dense (3N,3N) matrix."""
    with open(filename) as fcfile:
        idx1 = []

        line = fcfile.readline()
        idx = [int(x) for x in line.split()]
        if len(idx) == 1:
            idx = [idx[0], idx[0]]
        force_constants = np.zeros((idx[0], idx[1], 3, 3), dtype="double")
        for i in range(idx[0]):
            for j in range(idx[1]):
                s_i = int(fcfile.readline().split()[0]) - 1
                if s_i not in idx1:
                    idx1.append(s_i)
                tensor = []
                for _ in range(3):
                    tensor.append([float(x) for x in fcfile.readline().split()])
                force_constants[i, j] = tensor

        return force_constants.transpose([0, 2, 1, 3]).reshape(natoms * 3, natoms * 3)


def plot_distributions(structure_list, ref_pos, T):
    """Plot distributions of interatomic distances and displacements."""
    fs = 14
    lw = 2.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    distributions = get_distributions(structure_list, ref_pos)

    units = OrderedDict(displacement="A", distance="A")
    for ax, key in zip([ax1, ax2], units.keys()):
        ax.plot(*distributions[key], lw=lw, label="Rattle")
        ax.set_xlabel("{} ({})".format(key.title(), units[key]), fontsize=fs)
        ax.set_xlim([np.min(bins[key]), np.max(bins[key])])
        ax.set_ylim(bottom=0.0)
        ax.tick_params(labelsize=fs)
        ax.legend(fontsize=fs)

    ax1.set_ylabel("Distribution", fontsize=fs)
    ax2.set_ylabel("Distribution", fontsize=fs)

    plt.tight_layout()
    plt.savefig("structure_generation_distributions_T{}.svg".format(T))


# ==========================
# Typer command
# ==========================


def generate_phonon_rattled_structures(
    sposcar: str = typer.Argument(..., exists=True, help="Path to SPOSCAR file"),
    fc2: str = typer.Argument(
        ..., exists=True, help="Path to second-order force constants file"
    ),
    number: int = typer.Option(
        100,
        "--number",
        "-n",
        help="Number of rattled structures to generate per temperature",
    ),
    temperatures: str = typer.Option(
        "300",
        "--temperatures",
        "-t",
        help='Temperature in K, such as "300,400,500"',
    ),
    min_distance: float = typer.Option(
        1.5,
        "--min-distance",
        help="Minimum distance between atoms in A",
    ),
    if_qm: bool = typer.Option(
        True,
        "--if-qm",
        help="Whether to consider quantum effects",
    ),
    imag_freq_factor: float = typer.Option(
        1.0,
        "--imag-freq-factor",
        help="Imaginary frequency factor",
    ),
    output: str = typer.Option(
        "structures_phonon_rattle",
        "--output",
        "-o",
        help="Output filename prefix",
    ),
):
    """
    Generate phonon rattled structures with filtering based on displacement and distance criteria.
    """
    sposcar = read_atoms(sposcar)
    ref_pos = sposcar.positions.copy()
    natoms = len(sposcar)
    fc2 = parse_FORCE_CONSTANTS(fc2, natoms)
    temperatures = [float(t) for t in temperatures.split(",")]

    for t in temperatures:
        typer.echo(f"Processing temperature: {t} K")
        valid_structures = []
        attempts = 0
        max_attempts = number * 50  # Prevent infinite loop, set maximum attempts
        while len(valid_structures) < number and attempts < max_attempts:
            # Generate structures in batches for efficiency
            batch_size = min(number * 2, number * 10)  # Batch size
            batch_structures = generate_phonon_rattled_structures(
                sposcar,
                fc2,
                batch_size,
                t,
                QM_statistics=if_qm,
                imag_freq_factor=imag_freq_factor,
            )

            for atoms in batch_structures:
                # Check distance
                distances = atoms.get_all_distances(mic=True)
                # Exclude self-distance (diagonal is 0)
                mask = ~np.eye(len(atoms), dtype=bool)
                min_interatomic_dist = np.min(distances[mask])
                if min_interatomic_dist < min_distance:
                    continue

                # Passed filtering, add to valid structures list
                valid_structures.append(atoms)

                # Exit early if required number reached
                if len(valid_structures) >= number:
                    break

            attempts += batch_size
            typer.echo(
                f"  Generated {attempts} structures, found {len(valid_structures)} valid structures"
            )

        # Save results
        if len(valid_structures) > 0:
            output_filename = f"{output}_T{int(t)}.xyz"

            # Use uniform random selection to ensure statistical distribution
            if len(valid_structures) > number:
                # Randomly select specified number from valid structures to maintain distribution
                selected_indices = np.random.choice(
                    len(valid_structures), size=number, replace=False
                )
                selected_structures = [valid_structures[i] for i in selected_indices]
            else:
                selected_structures = valid_structures

            write(output_filename, selected_structures, format="extxyz")
            plot_distributions(selected_structures, ref_pos, T=t)
            typer.echo(
                f"  Saved {len(selected_structures)} structures to {output_filename}"
            )

        if len(valid_structures) < number:
            typer.echo(
                f"  Warning: Only found {len(valid_structures)} valid structures out of {number} requested"
            )
