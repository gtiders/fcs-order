"""
Generate and plot distributions of displacements
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

bins = {
    "displacement": np.linspace(0.0, 1, 200),
    "distance": np.linspace(1.0, 4.5, 200),
}


def get_histogram_data(data, bins=100):
    counts, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = [(bins[i + 1] + bins[i]) / 2.0 for i in range(len(bins) - 1)]
    return bin_centers, counts


def get_distributions(structure_list, ref_pos):
    """Gets distributions of interatomic distances and displacements.

    Parameters
    ----------
    structure_list : list(ase.Atoms)
        list of structures used for computing distributions
    ref_pos : numpy.ndarray
        reference positions used for computing the displacements (`Nx3` array)
    """
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
    """Parse FORCE_CONSTANTS.

    Parameters
    ----------
    filename : str, optional
        Filename.
    """
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
    """Plot distributions of interatomic distances and displacements.

    Parameters
    ----------
    structure_list : list(ase.Atoms)
        list of structures used for computing distributions
    ref_pos : numpy.ndarray
        reference positions used for computing the displacements (`Nx3` array)
    T : float
        temperature used for computing the phonon rattle distributions
    """
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
