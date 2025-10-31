"""
Generate and plot distributions of displacements
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

bins = {
    "displacement": np.linspace(0.0, 1, 100),
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
    displacements = []
    for atoms in structure_list:
        displacements.extend(np.linalg.norm(atoms.positions - ref_pos, axis=1))
    distributions = {}
    distributions["displacement"] = get_histogram_data(
        displacements, bins["displacement"]
    )
    return distributions


def read_fc2(filename):
    """Reads FC2 from a file.

    Parameters
    ----------
    filename : str
        path to the file containing FC2

    Returns
    -------
    FC2 : numpy.ndarray
        FC2 tensor (`Nx3x3` array)
    """
    FC2 = np.loadtxt(filename)
    return FC2
