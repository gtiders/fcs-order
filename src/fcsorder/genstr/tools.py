import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np


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
    plt.savefig("distributions_{}.png".format(T))
