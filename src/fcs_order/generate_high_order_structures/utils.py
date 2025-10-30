"""
Generate and plot distributions of displacements, distances, and forces.
Forces are calculated using the EMT calculator.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from ase.io import read
from ase.calculators.emt import EMT

bins = {
    "displacement": np.linspace(0.0, 0.7, 80),
    "distance": np.linspace(1.0, 4.5, 150),
    "force": np.linspace(0.0, 8.0, 50),
}

"""
Generate and plot distributions of displacements, distances, and forces.
Forces are calculated using the EMT calculator.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from ase.io import read
from ase.calculators.emt import EMT

bins = {
    "displacement": np.linspace(0.0, 0.7, 80),
    "distance": np.linspace(1.0, 4.5, 150),
    "force": np.linspace(0.0, 8.0, 50),
}


def get_histogram_data(data, bins=100):
    counts, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = [(bins[i + 1] + bins[i]) / 2.0 for i in range(len(bins) - 1)]
    return bin_centers, counts


def get_distributions(structure_list, ref_pos, calc):
    """Gets distributions of interatomic distances and displacements.

    Parameters
    ----------
    structure_list : list(ase.Atoms)
        list of structures used for computing distributions
    ref_pos : numpy.ndarray
        reference positions used for computing the displacements (`Nx3` array)
    calc : ASE calculator object
        `calculator
        <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
        used for computing forces
    """
    distances, displacements, forces = [], [], []
    for atoms in structure_list:
        distances.extend(atoms.get_all_distances(mic=True).flatten())
        displacements.extend(np.linalg.norm(atoms.positions - ref_pos, axis=1))
        atoms.calc = calc
        forces.extend(np.linalg.norm(atoms.get_forces(), axis=1))
    distributions = {}
    distributions["distance"] = get_histogram_data(distances, bins["distance"])
    distributions["displacement"] = get_histogram_data(
        displacements, bins["displacement"]
    )
    distributions["force"] = get_histogram_data(forces, bins["force"])
    return distributions


# read atoms
T = 800
reference_structure = read("reference_structure.xyz")
ref_pos = reference_structure.get_positions()

structures_rattle = read("structures_rattle.extxyz@:")
structures_mc = read("structures_mc_rattle.extxyz@:")
structures_phonon = read("structures_phonon_rattle_T{}.extxyz@:".format(T))
structures_md = read("md_trajectory_T{}.traj@:".format(T))

calc = EMT()

# generate distributions
distributions_rattle = get_distributions(structures_rattle, ref_pos, calc)
distributions_mc = get_distributions(structures_mc, ref_pos, calc)
distributions_phonon = get_distributions(structures_phonon, ref_pos, calc)
distributions_md = get_distributions(structures_md, ref_pos, calc)

# plot
fs = 14
lw = 2.0
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

units = OrderedDict(displacement="A", distance="A", force="eV/A")
for ax, key in zip([ax1, ax2, ax3], units.keys()):
    ax.plot(*distributions_rattle[key], lw=lw, label="Rattle")
    ax.plot(*distributions_mc[key], lw=lw, label="Monte Carlo rattle")
    ax.plot(*distributions_phonon[key], lw=lw, label="Phonon rattle {}K".format(T))
    ax.plot(*distributions_md[key], lw=lw, label="MD {}K".format(T))

    ax.set_xlabel("{} ({})".format(key.title(), units[key]), fontsize=fs)
    ax.set_xlim([np.min(bins[key]), np.max(bins[key])])
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs)

ax1.set_ylabel("Distribution", fontsize=fs)
plt.tight_layout()
plt.savefig("structure_generation_distributions.svg")


def get_distributions(structure_list, ref_pos, calc):
    """Gets distributions of interatomic distances and displacements.

    Parameters
    ----------
    structure_list : list(ase.Atoms)
        list of structures used for computing distributions
    ref_pos : numpy.ndarray
        reference positions used for computing the displacements (`Nx3` array)
    calc : ASE calculator object
        `calculator
        <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
        used for computing forces
    """
    distances, displacements, forces = [], [], []
    for atoms in structure_list:
        distances.extend(atoms.get_all_distances(mic=True).flatten())
        displacements.extend(np.linalg.norm(atoms.positions - ref_pos, axis=1))
        atoms.calc = calc
        forces.extend(np.linalg.norm(atoms.get_forces(), axis=1))
    distributions = {}
    distributions["distance"] = get_histogram_data(distances, bins["distance"])
    distributions["displacement"] = get_histogram_data(
        displacements, bins["displacement"]
    )
    distributions["force"] = get_histogram_data(forces, bins["force"])
    return distributions


# read atoms
T = 800
reference_structure = read("reference_structure.xyz")
ref_pos = reference_structure.get_positions()

structures_rattle = read("structures_rattle.extxyz@:")
structures_mc = read("structures_mc_rattle.extxyz@:")
structures_phonon = read("structures_phonon_rattle_T{}.extxyz@:".format(T))
structures_md = read("md_trajectory_T{}.traj@:".format(T))

calc = EMT()

# generate distributions
distributions_rattle = get_distributions(structures_rattle, ref_pos, calc)
distributions_mc = get_distributions(structures_mc, ref_pos, calc)
distributions_phonon = get_distributions(structures_phonon, ref_pos, calc)
distributions_md = get_distributions(structures_md, ref_pos, calc)

# plot
fs = 14
lw = 2.0
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

units = OrderedDict(displacement="A", distance="A", force="eV/A")
for ax, key in zip([ax1, ax2, ax3], units.keys()):
    ax.plot(*distributions_rattle[key], lw=lw, label="Rattle")
    ax.plot(*distributions_mc[key], lw=lw, label="Monte Carlo rattle")
    ax.plot(*distributions_phonon[key], lw=lw, label="Phonon rattle {}K".format(T))
    ax.plot(*distributions_md[key], lw=lw, label="MD {}K".format(T))

    ax.set_xlabel("{} ({})".format(key.title(), units[key]), fontsize=fs)
    ax.set_xlim([np.min(bins[key]), np.max(bins[key])])
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs)

ax1.set_ylabel("Distribution", fontsize=fs)
plt.tight_layout()
plt.savefig("structure_generation_distributions.svg")
