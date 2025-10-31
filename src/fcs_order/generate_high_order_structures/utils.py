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


import click

from ase.io import read,write
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from trainstation import Optimizer
from hiphive.utilities import prepare_structures
from hiphive.structure_generation import generate_phonon_rattled_structures


@click.command()
@click.argument("dft_structures", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--prim",
    default="POSCAR",
    type=click.Path(exists=True),
    help="Path to primitive POSCAR.",
)
@click.option(
    "--supercell",
    default="SPOSCAR",
    type=click.Path(exists=True),
    help="Path to supercell POSCAR.",
)
@click.option(
    "--cluster-radius",
    default="7.0:5.0:4.0",
    help="Cluster radius in Angstrom.such as '7.0:5.0:4.0'",
)
@click.option(
    "--fit_method",
    default="least-squares",
    help="Method to fit force constants. possible choice are 'ardr', 'bayesian-ridge', 'elasticnet', 'lasso', 'least-squares', 'omp', 'rfe', 'ridge', 'split-bregman'",
)
@click.option(
    "--out",
    default="output.fcp",
    help="Path to output force constant potential.",
)
def train(dft_structures, prim, supercell, cluster_radius, fit_method, out):
    cluster_radius = [float(x) for x in cluster_radius.split(":")]
    prim = read(prim)
    supercell = read(supercell)
    dft_structures_ans = []
    for dft_structure in dft_structures:
        dft_structures_ans.extend(read(dft_structure, index=":"))
    dft_structures_ans=dft_structures_ans[::50]
    # initial model
    cs = ClusterSpace(supercell, cluster_radius)
    sc = StructureContainer(cs)

    structures = prepare_structures(dft_structures_ans, supercell)
    for structure in structures:
        sc.add_structure(structure)
    opt = Optimizer(sc.get_fit_data(), train_size=1.0, fit_method=fit_method)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    print(fcp)
    fc = fcp.get_force_constants(supercell)
    fc.write_to_phonopy(
        "FORCE_CONSTANTS_" + "_".join([str(x) for x in cluster_radius]), format="text"
    )
    fcp.write(out)
    return fcp

@click.command()
@click.argument("fcp_file", type=click.Path(exists=True))
@click.option(
    "--supercell",
    "-s",
    default="SPOSCAR",
    type=click.Path(exists=True),
    help="Path to supercell POSCAR.",
)
@click.option(
    "--num-structures",
    "-n",
    default=100,
    help="Number of phonon rattled structures to generate.",
)
@click.option(
    "--tempatures",
    "-t",
    default="300:600:900",
    help="Temperature in Kelvin. such as '300:600:900'",
)
def generate_phonon_rattle_structures(fcp_file, supercell, num_structures, tempatures):
    supercell = read(supercell)
    fcp = ForceConstantPotential.read(fcp_file)
    fc2 = fcp.get_force_constants(supercell).get_fc_array(order=2, format='ase')
    tempatures = [float(x) for x in tempatures.split(":")]
    for T in tempatures:
        rattled_structures = generate_phonon_rattled_structures(
            supercell, fc2, num_structures, T
        )
        write('structures_phonon_rattle_T{}.extxyz'.format(T), rattled_structures)

if __name__ == "__main__":

    generate_phonon_rattle_structures()
