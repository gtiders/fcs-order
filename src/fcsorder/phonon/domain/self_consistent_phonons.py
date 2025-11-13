import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator


def parse_parameters_form_fcs2(fcs_2nd: str, supercell: Atoms, cs: Any):
    """Parse the parameters from the FCS2 file.

    Args:
        fcs_2nd (str): The path to the FCS2 file.
        supercell (Atoms): The supercell structure.
        cs (ClusterSpace): The cluster space.

    Returns:
        dict: The parsed parameters.
    """
    try:
        from hiphive import ForceConstants
        from hiphive.utilities import extract_parameters
    except ImportError as e:
        raise ImportError(
            "Failed to import hiphive utilities. Install hiphive: pip install hiphive"
        ) from e

    fcs = ForceConstants.read_phonopy(supercell, fcs_2nd)
    parameters = extract_parameters(fcs, cs)
    return parameters


def run_scph(
    primcell: Atoms,
    calc: Calculator,
    supercell: Atoms,
    temperatures: list[float],
    cutoff: float,
    alpha: float = 0.2,
    n_iterations: int = 100,
    n_structures: int = 50,
    fcs_2nd=None,
    is_qm: bool = True,
    imag_freq_factor: float = 1.0,
):
    """Run the self-consistent phonon calculation.

    Args:
        primcell (Atoms): The primitive cell structure.
        calc (Calculator): The calculator for computing forces.
        supercell (Atoms): The supercell structure.
        temperatures (list[float]): List of temperatures for the calculation.
        cutoff (float): Cutoff radius for the cluster space.
        alpha (float, optional): The mixing parameter for SCPH iterations. Defaults to 0.2.
        n_iterations (int, optional): The number of iterations for SCPH. Defaults to 100.
        n_structures (int, optional): The number of structures to generate. Defaults to 50.
        fcs_2nd (str, optional): The path to the FCS2 file for initial parameters.
            Only supports phonopy's hdf5 format or FORCE_CONSTANTS format. Defaults to None.
        is_qm (bool, optional): Whether to use quantum statistics. Defaults to True.
        imag_freq_factor (float, optional): Factor for handling imaginary frequencies. Defaults to 1.0.

    Returns:
        None: Results are saved to files in the 'fcps/' and 'scph_trajs/' directories.
    """
    # Lazy import hiphive here to avoid hard dependency at import-time
    try:
        from hiphive import ClusterSpace, ForceConstantPotential
        from hiphive.self_consistent_phonons import self_consistent_harmonic_model
    except ImportError as e:
        raise ImportError(
            "Failed to import hiphive. Please install hiphive package first. "
            "You can install it using: pip install hiphive"
        ) from e

    ## parameters
    cutoffs = [cutoff]
    # setup scph
    cs = ClusterSpace(primcell, cutoffs)
    if fcs_2nd is not None:
        typer.echo("using parameters from user!")
        parameters_start = parse_parameters_form_fcs2(fcs_2nd, supercell, cs)
    else:
        parameters_start = None

    # run scph
    os.makedirs("scph_trajs/", exist_ok=True)
    os.makedirs("fcps/", exist_ok=True)
    for T in temperatures:
        typer.echo(f"runing at {T}K")
        parameters_traj = self_consistent_harmonic_model(
            atoms_ideal=supercell,
            calc=calc,
            cs=cs,
            T=T,
            alpha=alpha,
            n_iterations=n_iterations,
            n_structures=n_structures,
            QM_statistics=is_qm,
            parameters_start=parameters_start,
            imag_freq_factor=imag_freq_factor,
        )
        fcp_scph = ForceConstantPotential(cs, parameters_traj[-1])
        fcp_scph.get_force_constants(supercell).write_to_phonopy(
            f"fcps/{T}_FORCE_CONSTANTS", format="text"
        )

        fcp_scph.write(f"fcps/scph_T{T}.fcp")
        np.savetxt(f"scph_trajs/scph_parameters_T{T}", np.array(parameters_traj))
        analyze_scph_convergence(T)


def analyze_scph_convergence(T: float):
    """Analyze the convergence of SCPH parameters.
    Returns:
        None: Plots are saved to 'scph_trajs/scph_parameter_T{T}.png'.
    """

    # read parameter trajs
    parameter_trajs = np.loadtxt(f"scph_trajs/scph_parameters_T{T}")

    # calculate parameter differences between iterations
    delta_parameters = [
        np.linalg.norm(p - p2) for p, p2 in zip(parameter_trajs, parameter_trajs[1:])
    ]

    # setup plot
    fig = plt.figure(figsize=(8, 3.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # plot parameters
    ax1.plot(parameter_trajs)

    # plot parameter differences
    ax2.plot(delta_parameters)

    # set labels and limits for parameter plot
    ax1.set_xlabel(f"SCPH iteration {T} K")
    ax1.set_ylabel("Parameters")
    ax1.set_xlim([0, len(parameter_trajs)])

    # set labels and limits for parameter difference plot
    ax2.set_xlabel(f"SCPH iteration {T} K")
    ax2.set_ylabel("$\\Delta$ Parameters")
    ax2.set_xlim([0, len(delta_parameters)])
    ax2.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(f"scph_trajs/scph_parameter_T{T}.png")
    typer.echo(
        f"SCPH parameter convergence plot for T={T} K saved to 'scph_trajs/scph_parameter_T{T}.png'"
    )
