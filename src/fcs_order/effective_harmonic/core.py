"""
Command-line interface commands for effective harmonic calculations.
"""

import os
from typing import List
import click
import numpy as np
from ase.io import read
from ase.build import make_supercell

from .calculator import initialize_calculator
from .md_simulation import run_molecular_dynamics
from .force_constants import extract_force_constants
from .utils import parse_temperature_string, validate_calc_potential_pair


@click.command()
@click.argument("na", type=int)
@click.argument("nb", type=int)
@click.argument("nc", type=int)
@click.option(
    "--prim",
    type=click.Path(exists=True),
    default="POSCAR",
    help="Path to primitive structure file.",
)
@click.option(
    "--sposcar",
    type=click.Path(exists=True),
    default="SPOSCAR",
    help="Path to supercell structure file.",
)
@click.option(
    "--calc",
    type=click.Choice(["nep", "dp", "hiphive", "ploymp"], case_sensitive=False),
    default=None,
    help="Calculator to use (nep or dp or hiphive or ploymp)",
)
@click.option(
    "--potential",
    type=click.Path(exists=True),
    default=None,
    help="Potential file to use (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml')",
)
@click.option(
    "--temperatures",
    type=str,
    default="300",
    help="Temperatures to use (e.g. '2000,1000,300' for 2000 K, 1000 K, and 300 K)",
)
@click.option(
    "--neq",
    type=int,
    default=3000,
    help="Number of equilibration steps to use.",
)
@click.option(
    "--nprod",
    type=int,
    default=3000,
    help="Number of production steps to use.",
)
@click.option(
    "--dt",
    type=float,
    default=1.0,
    help="Time step to use (in fs).",
)
@click.option(
    "--dump",
    type=int,
    default=100,
    help="Dump interval to use.",
)
@click.option(
    "--friction",
    type=float,
    default=0.02,
    help="Friction coefficient for Langevin thermostat (in ps^-1).",
)
@click.option(
    "--cutoff", type=float, default=6.0, help="Cutoff distance to use (in Angstrom)."
)
@click.option(
    "--outdir", type=click.Path(), default="md_runs", help="Output directory to use."
)
def extract_force_constants_from_md(
    na: int,
    nb: int,
    nc: int,
    prim: str = "POSCAR",
    sposcar: str = "SPOSCAR",
    calc: str = None,
    potential: str = None,
    temperatures: str = "300",
    neq: int = 3000,
    nprod: int = 5000,
    dt: float = 1.0,
    dump: int = 100,
    friction: float = 0.02,
    cutoff: float = 6.0,
    outdir: str = "md_runs",
) -> None:
    """
    Run effective harmonic molecular dynamics simulation.

    na, nb, nc is the supercell size for training which does not need to match the size of sposcar.
    prim is the primitive structure file.
    sposcar is the supercell structure file.
    calc is the calculator type (nep or dp or hiphive or ploymp).
    potential is the potential file path (e.g. 'nep.txt' or 'model.pb' or 'potential.fcp' or 'ploymp.yaml').
    temperatures is the temperature list (e.g. '2000,1000,300' for 2000 K, 1000 K, and 300 K).
    neq is the number of equilibration steps.
    nprod is the number of production steps.
    dt is the time step (in fs).
    dump is the dump interval.
    friction is the friction coefficient of Langevin thermostat (in ps^-1).
    cutoff is the cutoff distance (in Angstrom).
    """
    # Validate calculator and potential pair
    validate_calc_potential_pair(calc, potential)

    # Parse temperatures
    temperature_list = parse_temperature_string(temperatures)

    # Read and prepare structures
    primitive_atoms = read(prim)
    training_supercell = make_supercell(
        primitive_atoms, np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    )

    # Initialize calculator
    calculator = initialize_calculator(calc, potential, training_supercell)

    # Create output directories
    os.makedirs(outdir, exist_ok=True)

    # Run molecular dynamics simulation
    run_molecular_dynamics(
        temperatures=temperature_list,
        ideal_atoms=training_supercell,
        output_directory=outdir,
        calculator=calculator,
        equilibration_steps=neq,
        production_steps=nprod,
        time_step=dt,
        dump_interval=dump,
        friction_coefficient=friction,
    )

    # Extract force constants
    extract_force_constants(
        temperatures=temperature_list,
        cutoff_distances=[cutoff],
        primitive_atoms=primitive_atoms,
        supercell_atoms=read(sposcar),
        output_directory=outdir,
    )


@click.command()
@click.argument("dft_structures", type=click.Path(exists=True), nargs=-1)
@click.option("--prim", type=click.Path(exists=True), nargs=1)
@click.option("--sposcar", type=click.Path(exists=True), nargs=1)
@click.option("--cutoff", type=float, nargs=1)
def extract_force_constants_from_dft(
    dft_structures: List[str],
    prim: str,
    sposcar: str,
    cutoff: float,
) -> None:
    """
    Extract force constants from DFT structures.
    """
    from .utils import check_hiphive_imports

    cutoff_distances = [cutoff]
    primitive_atoms = read(prim)
    supercell_atoms = read(sposcar)

    # Read DFT structures
    dft_structures_list = []
    for dft_structure_file in dft_structures:
        dft_structures_list.extend(read(dft_structure_file, ":"))

    # Import hiphive components
    (
        ForceConstantPotential,
        ClusterSpace,
        StructureContainer,
        _,
        Optimizer,
        prepare_structures,
    ) = check_hiphive_imports()

    # Prepare and process structures
    cluster_space = ClusterSpace(primitive_atoms, cutoff_distances)
    structure_container = StructureContainer(cluster_space)

    prepared_structures = prepare_structures(dft_structures_list, supercell_atoms)
    for structure in prepared_structures:
        structure_container.add_structure(structure)

    # Optimize and create force constant potential
    optimizer = Optimizer(structure_container.get_fit_data(), train_size=1.0)
    optimizer.train()
    print(optimizer)

    force_constant_potential = ForceConstantPotential(
        cluster_space, optimizer.parameters
    )

    # Write force constants
    force_constant_potential.get_force_constants(supercell_atoms).write_to_phonopy(
        f"{cutoff}_FORCE_CONSTANT", format="text"
    )
    force_constant_potential.write(f"{cutoff}.fcp")
