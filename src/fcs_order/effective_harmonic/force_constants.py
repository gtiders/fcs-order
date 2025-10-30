"""
Force constants calculation and extraction for effective harmonic calculations.
"""

import os
import sys
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms


def extract_force_constants(
    calculator_type: str,
    temperatures: List[int],
    cutoff_distances: List[float],
    primitive_atoms: "Atoms",
    supercell_atoms: "Atoms",
    output_directory: str,
    snapshots_directory: str = None
) -> None:
    """
    Extract force constants from molecular dynamics snapshots.

    Parameters
    ----------
    calculator_type : str
        Type of calculator used for MD simulation
    temperatures : list[int]
        List of temperatures in Kelvin
    cutoff_distances : list[float]
        List of cutoff distances in Angstrom
    primitive_atoms : ase.Atoms
        Primitive cell atoms
    supercell_atoms : ase.Atoms
        Supercell atoms
    output_directory : str
        Directory for output files
    snapshots_directory : str, optional
        Directory containing snapshots (defaults to output_directory)
    """
    if snapshots_directory is None:
        snapshots_directory = output_directory
        
    if calculator_type.lower() != "hiphive":
        print(
            f"Force constant extraction is currently only supported for hiphive calculator, "
            f"not for {calculator_type}"
        )
        sys.exit(1)

    from .utils import check_hiphive_imports
    
    # Import hiphive components
    (
        ForceConstantPotential,
        ClusterSpace,
        StructureContainer,
        ForceConstantCalculator,
        Optimizer,
        _,
    ) = check_hiphive_imports()

    # Create output directories
    fcps_dir = os.path.join(output_directory, "fcps")
    second_order_dir = os.path.join(output_directory, "2nd")
    os.makedirs(fcps_dir, exist_ok=True)
    os.makedirs(second_order_dir, exist_ok=True)

    # Process each temperature
    for temperature in temperatures:
        print(f"Processing temperature: {temperature} K")
        
        # Read snapshots
        snapshots_file = os.path.join(snapshots_directory, f"snapshots_T{temperature}.xyz")
        if not os.path.exists(snapshots_file):
            print(f"Warning: Snapshots file {snapshots_file} not found")
            continue
            
        structures = read_snapshots(snapshots_file)
        
        # Create cluster space and structure container
        cluster_space = ClusterSpace(primitive_atoms, cutoff_distances)
        structure_container = StructureContainer(cluster_space)
        
        # Add structures to container
        for structure in structures:
            structure_container.add_structure(structure)

        # Optimize and create force constant potential
        optimizer = Optimizer(structure_container.get_fit_data(), train_size=1.0)
        optimizer.train()
        print(optimizer)
        
        force_constant_potential = ForceConstantPotential(cluster_space, optimizer.parameters)
        
        # Write force constants
        force_constant_potential.get_force_constants(supercell_atoms).write_to_phonopy(
            f"{temperature}_FORCE_CONSTANT", format="text"
        )
        force_constant_potential.write(os.path.join(fcps_dir, f"T{temperature}.fcp"))


def read_snapshots(snapshots_file: str) -> List["Atoms"]:
    """
    Read snapshots from XYZ file.
    
    Parameters
    ----------
    snapshots_file : str
        Path to snapshots file
        
    Returns
    -------
    list[ase.Atoms]
        List of atoms snapshots
    """
    from ase.io import read
    return read(snapshots_file, ":")