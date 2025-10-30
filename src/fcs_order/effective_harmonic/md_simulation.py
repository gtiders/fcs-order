"""
Molecular dynamics simulation utilities for effective harmonic calculations.
"""

import numpy as np
from typing import List, Union
from ase import units
from ase.io import write, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.io.trajectory import Trajectory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def run_molecular_dynamics(
    temperatures: List[int],
    ideal_atoms: "Atoms", 
    output_directory: str,
    calculator: "Calculator",
    equilibration_steps: int = 3000,
    production_steps: int = 3000,
    time_step: float = 1.0,
    dump_interval: int = 100,
    friction_coefficient: float = 0.02
) -> None:
    """
    Run molecular dynamics simulation at given temperatures.

    Parameters
    ----------
    temperatures : list[int]
        List of temperatures in Kelvin
    ideal_atoms : ase.Atoms
        Ideal atoms configuration
    output_directory : str
        Directory for output files
    calculator : ase.calculators.calculator.Calculator
        ASE calculator instance
    equilibration_steps : int, default=3000
        Number of equilibration steps
    production_steps : int, default=3000
        Number of production steps
    time_step : float, default=1.0
        Time step in femtoseconds
    dump_interval : int, default=100
        Interval for dumping trajectory
    friction_coefficient : float, default=0.02
        Friction coefficient for Langevin thermostat in ps^-1
    """
    for temperature in temperatures:
        print(f"Temperature: {temperature} K")

        # Set up molecular dynamics simulation
        atoms = ideal_atoms.copy()
        atoms.calc = calculator
        
        # Initialize Langevin dynamics
        dynamics = Langevin(
            atoms, 
            time_step * units.fs, 
            temperature_K=temperature, 
            friction=friction_coefficient
        )

        # Equilibration run
        random_state = np.random.RandomState(2020)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=random_state)
        dynamics.run(equilibration_steps)

        # Production run
        log_file = f"{output_directory}/T{temperature}.log"
        traj_file = f"{output_directory}/T{temperature}.traj"
        
        logger = MDLogger(
            dynamics, atoms, log_file, header=True, stress=False, peratom=True, mode="w"
        )
        traj_writer = Trajectory(traj_file, "w", atoms)
        
        dynamics.attach(logger, interval=dump_interval)
        dynamics.attach(traj_writer.write, interval=dump_interval)
        dynamics.run(production_steps)

        # Prepare snapshots for later use
        frames = _prepare_snapshots(traj_file, ideal_atoms)
        print(f"Number of snapshots: {len(frames)}")
        write(f"{output_directory}/snapshots_T{temperature}.xyz", frames, format="extxyz")


def _prepare_snapshots(trajectory_file: str, ideal_atoms: "Atoms") -> List["Atoms"]:
    """
    Prepare snapshots from trajectory file.
    
    Parameters
    ----------
    trajectory_file : str
        Path to trajectory file
    ideal_atoms : ase.Atoms
        Ideal atoms configuration for reference
        
    Returns
    -------
    list[ase.Atoms]
        List of processed snapshots
    """
    frames = []
    ideal_positions = ideal_atoms.get_positions()
    
    for atoms in read(trajectory_file, ":"):
        forces = atoms.get_forces()
        displacements = atoms.positions - ideal_positions
        
        # Reset positions to ideal and add arrays
        atoms.positions = ideal_positions.copy()
        atoms.new_array("displacements", displacements)
        atoms.new_array("forces", forces)
        frames.append(atoms.copy())
        
    return frames