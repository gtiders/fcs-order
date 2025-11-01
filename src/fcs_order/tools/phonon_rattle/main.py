#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click 
import numpy as np

from ase.io import read, write
 
from .core import generate_phonon_rattled_structures
from .utils import parse_FORCE_CONSTANTS, plot_distributions

@click.command()
@click.argument('sposcar', type=click.Path(exists=True))
@click.argument('fc2', type=click.Path(exists=True))
@click.option('--number',"-n", type=int, default=100, help='Number of rattled structures to generate per temperature')
@click.option('--temperatures',"-t", type=str, default="300", help='Temperature in K,such as "300,400,500"')
@click.option('--min_distance', type=float, default=1.5, help='Minimum distance between atoms in A')
@click.option("--if_qm", type=bool, default=True, is_flag=True, help='Whether to consider quantum effects')
@click.option("--imag_freq_factor", type=float, default=1.0, help='Imaginary frequency factor')
@click.option("--output", "-o", type=str, default="structures_phonon_rattle", help='Output filename prefix')
def phononrattle(sposcar, fc2, number, temperatures, min_distance,if_qm, imag_freq_factor, output):
    """
    Generate phonon rattled structures with filtering based on displacement and distance criteria.
    For each temperature, generate structures until reaching the required number,
    filtering out structures with: 
    - any displacement > max_disp
    - any interatomic distance < min_distance
    """
    sposcar = read(sposcar)
    ref_pos = sposcar.positions.copy()
    natoms = len(sposcar)
    fc2 = parse_FORCE_CONSTANTS(fc2, natoms)
    temperatures = [float(t) for t in temperatures.split(',')]
    
    for t in temperatures:
        print(f"Processing temperature: {t} K")
        valid_structures = []
        attempts = 0
        max_attempts = number * 50  # Prevent infinite loop, set maximum attempts
        while len(valid_structures) < number and attempts < max_attempts:
            # Generate structures in batches for efficiency
            batch_size = min(number * 2, number * 10)  # Batch size
            batch_structures = generate_phonon_rattled_structures(
                sposcar, fc2, batch_size, t, QM_statistics=if_qm, imag_freq_factor=imag_freq_factor)
            
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
            print(f"  Generated {attempts} structures, found {len(valid_structures)} valid structures")
        
        # Save results
        if len(valid_structures) > 0:
            output_filename = f"{output}_T{int(t)}.xyz"
            
            # Use uniform random selection to ensure statistical distribution
            if len(valid_structures) > number:
                # Randomly select specified number from valid structures to maintain distribution
                selected_indices = np.random.choice(len(valid_structures), size=number, replace=False)
                selected_structures = [valid_structures[i] for i in selected_indices]
            else:
                selected_structures = valid_structures
            
            write(output_filename, selected_structures, format="xyz")
            plot_distributions(selected_structures, ref_pos, T=t)
            print(f"  Saved {len(selected_structures)} structures to {output_filename}")
        
        if len(valid_structures) < number:
            print(f"  Warning: Only found {len(valid_structures)} valid structures out of {number} requested")