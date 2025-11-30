from typing import Optional,Dict

from ase.atoms import Atoms
from ase.io import read

import copy
import itertools

import numpy as np


class IO:
    def __init__(self):
        self.poscar_atoms : Optional[Atoms] = None
        self.sposcar_atoms : Optional[Atoms] = None
        self.poscar_dict : Optional[Dict] = None
        self.sposcar_dict : Optional[Dict] = None

    def read_poscar_atoms(self, filename):
        self.poscar_atoms = read(filename)

    def convert_atoms_to_dict(self):
        """
        Convert the loaded ASE Atoms object (self.poscar_atoms) to the internal 
        dictionary structure (self.poscar_dict).
        """
        if self.poscar_atoms is None:
            return

        atoms = self.poscar_atoms
        structure_dict = {}
        
        # Convert lattice vectors from Angstrom (ASE default) to nm (internal unit)
        # internal format uses column vectors (3x3), so we transpose
        structure_dict["lattvec"] = 0.1 * atoms.get_cell().T

        chemical_symbols = atoms.get_chemical_symbols()
        
        element_list = []
        atom_counts = []
        
        if chemical_symbols:
            last_symbol = chemical_symbols[0]
            current_count = 1
            
            for symbol in chemical_symbols[1:]:
                if symbol == last_symbol:
                    current_count += 1
                else:
                    element_list.append(last_symbol)
                    atom_counts.append(current_count)
                    last_symbol = symbol
                    current_count = 1
            element_list.append(last_symbol)
            atom_counts.append(current_count)

        structure_dict["elements"] = element_list
        structure_dict["numbers"] = np.array(atom_counts, dtype=np.intc)

        # Positions: 3xN array of fractional coordinates
        structure_dict["positions"] = np.asarray(atoms.get_scaled_positions()).T

        # Types: list of integers mapping each atom to its species index
        structure_dict["types"] = np.repeat(
            range(len(structure_dict["numbers"])), 
            structure_dict["numbers"]
        ).tolist()

        self.poscar_dict = structure_dict

    @staticmethod
    def convert_dict_to_atoms(structure_dict: dict) -> Atoms:
        """
        Convert internal structure dictionary to ASE Atoms object.
        """
        symbols = np.repeat(
            structure_dict["elements"], structure_dict["numbers"]
        ).tolist()

        atoms = Atoms(
            symbols=symbols,
            scaled_positions=structure_dict["positions"].T,
            cell=structure_dict["lattvec"].T * 10.0,  # nm -> Angstrom
            pbc=True,
        )
        return atoms

    @staticmethod
    def generate_supercell_structure(structure_dict, na, nb, nc):
        """
        Create a dictionary describing a supercell.
        The atom order is determined by iterating over supercell images (k, j, i) 
        for each primitive cell atom, preserving the natural expansion order 
        (not grouped by species).
        """
        supercell_structure = dict()
        supercell_structure["na"] = na
        supercell_structure["nb"] = nb
        supercell_structure["nc"] = nc
        supercell_structure["lattvec"] = np.array(structure_dict["lattvec"])
        supercell_structure["lattvec"][:, 0] *= na
        supercell_structure["lattvec"][:, 1] *= nb
        supercell_structure["lattvec"][:, 2] *= nc
        supercell_structure["elements"] = copy.copy(structure_dict["elements"])
        supercell_structure["numbers"] = na * nb * nc * structure_dict["numbers"]
        
        num_atoms_primitive = structure_dict["positions"].shape[1]
        num_atoms_supercell = num_atoms_primitive * na * nb * nc
        supercell_structure["positions"] = np.empty((3, num_atoms_supercell))
        
        # Generate supercell positions in natural expansion order
        # Iterate over supercell images (k, j, i) and primitive atoms (atom_idx)
        # Note: This matches the original gen_SPOSCAR logic
        for pos_idx, (k, j, i, atom_idx) in enumerate(
            itertools.product(
                range(nc), range(nb), range(na), range(num_atoms_primitive)
            )
        ):
            supercell_structure["positions"][:, pos_idx] = (
                structure_dict["positions"][:, atom_idx] + [i, j, k]
            ) / [na, nb, nc]
            
        supercell_structure["types"] = []
        for _ in range(na * nb * nc):
            supercell_structure["types"].extend(structure_dict["types"])
            
        return supercell_structure



