from ase.io import read
from fcs_order.utils.calculator import initialize_calculator


def dft_read(super_cell, structures, step):
    all_atoms = []
    super_cell = read(super_cell)
    for p in structures:
        atom = read(p, index=":")
        all_atoms.extend(atom)
    return super_cell, all_atoms[::step]


def read_with_calc(super_cell, structures, step, calc_type, potential_file):
    super_cell, all_atoms = dft_read(super_cell, structures, step)
    calc = initialize_calculator(calc_type, potential_file, super_cell)
    for atom in all_atoms:
        atom.calc = calc
    super_cell.calc = calc
    return super_cell, all_atoms
