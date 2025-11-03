import click

from .duck_read import read_with_calc,dft_read
from .calculators_writer.alamode import write2alm


@click.group()
def write2fcsets():
    """
    Write FCSets files in the format required by many softwares. such as alamode phonopy or noraml extxyz!
    """
    pass

@write2fcsets.command()
@click.argument("super_cell")
@click.argument("structures", nargs=-1)
@click.option("--step", default=1, help="Step to select structures")
@click.option("--calc-type", default="vasp", help="Calculator type")
@click.option("--potential-file", default=None, help="Path to potential file")
@click.option("--is-correct-with-spuer-cell", default=False, help="Whether to correct atomic forces based on supercell reference")
@click.option("--output-file", default="DFTSETS", help="Output file name")
def alamode(
    super_cell, structures, step, calc_type, potential_file, is_correct_with_spuer_cell, output_file
):
    super_cell, all_atoms = read_with_calc(
        super_cell, structures, step, calc_type, potential_file
    )
    write2alm(
        super_cell, all_atoms, is_correct_with_spuer_cell, output_file=output_file
    )

@write2fcsets.command()
@click.argument("super_cell")
@click.argument("structures", nargs=-1)
@click.option("--step", default=1, help="Step to select structures")
@click.option("--output-file", default="DFTSETS.extxyz", help="Output file name")
def dft_extxyz(
    super_cell, structures, step, output_file
):
    super_cell, all_atoms = dft_read(
        super_cell, structures, step
    )
    for atom in all_atoms:
        atom.get_forces()
        atom.get_stress()
        atom.get_energy()
    from ase.io import write
    write(output_file, all_atoms, format="extxyz")


@write2fcsets.command()
@click.argument("super_cell")
@click.argument("structures", nargs=-1)
@click.option("--step", default=1, help="Step to select structures")
@click.option("--calc-type", default="mlp", help="Calculator type")
@click.option("--potential-file", default=None, help="Path to potential file")
@click.option("--output-file", default="MLPSETS.extxyz", help="Output file name")
def mlp_extxyz(
    super_cell, structures, step, calc_type, potential_file, output_file
):
    super_cell, all_atoms = read_with_calc(
        super_cell, structures, step, calc_type, potential_file
    )
    for atom in all_atoms:
        atom.get_forces()
        atom.get_stress()
        atom.get_energy()
    from ase.io import write
    write(output_file, all_atoms, format="extxyz")