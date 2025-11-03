import click

from .duck_read import read_with_calc
from .calculators_writer.alamode import write2alm


@click.group()
def write2fcsets():
    """
    Write FCSets files in the format required by many softwares. such as alamode phonopy gpumd!
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
