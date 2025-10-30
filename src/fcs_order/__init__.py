import click
from .effective_harmonic import main as effective_harmonic_main
from .generate2alm.dftset import gen_dftset
from .thirdorder.thirdorder_wrapper import thirdorder
from .fourthorder.fourthorder_wrapper import fourthorder


@click.group()
def cli():
    pass


cli.add_command(effective_harmonic_main, name="effective-harmonic")
cli.add_command(gen_dftset, name="generate2alm")
cli.add_command(thirdorder, name="thirdorder")
cli.add_command(fourthorder, name="fourthorder")
