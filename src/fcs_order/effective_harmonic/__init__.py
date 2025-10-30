"""
Effective harmonic force constants calculation package.

This package provides tools for calculating effective harmonic force constants
from molecular dynamics simulations using various calculators (NEP, DP, hiphive, ploymp).
"""

import click
from .core import extract_force_constants_from_md, extract_force_constants_from_dft


@click.group()
def main():
    """Effective harmonic force constants calculation toolkit."""
    pass


# Register commands
main.add_command(extract_force_constants_from_md, name="extract-fc-md")
main.add_command(extract_force_constants_from_dft, name="extract-fc-dft")


__all__ = [
    "main",
]
