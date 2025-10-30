#!/usr/bin/env python
import os
import click
import spglib
from .fourthorder_sow import sow
from .fourthorder_reap import reap
from .fourthorder_with_calc import get_fc


spglib_dir = os.path.dirname(spglib.__file__)

LD_LIBRARY_PATH = os.path.join(spglib_dir, "lib64")
os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


@click.group()
def fourthorder():
    """Fourth-order force constants calculation toolkit."""
    pass


# Add subcommands
fourthorder.add_command(sow, name="sow")
fourthorder.add_command(reap, name="reap")
fourthorder.add_command(get_fc, name="get-fc")


if __name__ == "__main__":
    fourthorder()
