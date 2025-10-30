#!/usr/bin/env python
import os
import click
import spglib
from .thirdorder_sow import sow
from .thirdorder_reap import reap
from .thirdorder_with_calc import get_fc

spglib_dir = os.path.dirname(spglib.__file__)

LD_LIBRARY_PATH = os.path.join(spglib_dir, "lib64")
os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


@click.group()
def thirdorder():
    """Third-order force constants calculation toolkit."""
    pass


# Add subcommands
thirdorder.add_command(sow, name="sow")
thirdorder.add_command(reap, name="reap")
thirdorder.add_command(get_fc, name="get-fc")


if __name__ == "__main__":
    thirdorder()
