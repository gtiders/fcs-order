#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

from .core.sow import sow3, sow4
from .core.reap import reap3, reap4
from .core.mlp import mlp3, mlp4

from .phonon_rattle.cli import generate_rattled_structures


@click.group()
def cli():
    pass


cli.add_command(sow3, name="sow3")
cli.add_command(sow4, name="sow4")
cli.add_command(reap3, name="reap3")
cli.add_command(reap4, name="reap4")
cli.add_command(mlp3, name="mlp3")
cli.add_command(mlp4, name="mlp4")
cli.add_command(generate_rattled_structures, name="generate-rattled-structures")
