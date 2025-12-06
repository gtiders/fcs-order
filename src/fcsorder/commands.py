#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unified CLI for fcs-order force constants calculations."""

from __future__ import annotations

import typer

# Import sub-apps from modules
from fcsorder.sscha import scph
from fcsorder.second_order import fc2
from fcsorder.third_order import fc3
from fcsorder.fourth_order import fc4

from fcsorder.genstr.phonon_rattle import phonon_rattle_cli
from fcsorder.genstr.rattle import rattle_cli
from fcsorder.genstr.monte_rattle import monte_rattle_cli

# Create the main forcekit app
app = typer.Typer(
    name="forcekit",
    help="Force constants toolkit for phonon calculations using ML potentials.",
    no_args_is_help=True,
)

# Add sub-commands
app.command(name="fc2")(fc2)
app.command(name="fc3")(fc3)
app.command(name="fc4")(fc4)
app.command(name="scph")(scph)

app.command(name="phonon-rattle")(phonon_rattle_cli)
app.command(name="rattle")(rattle_cli)
app.command(name="monte-rattle")(monte_rattle_cli)


def main():
    """Entry point for the forcekit CLI."""
    app()
