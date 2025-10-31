#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

from .core import generate_phonon_rattled_structures


@click.command()
def generate_rattled_structures():
    """Generate phonon rattled structures."""
    pass
