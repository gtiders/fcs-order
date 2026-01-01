#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup script for building Cython extensions."""

from setuptools import setup, Extension
import numpy

USE_CYTHON = True
ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "mlfcs.thirdorder.thirdorder_core",
        ["src/mlfcs/thirdorder/thirdorder_core" + ext],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "mlfcs.fourthorder.fourthorder_core",
        ["src/mlfcs/fourthorder/fourthorder_core" + ext],
        include_dirs=[numpy.get_include()],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, language_level=3)

setup(ext_modules=extensions)
