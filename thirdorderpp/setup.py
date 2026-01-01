#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
from distutils.core import setup
from distutils.extension import Extension

# Add the location of the "spglib/spglib.h" to this list if necessary.
# Example: INCLUDE_DIRS=["/home/user/local/include"]
# NO LONGER NEEDED - using Python spglib package
INCLUDE_DIRS = []
# Add the location of the spglib shared library to this list if necessary.
# Example: LIBRARY_DIRS=["/home/user/local/lib"]
# NO LONGER NEEDED - using Python spglib package
LIBRARY_DIRS = []

# Set USE_CYTHON to True if you want include the cythonization in your build
# process.
USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "thirdorder_core", ["thirdorder_core" + ext],
        include_dirs=[numpy.get_include()] + INCLUDE_DIRS
        # NO LONGER NEEDED: library_dirs, runtime_library_dirs, libraries
        # We now use Python spglib package instead of C library
    )
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(name="thirdorder", ext_modules=extensions)
