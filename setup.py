#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
import os
import spglib
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Auto-detect spglib paths for pip installation
spglib_dir = os.path.dirname(spglib.__file__)
print(f"spglib_dir: {spglib_dir}")
INCLUDE_DIRS = [spglib_dir]
LIBRARY_DIRS = [os.path.join(spglib_dir, "lib64")]

# Set USE_CYTHON to True if you want include the cythonization in your build process.
USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "thirdorder.thirdorder_core",
        ["src/thirdorder/thirdorder_core" + ext],
        include_dirs=[numpy.get_include(), "src/thirdorder"] + INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=["symspg"],
    ),
    Extension(
        "fourthorder.Fourthorder_core",
        ["src/fourthorder/Fourthorder_core" + ext],
        include_dirs=[numpy.get_include(), "src/fourthorder"] + INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=["symspg"],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setup(
    name="fcs-order",
    ext_modules=extensions,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7",
)
