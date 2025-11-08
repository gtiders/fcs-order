#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy

from setuptools import setup, find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Compiler import Options

# Increase Cython dimension limit from 7 to 100
Options.buffer_max_dims = 10

# Auto-detect spglib paths for pip installation
extensions = [
    Extension(
        "fcsorder.core.thirdorder_core",
        ["src/fcsorder/core/thirdorder_core" + ".pyx"],
        include_dirs=[numpy.get_include(), "src/fcsorder/core/bin"],
    ),
    Extension(
        "fcsorder.core.fourthorder_core",
        ["src/fcsorder/core/fourthorder_core" + ".pyx"],
        include_dirs=[numpy.get_include(), "src/fcsorder/core/bin"],
    ),
]

extensions = cythonize(extensions)

setup(
    name="fcsorder",
    ext_modules=extensions,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10,<3.14",
)
