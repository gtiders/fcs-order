#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
import os
import spglib
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# Auto-detect spglib paths for pip installation
spglib_dir = os.path.dirname(spglib.__file__)
print(f"spglib_dir: {spglib_dir}")
INCLUDE_DIRS = [spglib_dir]
LIBRARY_DIRS = [os.path.join(spglib_dir, "lib64")]

extensions = [
    Extension(
        "fcs_order.thirdorder.thirdorder_core",
        ["src/fcs_order/thirdorder/thirdorder_core" + "pyx"],
        include_dirs=[numpy.get_include(), "src/fcs_order/thirdorder"] + INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=["symspg"],
    ),
    Extension(
        "fcs_order.fourthorder.fourthorder_core",
        ["src/fcs_order/fourthorder/fourthorder_core" + "pyx"],
        include_dirs=[numpy.get_include(), "src/fcs_order/fourthorder"] + INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=["symspg"],
    ),
]

extensions = cythonize(extensions)

setup(
    name="fcs-order",
    ext_modules=extensions,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
)
