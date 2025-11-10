#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import numpy as np

# Re-export helper to keep a single import surface for callers
try:  # pragma: no cover - passthrough import
    from ..core.secondorder_core import (
        build_supercell_from_matrix as build_supercell_from_matrix,
    )  # type: ignore
except Exception:  # Import lazily when actually used by callers
    build_supercell_from_matrix = None  # type: ignore


def parse_supercell_matrix(supercell_matrix: List[int]) -> np.ndarray:
    """
    Parse a 3- or 9-element supercell specification into a 3x3 numpy array.

    This is a pure utility with no side effects. It raises ValueError on invalid input.
    """
    if len(supercell_matrix) not in (3, 9):
        raise ValueError("Supercell matrix must have 3 (diagonal) or 9 (3x3) integers")
    if len(supercell_matrix) == 3:
        na, nb, nc = supercell_matrix
        return np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]], dtype=int)
    # len == 9
    return np.array(supercell_matrix, dtype=int).reshape(3, 3)
