"""Parse ShengBTE FC3 into periodic-canonical structured blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_fc3(path: Path, primitive_cell: np.ndarray, repeats: int = 3):
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    blocks = int(lines[0][0])
    if len(lines) != 1 + 31 * blocks:
        raise ValueError("unexpected ShengBTE FC3 line count")
    inverse = np.linalg.inv(primitive_cell)
    translations = np.empty((blocks, 2, 3), dtype=np.int8)
    atoms = np.empty((blocks, 3), dtype=np.int8)
    values = np.empty((blocks, 3, 3, 3), dtype=float)
    for block in range(blocks):
        chunk = lines[1 + 31 * block : 1 + 31 * (block + 1)]
        if int(chunk[0][0]) != block + 1:
            raise ValueError(f"unexpected block number at index {block}")
        cartesian = np.asarray(chunk[1:3], dtype=float)
        fractional = np.rint(cartesian @ inverse).astype(np.int8)
        if not np.allclose(fractional @ primitive_cell, cartesian, atol=1e-8, rtol=0):
            raise ValueError(f"block {block + 1} translation is not a lattice vector")
        translations[block] = np.mod(fractional, repeats)
        atoms[block] = np.asarray(chunk[3], dtype=np.int8)
        for line in chunk[4:]:
            directions = tuple(int(value) - 1 for value in line[:3])
            values[(block, *directions)] = float(line[3])
    return translations, atoms, values
