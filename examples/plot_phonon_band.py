#!/usr/bin/env python3
"""Plot a publication-ready phonon band structure from a text FC2 file.

The input structure is the *reference supercell* used to write the force
constants. Phonopy finds its primitive cell (``primitive_matrix='auto'``),
while SeeK-path supplies the standardized high-symmetry path.

Example::

    uv run python examples/plot_phonon_band.py \
        --supercell POSCAR --force-constants FORCE_CONSTANTS \
        --output phonon-band.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supercell", required=True, type=Path, help="reference supercell structure"
    )
    parser.add_argument("--force-constants", required=True, type=Path, help="phonopy FC2 text file")
    parser.add_argument("--output", type=Path, default=Path("phonon-band.png"))
    parser.add_argument("--format", default="vasp", help="phonopy structure format (default: vasp)")
    parser.add_argument("--npoints", type=int, default=101, help="points per path segment")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def _load_phonopy(path: Path, fmt: str) -> Phonopy:
    if not path.is_file():
        raise FileNotFoundError(
            f"structure file does not exist: {path}. "
            "For the ALAMODE examples use reference.vasp, not POSCAR."
        )
    cell, _ = read_crystal_structure(filename=str(path), interface_mode=fmt)
    if cell is None:
        raise ValueError(f"phonopy could not read a structure from {path} (format={fmt!r})")
    # The supplied cell is the FC2 reference supercell.  Identity keeps its
    # atom order intact; phonopy's automatic primitive finder reduces it.
    return Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")


def _read_force_constants(phonon: Phonopy, source: Path) -> None:
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(source))


def main() -> None:
    args = _parser().parse_args()
    if args.npoints < 2:
        raise ValueError("--npoints must be at least 2")
    phonon = _load_phonopy(args.supercell, args.format)
    _read_force_constants(phonon, args.force_constants)

    primitive = phonon.primitive
    positions = np.asarray(primitive.scaled_positions)
    numbers = np.asarray(primitive.numbers)
    structure = (np.asarray(primitive.cell), positions, numbers)
    path_data = seekpath.get_path(structure, recipe="hpkot")
    point_coords = path_data["point_coords"]
    segments = path_data["path"]
    paths = []
    connections = []
    for index, (start, end) in enumerate(segments):
        q0, q1 = np.asarray(point_coords[start]), np.asarray(point_coords[end])
        paths.append(np.linspace(q0, q1, args.npoints))
        connections.append(index + 1 < len(segments) and end == segments[index + 1][0])

    phonon.run_band_structure(paths, path_connections=connections)
    band_data = phonon.band_structure
    distances = band_data.distances
    frequencies = band_data.frequencies
    tick_labels: dict[float, str] = {}
    for (start, end), distance in zip(segments, distances, strict=True):
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            if location in tick_labels and tick_labels[location] != label:
                tick_labels[location] += f"|{label}"
            else:
                tick_labels[location] = label

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for distance, segment_frequencies in zip(distances, frequencies, strict=True):
        for branch in np.asarray(segment_frequencies).T:
            axis.plot(distance, branch, color="#164e63", linewidth=1.05)
    for location in tick_labels:
        axis.axvline(location, color="#cbd5e1", linewidth=0.65, zorder=0)
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.set_xticks(list(tick_labels), list(tick_labels.values()))
    axis.set_ylabel("Frequency (THz)")
    axis.set_xlim(float(distances[0][0]), float(distances[-1][-1]))
    if args.ylim:
        axis.set_ylim(*args.ylim)
    axis.set_title(f"Phonon band structure: {args.supercell.name}")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(
        f"wrote {args.output} ({len(primitive)} primitive atoms, {sum(map(len, paths))} q-points)"
    )


if __name__ == "__main__":
    main()
