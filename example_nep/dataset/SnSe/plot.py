#!/usr/bin/env python3
"""Plot the SnSe phonon band structure from MLFCS FC2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supercell", type=Path, default=CASE / "mlfcs/fd_supercell.vasp")
    parser.add_argument(
        "--force-constants",
        type=Path,
        default=CASE / "mlfcs/fc2/FORCE_CONSTANTS_2ND",
    )
    parser.add_argument("--output", type=Path, default=CASE / "mlfcs/fc2/phonon-band.png")
    parser.add_argument("--npoints", type=int, default=101)
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    for path in (args.supercell, args.force_constants):
        if not path.is_file():
            raise FileNotFoundError(path)
    cell, _ = read_crystal_structure(filename=str(args.supercell), interface_mode="vasp")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(args.force_constants))
    primitive = phonon.primitive
    structure = (
        np.asarray(primitive.cell),
        np.asarray(primitive.scaled_positions),
        np.asarray(primitive.numbers),
    )
    path_data = seekpath.get_path(structure, recipe="hpkot")
    labels = path_data["point_coords"]
    paths = []
    connections = []
    for index, (start, end) in enumerate(path_data["path"]):
        paths.append(
            np.linspace(labels[start], labels[end], args.npoints)
        )
        connections.append(
            index + 1 < len(path_data["path"])
            and end == path_data["path"][index + 1][0]
        )
    phonon.run_band_structure(paths, path_connections=connections)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    tick_labels: dict[float, str] = {}
    for (start, end), distance, frequencies in zip(
        path_data["path"], phonon.band_structure.distances,
        phonon.band_structure.frequencies, strict=True
    ):
        for branch in np.asarray(frequencies).T:
            axis.plot(distance, branch, color="#164e63", linewidth=1.05)
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            if location in tick_labels and tick_labels[location] != label:
                tick_labels[location] += f"|{label}"
            else:
                tick_labels[location] = label
    for location in tick_labels:
        axis.axvline(location, color="#cbd5e1", linewidth=0.65, zorder=0)
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.set_xticks(list(tick_labels), list(tick_labels.values()))
    axis.set_ylabel("Frequency (THz)")
    axis.set_xlim(float(phonon.band_structure.distances[0][0]),
                  float(phonon.band_structure.distances[-1][-1]))
    if args.ylim:
        axis.set_ylim(*args.ylim)
    axis.set_title("SnSe phonon band structure")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
