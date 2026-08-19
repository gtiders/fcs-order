"""Plot the fitted K4As4Pt2 phonon band using a seekpath path."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercell", type=Path, default=ROOT / "input/reference.vasp")
    parser.add_argument("--force-constants", type=Path, default=ROOT / "results/three-body/FORCE_CONSTANTS_2ND")
    parser.add_argument("--output", type=Path, default=ROOT / "results/three-body/phonon-band.png")
    parser.add_argument("--npoints", type=int, default=101)
    args = parser.parse_args()
    cell, _ = read_crystal_structure(filename=str(args.supercell), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read structure: {args.supercell}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(args.force_constants))
    primitive = phonon.primitive
    path = seekpath.get_path((np.asarray(primitive.cell), primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], args.npoints) for a, b in path["path"]]
    connections = [i + 1 < len(paths) and path["path"][i][1] == path["path"][i + 1][0] for i in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=connections)
    band = phonon.band_structure
    ticks, labels = [], []
    for (a, b), distance in zip(path["path"], band.distances, strict=True):
        ticks.extend((float(distance[0]), float(distance[-1])))
        labels.extend((a, b))
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for distance, values in zip(band.distances, band.frequencies, strict=True):
        axis.plot(distance, np.asarray(values), color="#a34e25", linewidth=0.8)
    axis.set_xticks(ticks, labels)
    axis.set_ylabel("Frequency (THz)")
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    axis.set_title("K4As4Pt2 fitted phonon bands")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
