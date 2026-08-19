"""Plot the fitted Si phonon band along a seekpath high-symmetry path."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def main() -> None:
    supercell = ROOT / "anharmonic/input/supercell.vasp"
    source = ROOT / "anharmonic/results/FORCE_CONSTANTS_2ND"
    output = ROOT / "anharmonic/results/phonon-band.png"
    cell, _ = read_crystal_structure(filename=str(supercell), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read structure: {supercell}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(source))
    primitive = phonon.primitive
    path = seekpath.get_path((np.asarray(primitive.cell), primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], 101) for a, b in path["path"]]
    links = [i + 1 < len(paths) and path["path"][i][1] == path["path"][i + 1][0] for i in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=links)
    band = phonon.band_structure
    ticks, labels = [], []
    for (a, b), distance in zip(path["path"], band.distances, strict=True):
        ticks.extend((float(distance[0]), float(distance[-1])))
        labels.extend((a, b))
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for distance, values in zip(band.distances, band.frequencies, strict=True):
        axis.plot(distance, np.asarray(values), color="#176b87", linewidth=0.9)
    axis.set_xticks(ticks, labels)
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("Si fitted phonon bands")
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
