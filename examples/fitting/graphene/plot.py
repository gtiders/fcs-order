"""Plot ASR and Born-Huang/Huang graphene phonon bands."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def _bands(force_constants: Path):
    cell, _ = read_crystal_structure(filename=str(ROOT / "input/reference.vasp"), interface_mode="vasp")
    if cell is None:
        raise ValueError("cannot read graphene reference supercell")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants))
    primitive = phonon.primitive
    path = seekpath.get_path((np.asarray(primitive.cell), primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], 101) for a, b in path["path"]]
    links = [i + 1 < len(paths) and path["path"][i][1] == path["path"][i + 1][0] for i in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=links)
    return path, phonon.band_structure


def main() -> None:
    path, asr = _bands(ROOT / "results/asr/FORCE_CONSTANTS_2ND")
    _, constrained = _bands(ROOT / "results/born-huang-huang/FORCE_CONSTANTS_2ND")
    ticks, labels = [], []
    for (a, b), distance in zip(path["path"], asr.distances, strict=True):
        ticks.extend((float(distance[0]), float(distance[-1])))
        labels.extend((a, b))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for axis, band, title, color in zip(axes, (asr, constrained), ("ASR", "Born-Huang + Huang"), ("#176b87", "#a34e25"), strict=True):
        for distance, values in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, np.asarray(values), color=color, linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(ticks, labels)
        axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
        axis.set_xlabel("Seekpath high-symmetry path")
    axes[0].set_ylabel("Frequency (THz)")
    output = ROOT / "results/phonon-bands.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
