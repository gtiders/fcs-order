"""Plot phonopy/SeeK-path bands for harmonic and loop-SCPH FC2 files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from matplotlib import colormaps
from matplotlib.lines import Line2D
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure
from ase.io import read as ase_read
from mlfcs import read_hdf5

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "input" / "reference.vasp"


def _phonon(force_constants: Path) -> Phonopy:
    cell, _ = read_crystal_structure(filename=str(REFERENCE), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"could not read {REFERENCE}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    hdf5 = force_constants.with_name("mlfcs.h5")
    if hdf5.is_file():
        target = ase_read(REFERENCE)
        model = read_hdf5(hdf5).realize(reference=target)
        phonon.force_constants = model.materialize(2)
    else:
        phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants))
    return phonon


def _paths(phonon: Phonopy):
    primitive = phonon.primitive
    structure = (
        np.asarray(primitive.cell),
        np.asarray(primitive.scaled_positions),
        np.asarray(primitive.numbers),
    )
    data = seekpath.get_path(structure, recipe="hpkot")
    points = data["point_coords"]
    paths = []
    connections = []
    labels = []
    for i, (start, end) in enumerate(data["path"]):
        q0, q1 = np.asarray(points[start]), np.asarray(points[end])
        paths.append(np.linspace(q0, q1, 101))
        connections.append(i + 1 < len(data["path"]) and end == data["path"][i + 1][0])
        labels.append((start, end))
    return paths, connections, labels


def _band_data(source: Path):
    phonon = _phonon(source)
    paths, connections, labels = _paths(phonon)
    phonon.run_band_structure(paths, path_connections=connections)
    return phonon.band_structure, labels


def main() -> None:
    harmonic_source = (
        HERE.parent.parent
        / "fitting"
        / "K4As4Pt2"
        / "results"
        / "three-body"
        / "FORCE_CONSTANTS_2ND"
    )
    temperature_sources = sorted(
        (
            int(directory.name[1:]),
            directory / "FORCE_CONSTANTS_2ND",
        )
        for directory in (HERE / "results").glob("T*")
        if directory.name[1:].isdigit() and (directory / "FORCE_CONSTANTS_2ND").is_file()
    )
    if not temperature_sources:
        raise FileNotFoundError("no results/T*/FORCE_CONSTANTS_2ND files were found")

    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    harmonic, labels = _band_data(harmonic_source)
    ticks: dict[float, str] = {}
    for distance, (start, end) in zip(harmonic.distances, labels, strict=True):
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            ticks.setdefault(location, _pretty_label(label))
    for distance, frequencies in zip(harmonic.distances, harmonic.frequencies, strict=True):
        axis.plot(distance, np.asarray(frequencies), color="#6b7280", linewidth=1.0, alpha=0.8,
                  linestyle="--", zorder=1)

    colors = colormaps["turbo"](np.linspace(0.12, 0.88, len(temperature_sources)))
    legend = [Line2D([0], [0], color="#6b7280", linestyle="--", linewidth=1.3, label="Harmonic")]
    for (temperature, source), color in zip(temperature_sources, colors, strict=True):
        band, labels = _band_data(source)
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, np.asarray(frequencies), color=color, linewidth=1.05, alpha=0.88,
                      zorder=2)
        legend.append(Line2D([0], [0], color=color, linewidth=1.8, label=f"{temperature} K"))

    for location in ticks:
        axis.axvline(location, color="#d1d5db", linewidth=0.55, zorder=0)
    axis.axhline(0.0, color="#374151", linewidth=0.7, linestyle=":", zorder=0)
    axis.set_xticks(list(ticks), list(ticks.values()))
    axis.set_xlim(float(harmonic.distances[0][0]), float(harmonic.distances[-1][-1]))
    axis.set_xlabel("High-symmetry path")
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("K4As4Pt2 harmonic and loop-SCPH phonon bands")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.5)
    axis.legend(handles=legend, loc="best", frameon=False, ncol=2)
    figure.savefig(
        HERE / "results" / "phonopy-seekpath-harmonic-vs-scph.png", dpi=220, bbox_inches="tight"
    )


def _pretty_label(label: str) -> str:
    return {
        "GAMMA": r"$\Gamma$",
        "SIGMA_0": r"$\Sigma_0$",
        "E_0": r"$E_0$",
    }.get(label, label)


if __name__ == "__main__":
    main()
