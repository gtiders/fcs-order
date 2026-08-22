"""Plot phonopy/SeeK-path bands for harmonic and loop-SCPH FC2 files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent.parent / "finite-difference" / "K4As4Pt2" / "input" / "supercell.vasp"


def _phonon(force_constants: Path) -> Phonopy:
    cell, _ = read_crystal_structure(filename=str(REFERENCE), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"could not read {REFERENCE}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
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
    sources = [
        (
            "Harmonic",
            HERE.parent.parent
            / "fitting"
            / "K4As4Pt2"
            / "results"
            / "three-body"
            / "FORCE_CONSTANTS_2ND",
        )
    ]
    sources.extend(
        (f"Loop-SCPH {temperature} K", HERE / "results" / f"T{temperature}" / "FORCE_CONSTANTS_2ND")
        for temperature in (300, 600, 900)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for axis, (title, source) in zip(axes.flat, sources, strict=True):
        band, labels = _band_data(source)
        ticks: dict[float, str] = {}
        for distance, (start, end) in zip(band.distances, labels, strict=True):
            for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
                ticks.setdefault(location, _pretty_label(label))
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            for branch in np.asarray(frequencies).T:
                axis.plot(distance, branch, color="#2563eb", linewidth=0.9)
        axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
        if title.startswith("Loop-SCPH"):
            temperature = int(title.split()[-2])
            status = json.loads((HERE / "results" / f"T{temperature}" / "history.json").read_text())
            if not status["converged"]:
                title += " (not converged)"
        axis.set_title(title)
        axis.set_xticks(list(ticks), list(ticks.values()))
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
        axis.set_xlabel("SeeK-path")
    axes[0, 0].set_ylabel("Frequency (THz)")
    axes[1, 0].set_ylabel("Frequency (THz)")
    figure.savefig(
        HERE / "results" / "phonopy-seekpath-harmonic-vs-scph.png", dpi=180, bbox_inches="tight"
    )


def _pretty_label(label: str) -> str:
    return {
        "GAMMA": r"$\Gamma$",
        "SIGMA_0": r"$\Sigma_0$",
        "E_0": r"$E_0$",
    }.get(label, label)


if __name__ == "__main__":
    main()
