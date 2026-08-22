"""Plot KCl harmonic and SSCHA band comparisons from regenerated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES,
    RESULTS,
    bands,
    harmonic_phonopy,
)
from matplotlib.lines import Line2D
from phonopy.file_IO import parse_FORCE_CONSTANTS


def plot(band_data: dict[str, tuple], output: Path) -> None:
    colors = {
        "phonopy harmonic": "#5b8f8a",
        "phonopy SSCHA": "#326b68",
        "MLFCS harmonic": "#d69a73",
        "MLFCS SSCHA": "#c27b70",
    }
    panels = (
        ("phonopy: harmonic vs SSCHA", ("phonopy harmonic", "phonopy SSCHA")),
        ("MLFCS: harmonic vs SSCHA", ("MLFCS harmonic", "MLFCS SSCHA")),
        ("Harmonic: MLFCS vs phonopy", ("MLFCS harmonic", "phonopy harmonic")),
        ("SSCHA: MLFCS vs phonopy", ("MLFCS SSCHA", "phonopy SSCHA")),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharey=True, constrained_layout=True)
    for axis, (title, selected) in zip(axes.flat, panels, strict=True):
        selected = tuple(name for name in selected if name in band_data)
        if not selected:
            axis.set_visible(False)
            continue
        ticks = {}
        for name in selected:
            distances, frequencies, labels = band_data[name]
            linestyle = "--" if "SSCHA" in name else "-"
            for distance, values in zip(distances, frequencies, strict=True):
                for branch in np.asarray(values).T:
                    axis.plot(
                        distance,
                        branch,
                        color=colors[name],
                        linewidth=1.0,
                        linestyle=linestyle,
                    )
            if not ticks:
                for (start, end), distance in zip(labels, distances, strict=True):
                    ticks.setdefault(float(distance[0]), start)
                    ticks[float(distance[-1])] = end
        for location in ticks:
            axis.axvline(location, color="#d7e0e0", linewidth=0.6, zorder=0)
        axis.axhline(0.0, color="#718080", linewidth=0.6, linestyle=":")
        axis.set_xticks(list(ticks), list(ticks.values()))
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", color="#edf1f1", linewidth=0.5)
        legend = [
            Line2D(
                [0],
                [0],
                color=colors[name],
                linewidth=1.6,
                linestyle="--" if "SSCHA" in name else "-",
                label=name,
            )
            for name in selected
        ]
        axis.legend(handles=legend, frameon=False, fontsize=8, loc="best")
    axes[0, 0].set_ylabel("Frequency (THz)")
    axes[1, 0].set_ylabel("Frequency (THz)")
    axes[1, 0].set_xlabel("Wave vector")
    axes[1, 1].set_xlabel("Wave vector")
    figure.suptitle("KCl harmonic and SSCHA phonon-band comparisons", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES / "kcl_sscha_bands.png")
    args = parser.parse_args()
    phonon = harmonic_phonopy()
    phonopy_final = np.load(RESULTS / "phonopy_sscha_final_fc2.npy")
    mlfcs_final = parse_FORCE_CONSTANTS(filename=RESULTS / "FORCE_CONSTANTS_MLFCS_SSCHA")
    mlfcs_harmonic = parse_FORCE_CONSTANTS(filename=RESULTS / "FORCE_CONSTANTS_MLFCS_HARMONIC")
    band_data = {
        "phonopy harmonic": bands(phonon, phonon.force_constants),
        "phonopy SSCHA": bands(phonon, phonopy_final),
        "MLFCS harmonic": bands(phonon, mlfcs_harmonic),
        "MLFCS SSCHA": bands(phonon, mlfcs_final),
    }
    plot(band_data, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
