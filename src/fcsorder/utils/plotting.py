#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .io_abstraction import parse_supercell_matrix, read as io_read


def plot_phband(
    supercell: List[int],
    poscar: str,
    fcs_orders: List[str],
    labels: List[str] | None = None,
) -> None:
    """
    Plot phonon band structure from multiple FORCE_CONSTANTS files.

    This tool generates a phonon band structure plot using the primitive cell
    structure and force constants from multiple FORCE_CONSTANTS files.
    Each dataset is plotted with a different color from a colormap.

    poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ).
    supercell: Supercell specification as 3 or 9 integers (same semantics as mlp3),
               e.g. [na, nb, nc] or a full 3x3 flattened matrix.
    labels: Optional list of legend labels. If provided, its length must equal the number of FORCE_CONSTANTS files.
            If not provided, defaults to the basenames of the files in fcs_orders.
    """

    poscar = io_read(poscar)

    # Initialize Phonopy object
    try:
        from phonopy import Phonopy
        from phonopy.file_IO import parse_FORCE_CONSTANTS
        from phonopy.structure.atoms import PhonopyAtoms
        from seekpath import get_explicit_k_path
    except ImportError:
        import typer

        typer.echo(
            "Required dependencies not found. Please install phonopy and seekpath.",
            err=True,
        )
        raise typer.Exit(1)

    def ase_to_phonopy(atoms, **kwargs):
        return PhonopyAtoms(
            numbers=atoms.numbers, cell=atoms.cell, positions=atoms.positions, **kwargs
        )

    # Create supercell matrix (accept 3 or 9 integers like mlp3)
    supercell_matrix = parse_supercell_matrix(supercell)

    # Get the k-point path for the band structure (same for all datasets)
    structure_tuple = (
        poscar.cell,
        poscar.get_scaled_positions(),
        poscar.numbers,
    )
    path = get_explicit_k_path(structure_tuple)

    # Create the plot
    fig = plt.figure(figsize=(10, 4), dpi=140)
    ax_band = fig.add_subplot(111)

    # Get colors from colormap
    colors = cm.tab10(np.linspace(0, 1, len(fcs_orders)))

    # Validate labels length if provided
    if labels is not None and len(labels) != len(fcs_orders):
        raise ValueError("labels length must equal number of FORCE_CONSTANTS files")

    # Process each FORCE_CONSTANTS file
    for i, fcs_order in enumerate(fcs_orders):
        phonon = Phonopy(ase_to_phonopy(poscar), supercell_matrix=supercell_matrix)
        phonon.force_constants = parse_FORCE_CONSTANTS(fcs_order)

        # Calculate the band structure
        phonon.run_band_structure([path["explicit_kpoints_rel"]])
        band = phonon.get_band_structure_dict()

        # Plot bands using NumPy arrays (no pandas)
        freqs = np.array(band["frequencies"][0])  # shape: (n_k, n_bands)
        kline = np.array(path["explicit_kpoints_linearcoord"])  # shape: (n_k,)

        # Plot each band with the same color for this dataset
        n_bands = freqs.shape[1] if freqs.ndim == 2 else 0
        for b in range(n_bands):
            ax_band.plot(kline, freqs[:, b], color=colors[i], alpha=0.8, linewidth=1.2)

    # Beautify the band structure plot
    ax_band.set_xlim(
        path["explicit_kpoints_linearcoord"].min(),
        path["explicit_kpoints_linearcoord"].max(),
    )
    ax_band.set_ylabel("Frequency (THz)", fontsize=11, fontweight="bold")
    ax_band.set_xlabel("Wave Vector", fontsize=11, fontweight="bold")
    ax_band.tick_params(axis="both", which="major", labelsize=10)
    ax_band.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add horizontal line at 0 frequency
    ax_band.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Beautify the labels on the x-axis
    kpt_labels = path["explicit_kpoints_labels"]
    kpt_labels = [r"$\\Gamma$" if m == "GAMMA" else m for m in kpt_labels]
    kpt_labels = [m.replace("_", "$_") + "$" if "_" in m else m for m in kpt_labels]

    # Use NumPy to filter out empty labels
    kcoords = np.array(path["explicit_kpoints_linearcoord"])  # (n_k,)
    label_array = np.array(kpt_labels, dtype=object)
    mask = label_array != ""
    tick_positions = kcoords[mask]
    tick_labels = label_array[mask].tolist()

    ax_band.set_xticks(tick_positions)
    ax_band.set_xticklabels(tick_labels, fontsize=11)
    for xp in tick_positions:
        ax_band.axvline(xp, color="0.6", linestyle="-", linewidth=0.8, alpha=0.7)

    legend_labels = (
        labels if labels is not None else [os.path.basename(fcs) for fcs in fcs_orders]
    )
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(fcs_orders) + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_band, ticks=np.arange(len(fcs_orders)), pad=0.02)
    cb.ax.set_yticklabels(legend_labels)
    cb.set_label("Force Constants", fontsize=11, fontweight="bold")

    # Adjust layout to avoid tight_layout warning
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)

    # Save the plot
    output_file = "phband.svg"
    plt.savefig(output_file, bbox_inches="tight")
    typer.echo(f"Phonon band structure plot saved to {output_file}")
