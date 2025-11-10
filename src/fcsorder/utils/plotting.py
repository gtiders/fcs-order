#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List


def plot_phband(
    na: int, nb: int, nc: int, primcell: str, fcs_orders: List[str]
) -> None:
    """
    Plot phonon band structure from multiple FORCE_CONSTANTS files.

    This tool generates a phonon band structure plot using the primitive cell
    structure and force constants from multiple FORCE_CONSTANTS files.
    Each dataset is plotted with a different color from a colormap.
    """
    # Import matplotlib here to avoid issues when not plotting
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Read the primitive cell structure
    from ase.io import read

    primcell_atoms = read(primcell)

    # Initialize Phonopy object
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.file_IO import parse_FORCE_CONSTANTS
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

    # Create supercell matrix
    import numpy as np

    supercell_matrix = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])

    # Get the k-point path for the band structure (same for all datasets)
    structure_tuple = (
        primcell_atoms.cell,
        primcell_atoms.get_scaled_positions(),
        primcell_atoms.numbers,
    )
    path = get_explicit_k_path(structure_tuple)

    # Create the plot with two subplots (band structure + legend)
    fig = plt.figure(figsize=(10, 4), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.15)
    ax_band = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # Hide axes for legend subplot
    ax_legend.axis("off")

    # Get colors from colormap
    colors = cm.tab10(np.linspace(0, 1, len(fcs_orders)))

    # Process each FORCE_CONSTANTS file
    for i, fcs_order in enumerate(fcs_orders):
        phonon = Phonopy(
            ase_to_phonopy(primcell_atoms), supercell_matrix=supercell_matrix
        )
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
    labels = path["explicit_kpoints_labels"]
    labels = [r"$\Gamma$" if m == "GAMMA" else m for m in labels]
    labels = [m.replace("_", "$_") + "$" if "_" in m else m for m in labels]

    # Use NumPy to filter out empty labels
    kcoords = np.array(path["explicit_kpoints_linearcoord"])  # (n_k,)
    label_array = np.array(labels, dtype=object)
    mask = label_array != ""
    tick_positions = kcoords[mask]
    tick_labels = label_array[mask].tolist()

    ax_band.set_xticks(tick_positions)
    ax_band.set_xticklabels(tick_labels, fontsize=11)
    for xp in tick_positions:
        ax_band.axvline(xp, color="0.6", linestyle="-", linewidth=0.8, alpha=0.7)

    # Add legend in separate subplot
    import os
    from matplotlib.lines import Line2D

    legend_labels = [os.path.basename(fcs) for fcs in fcs_orders]
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2.5, label=legend_labels[i])
        for i in range(len(fcs_orders))
    ]
    ax_legend.legend(
        handles=legend_elements,
        loc="center left",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Force Constants",
        title_fontsize=11,
    )

    # Adjust layout to avoid tight_layout warning
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)

    # Save the plot
    output_file = "phband.svg"
    plt.savefig(output_file, bbox_inches="tight")
    import typer

    typer.echo(f"Phonon band structure plot saved to {output_file}")
