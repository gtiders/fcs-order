import typer
from .commands.sow3 import sow3
from .commands.sow4 import sow4
from .commands.reap3 import reap3
from .commands.reap4 import reap4
from .commands.mlp2 import app as mlp2_app
from .commands.mlp3 import app as mlp3_app
from .commands.mlp4 import app as mlp4_app
from .commands.scph import app as scph_app
from .commands.phonon_sow import phononrattle

cli = typer.Typer(help="Force constants calculation tool for VASP")


@cli.command(name="sow3")
def sow3_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
):
    """
    Generate 3RD.POSCAR.* files for 3-phonon calculations.
    """
    sow3(na, nb, nc, cutoff)


@cli.command(name="sow4")
def sow4_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
):
    """
    Generate 4TH.POSCAR.* files for 4-phonon calculations.
    """
    sow4(na, nb, nc, cutoff)


@cli.command(name="reap3")
def reap3_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
    is_sparse: bool = typer.Option(
        False,
        "--is-sparse",
        help="Use sparse tensor method for memory efficiency",
    ),
    vaspruns: list[str] = typer.Argument(
        ..., help="Paths to vasprun.xml files from VASP calculations"
    ),
):
    """
    Extract 3-phonon force constants from VASP calculation results.
    """
    reap3(na, nb, nc, cutoff, vaspruns, is_sparse)


@cli.command(name="reap4")
def reap4_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
    is_sparse: bool = typer.Option(
        False,
        "--is-sparse",
        help="Use sparse tensor method for memory efficiency",
    ),
    vaspruns: list[str] = typer.Argument(
        ..., help="Paths to vasprun.xml files from VASP calculations"
    ),
):
    """
    Extract 4-phonon force constants from VASP calculation results.
    """
    reap4(na, nb, nc, cutoff, vaspruns, is_sparse)


@cli.command(name="plot_phband")
def plot_phband_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    primcell: str = typer.Argument(
        ..., help="Path to the primitive cell file (e.g., POSCAR)"
    ),
    fcs_orders: list[str] = typer.Argument(
        ..., help="Paths to the FORCE_CONSTANTS files"
    ),
):
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

    # Create the plot
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=140)

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

        # Create DataFrame for this dataset
        import pandas as pd

        df = pd.DataFrame(band["frequencies"][0])
        df.index = path["explicit_kpoints_linearcoord"]

        # Plot each band with the same color for this dataset
        for col in df.columns:
            ax.plot(df.index, df[col], color=colors[i], alpha=0.8)

    ax.set_xlim(
        path["explicit_kpoints_linearcoord"].min(),
        path["explicit_kpoints_linearcoord"].max(),
    )
    ax.set_ylabel("Frequency (THz)")

    # Beautify the labels on the x-axis
    labels = path["explicit_kpoints_labels"]
    labels = [r"$\Gamma$" if m == "GAMMA" else m for m in labels]
    labels = [m.replace("_", "$_") + "$" if "_" in m else m for m in labels]
    df_path = pd.DataFrame(
        dict(labels=labels, positions=path["explicit_kpoints_linearcoord"])
    )
    df_path.drop(df_path.index[df_path.labels == ""], axis=0, inplace=True)
    ax.set_xticks(df_path.positions)
    ax.set_xticklabels(df_path.labels)
    for xp in df_path.positions:
        ax.axvline(xp, color="0.8")

    # Add legend with file names
    import os

    legend_labels = [os.path.basename(fcs) for fcs in fcs_orders]
    # Create legend entries with corresponding colors
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=legend_labels[i])
        for i in range(len(fcs_orders))
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # Make room for the legend on the right side

    # Save the plot
    output_file = "phband.svg"
    plt.savefig(output_file)
    typer.echo(f"Phonon band structure plot saved to {output_file}")


# Add mlp2 as a subcommand group
cli.add_typer(
    mlp2_app,
    name="mlp2",
    help="Calculate 2-phonon force constants using machine learning potentials",
)
# Add mlp3 as a subcommand group
cli.add_typer(
    mlp3_app,
    name="mlp3",
    help="Calculate 3-phonon force constants using machine learning potentials",
)
# Add mlp4 as a subcommand group
cli.add_typer(
    mlp4_app,
    name="mlp4",
    help="Calculate 4-phonon force constants using machine learning potentials",
)
# Add scph as a subcommand group
cli.add_typer(
    scph_app,
    name="scph",
    help="Run self-consistent phonon calculations using machine learning potentials",
)
cli.command(name="phonon_sow")(phononrattle)


__all__ = ["cli"]
