import typer
from .commands.sow import sow
from .commands.reap import reap
from .commands.mlp2 import app as mlp2_app
from .commands.mlp3 import app as mlp3_app
from .commands.mlp4 import app as mlp4_app
from .commands.scph import app as scph_app
from .commands.generate_phonon_rattled_structures import (
    generate_phonon_rattled_structures,
)
from .utils.plotting import plot_phband

cli = typer.Typer(help="Force constants calculation tool for VASP")


@cli.command(name="sow")
def sow_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
    order: int = typer.Option(3, "--order", help="Order of IFCs to generate: 3 or 4"),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Generate displaced POSCAR files for 3-phonon (order=3) or 4-phonon (order=4) calculations.
    """
    sow(na, nb, nc, cutoff, order, poscar)


@cli.command(name="reap")
def reap_command(
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
    order: int = typer.Option(3, "--order", help="Order of IFCs to extract: 3 or 4"),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Extract 3-phonon (order=3) or 4-phonon (order=4) force constants from VASP calculation results.
    """
    return reap(na, nb, nc, cutoff, vaspruns, is_sparse, order, poscar)


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

    Implementation is in utils.plotting.plot_phband.
    """

    plot_phband(na, nb, nc, primcell, fcs_orders)


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
cli.command(name="generate_phonon_rattled_structures")(
    generate_phonon_rattled_structures
)


__all__ = ["cli"]
