import typer
from .commands.sow3 import sow3
from .commands.sow4 import sow4
from .commands.reap3 import reap3
from .commands.reap4 import reap4
from .commands.mlp2 import app as mlp2_app
from .commands.mlp3 import app as mlp3_app
from .commands.mlp4 import app as mlp4_app
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
cli.command(name="phonon_sow")(phononrattle)


__all__ = ["cli"]
