import typer


from fcsorder.phonon.mlp2 import app as mlp2_app
from fcsorder.phonon.scph import app as scph_app


from fcsorder.commands.sow import sow
from fcsorder.commands.reap import reap

from fcsorder.commands.mlp3 import app as mlp3_app
from fcsorder.commands.mlp4 import app as mlp4_app

from fcsorder.genstr.phonon_rattle import phonon_rattle_cli
from fcsorder.genstr.monte_rattle import monte_rattle_cli


cli = typer.Typer(help="Force constants calculation tool for VASP")


@cli.command(name="sow")
def sow_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
    order: int = typer.Option(
        3, "--order", "-r", help="Order of IFCs to generate: 3 or 4"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        "-p",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
    out_format: str = typer.Option(
        "poscar",
        "--out-format",
        "-f",
        help="Output format: poscar|vasp|cif|xyz (default: poscar)",
    ),
    out_dir: str = typer.Option(
        ".",
        "--out-dir",
        "-o",
        help="Directory to write generated structures (default: current directory)",
    ),
    name_template: str = typer.Option(
        "",
        "--name-template",
        "-t",
        help=(
            "Filename template for displaced structures. Available placeholders: "
            "{order} (e.g., 3RD/4TH), {phase} (disp), {index}, {index_padded}, {width}, {ext}. "
            "Example: '{order}.{phase}.{index_padded}.{ext}'"
        ),
    ),
    undisplaced_name: str = typer.Option(
        "",
        "--undisplaced-name",
        "-u",
        help=(
            "Filename for undisplaced structure. Placeholders supported: {order}, {phase} (structure), {ext}. "
            "Example: '{order}.structure.{ext}'"
        ),
    ),
):
    """
    Generate displaced POSCAR files for 3-phonon (order=3) or 4-phonon (order=4) calculations.
    """
    sow(
        na,
        nb,
        nc,
        cutoff,
        order,
        poscar,
        out_format,
        out_dir,
        name_template or None,
        undisplaced_name or None,
    )


@cli.command(name="reap")
def reap_command(
    na: int = typer.Argument(..., help="Supercell dimension along a direction"),
    nb: int = typer.Argument(..., help="Supercell dimension along b direction"),
    nc: int = typer.Argument(..., help="Supercell dimension along c direction"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff distance (negative for nearest neighbors, positive for distance in nm)",
    ),
    vaspruns: list[str] = typer.Argument(
        ...,
        help=(
            "Paths to force files readable by ASE (e.g., VASP vasprun.xml/OUTCAR, extxyz with forces, etc.). "
            "Order must match the displaced structure sequence."
        ),
    ),
    order: int = typer.Option(
        3, "--order", "-r", help="Order of IFCs to extract: 3 or 4"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        "-p",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Extract 3-phonon (order=3) or 4-phonon (order=4) force constants from VASP calculation results.
    """
    return reap(na, nb, nc, cutoff, vaspruns, order, poscar)


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

# add structure generation commands
cli.command(name="phonon_rattle")(phonon_rattle_cli)
cli.command(name="monte_rattle")(monte_rattle_cli)


__all__ = ["cli"]
