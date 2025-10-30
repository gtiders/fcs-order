import click
from .thirdorder_vasp import (
    _prepare_calculation,
    write_POSCAR,
    normalize_SPOSCAR,
)
from .thirdorder_common import H, move_two_atoms
from . import thirdorder_core  # type: ignore


@click.command()
@click.argument("na", type=int)
@click.argument("nb", type=int)
@click.argument("nc", type=int)
@click.option(
    "--cutoff",
    type=str,
    required=True,
    help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
)
def sow(na, nb, nc, cutoff):
    """
    Generate 3RD.POSCAR.* files for 3-phonon calculations.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation(
        na, nb, nc, cutoff
    )
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} triplet equivalence classes found")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print(f"- {nruns} DFT runs are needed")

    print("Writing undisplaced coordinates to 3RD.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "3RD.SPOSCAR")
    width = len(str(4 * (len(list4) + 1)))
    namepattern = f"3RD.POSCAR.{{:0{width}d}}"
    print("Writing displaced coordinates to 3RD.POSCAR.*")
    for i, e in enumerate(list4):
        for n in range(4):
            isign = (-1) ** (n // 2)
            jsign = -((-1) ** (n % 2))
            number = nirred * n + i + 1
            dsposcar = normalize_SPOSCAR(
                move_two_atoms(sposcar, e[1], e[3], isign * H, e[0], e[2], jsign * H)
            )
            filename = namepattern.format(number)
            write_POSCAR(dsposcar, filename)
