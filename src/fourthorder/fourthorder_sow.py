import click
from . import fourthorder_core  # type: ignore
from .fourthorder_common import H, write_POSCAR, normalize_SPOSCAR, move_three_atoms
from .fourthorder_vasp import _prepare_calculation


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
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation(
        na, nb, nc, cutoff
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} quartet equivalence classes found")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred
    print(f"- {nruns} DFT runs are needed")
    print("Writing undisplaced coordinates to 4TH.SPOSCAR")
    write_POSCAR(normalize_SPOSCAR(sposcar), "4TH.SPOSCAR")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = "4TH.POSCAR.{{0:0{0}d}}".format(width)
    print("Writing displaced coordinates to 4TH.POSCAR.*")
    for i, e in enumerate(list6):
        for n in range(8):
            isign = (-1) ** (n // 4)
            jsign = (-1) ** (n % 4 // 2)
            ksign = (-1) ** (n % 2)
            number = nirred * n + i + 1
            dsposcar = normalize_SPOSCAR(
                move_three_atoms(
                    sposcar,
                    e[2],
                    e[5],
                    isign * H,
                    e[1],
                    e[4],
                    jsign * H,
                    e[0],
                    e[3],
                    ksign * H,
                )
            )
            filename = namepattern.format(number)
            write_POSCAR(dsposcar, filename)
