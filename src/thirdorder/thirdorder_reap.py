import click
import numpy as np
from .thirdorder_vasp import _prepare_calculation, build_unpermutation, read_forces
from .thirdorder_common import H, write_ifcs
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
@click.argument("vaspruns", type=click.Path(exists=True), nargs=-1, required=True)
def reap(na, nb, nc, cutoff, vaspruns):
    """Collect forces and write 3RD force constants (reap phase)."""
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation(
        na, nb, nc, cutoff
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} triplet equivalence classes found")
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print(f"- {nruns} DFT runs are needed")

    if len(vaspruns) != nruns:
        raise click.ClickException(
            f"Error: {nruns} vasprun.xml files were expected, got {len(vaspruns)}"
        )

    print("Reading the forces")
    p = build_unpermutation(sposcar)
    forces = []
    for f in vaspruns:
        forces.append(read_forces(f)[p, :])
        print(f"- {f} read successfully")
        res = forces[-1].mean(axis=0)
        print("- \t Average force:")
        print(f"- \t {res} eV/(A * atom)")
    print("Computing an irreducible set of anharmonic force constants")
    phipart = np.zeros((3, nirred, ntot))
    for i, e in enumerate(list4):
        for n in range(4):
            isign = (-1) ** (n // 2)
            jsign = -((-1) ** (n % 2))
            number = nirred * n + i
            phipart[:, i, :] -= isign * jsign * forces[number].T
    phipart /= 400.0 * H * H
    print("Reconstructing the full array")
    phifull = thirdorder_core.reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar)
    print("Writing the constants to FORCE_CONSTANTS_3RD")
    write_ifcs(
        phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_3RD"
    )
