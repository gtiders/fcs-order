import click
import numpy as np
from . import fourthorder_core  # type: ignore
from .fourthorder_common import H, write_ifcs
from .fourthorder_vasp import _prepare_calculation, build_unpermutation, read_forces


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
    """
    Extract 4-phonon force constants from VASP calculation results.

    Parameters:
        na, nb, nc: supercell size, corresponding to expansion times in a, b, c directions
        cutoff: cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        vaspruns: paths to vasprun.xml files from VASP calculations, in order
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = _prepare_calculation(
        na, nb, nc, cutoff
    )
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    print(f"- {wedge.nlist} quartet equivalence classes found")
    list6 = wedge.build_list4()
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    nirred = len(list6)
    nruns = 8 * nirred
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
    for i, e in enumerate(list6):
        for n in range(8):
            isign = (-1) ** (n // 4)
            jsign = (-1) ** (n % 4 // 2)
            ksign = (-1) ** (n % 2)
            number = nirred * n + i
            phipart[:, i, :] -= isign * jsign * ksign * forces[number].T
    phipart /= 8000.0 * H * H * H
    print("Reconstructing the full array")
    phifull = fourthorder_core.reconstruct_ifcs(phipart, wedge, list6, poscar, sposcar)
    print("Writing the constants to FORCE_CONSTANTS_4TH")
    write_ifcs(
        phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_4TH"
    )
