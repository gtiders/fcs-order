"""Module for generating rattled structures using Monte Carlo rattle."""

import numpy as np
from scipy.special import erf
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.io import write
import typer

from fcsorder.io.io_abstraction import read_atoms
from fcsorder.genstr.tools import plot_distributions


def generate_mc_rattled_structures(
    atoms: Atoms,
    n_structures: int,
    rattle_std: float,
    d_min: float,
    seed: int = 42,
    **kwargs,
) -> list[Atoms]:
    r"""Returns list of Monte Carlo rattled configurations.

    Rattling atom `i` is carried out as a Monte Carlo move that is
    accepted with a probability determined from the minimum
    interatomic distance :math:`d_{ij}`.  If :math:`\min(d_{ij})` is
    smaller than :math:`d_{min}` the move is only accepted with a low
    probability.

    This process is repeated for each atom a number of times meaning
    the magnitude of the final displacements is not *directly*
    connected to `rattle_std`.

    Warning
    -------
    Repeatedly calling this function *without* providing different
    seeds will yield identical or correlated results. To avoid this
    behavior it is recommended to specify a different seed for each
    call to this function.

    Notes
    ------
    The procedure implemented here might not generate a symmetric
    distribution for the displacements `kwargs` will be forwarded to
    `mc_rattle` (see user guide for a detailed explanation)

    The displacements generated will roughly be `n_iter**0.5 * rattle_std`
    for small values of `n_iter`.

    Parameters
    ----------
    atoms
        Prototype structure.
    n_structures
        Number of structures to generate.
    rattle_std
        Rattle amplitude (standard deviation in normal distribution);
        note this value is not connected to the final
        average displacement for the structures.
    d_min
        Interatomic distance used for computing the probability for each rattle move.
    seed
        Seed for setting up NumPy random state from which random numbers are generated.
    n_iter
        Number of Monte Carlo cycles (iterations), larger number of iterations will
        generate larger displacements (defaults to 10).

    Returns
    -------
        Generated structures.
    """
    rs = np.random.RandomState(seed)
    atoms_list = []
    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        seed = rs.randint(1, 1000000000)
        displacements = mc_rattle(atoms_tmp, rattle_std, d_min, seed=seed, **kwargs)
        atoms_tmp.positions += displacements
        atoms_list.append(atoms_tmp)
    return atoms_list


def _probability_mc_rattle(d: float, d_min: float, width: float):
    """Monte Carlo probability function as an error function.

    Parameters
    ----------
    d
        Value at which to evaluate function.
    d_min
        Center value for the error function.
    width
        Width of error function.
    """

    return (erf((d - d_min) / width) + 1.0) / 2


def mc_rattle(
    atoms: Atoms,
    rattle_std: float,
    d_min: float,
    width: float = 0.1,
    n_iter: int = 10,
    max_attempts: int = 5000,
    max_disp: float = 2.0,
    active_atoms: list[int] = None,
    nbr_cutoff: float = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate displacements using the Monte Carlo rattle method.

    Parameters
    ----------
    atoms
        Prototype structure.
    rattle_std
        Rattle amplitude (standard deviation in normal distribution).
    d_min
        Interatomic distance used for computing the probability for each rattle
        move. Center position of the error function.
    width
        Width of the error function.
    n_iter
        Number of Monte Carlo cycle.
    max_disp
        Rattle moves that yields a displacement larger than max_disp will
        always be rejected. This occurs rarely and is more used as a safety precaution
        for not generating structures where two or more have swapped positions.
    max_attempts
        Limit for how many attempted rattle moves are allowed a single atom;
        if this limit is reached an `Exception` is raised.
    active_atoms
        List of indices of atoms that should undergo Monte Carlo rattling.
    nbr_cutoff
        The cutoff used to construct the neighborlist used for checking
        interatomic distances, defaults to `2 * d_min`.
    seed
        Seed for setting up NumPy random state from which random numbers are
        generated.

    Returns
    -------
        Atomic displacements (`Nx3`).
    """

    # setup
    rs = np.random.RandomState(seed)

    if nbr_cutoff is None:
        nbr_cutoff = 2 * d_min

    if active_atoms is None:
        active_atoms = range(len(atoms))

    atoms_rattle = atoms.copy()
    reference_positions = atoms_rattle.get_positions()
    nbr_list = NeighborList(
        [nbr_cutoff / 2] * len(atoms_rattle),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nbr_list.update(atoms_rattle)

    # run Monte Carlo
    for _ in range(n_iter):
        for i in active_atoms:
            i_nbrs = np.setdiff1d(nbr_list.get_neighbors(i)[0], [i])
            for n in range(max_attempts):
                # generate displacement
                delta_disp = rs.normal(0.0, rattle_std, 3)
                atoms_rattle.positions[i] += delta_disp
                disp_i = atoms_rattle.positions[i] - reference_positions[i]

                # if total displacement of atom is greater than max_disp, then reject delta_disp
                if np.linalg.norm(disp_i) > max_disp:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
                    continue

                # compute min distance
                if len(i_nbrs) == 0:
                    min_distance = np.inf
                else:
                    min_distance = np.min(
                        atoms_rattle.get_distances(i, i_nbrs, mic=True)
                    )

                # accept or reject delta_disp
                if _probability_mc_rattle(min_distance, d_min, width) > rs.rand():
                    # accept delta_disp
                    break
                else:
                    # revert delta_disp
                    atoms_rattle[i].position -= delta_disp
            else:
                raise Exception(f"Exceeded the maximum number of attempts for atom {i}")
    displacements = atoms_rattle.positions - reference_positions
    return displacements


def monte_rattle_cli(
    sposcar: str = typer.Argument(..., help="Path to SPOSCAR/POSCAR structure file"),
    n_structures: int = typer.Option(
        10, "--n-structures", "-n", help="Number of structures to generate"
    ),
    rattle_std: float = typer.Option(
        0.05, "--rattle-std", help="Rattle amplitude (standard deviation)"
    ),
    d_min: float = typer.Option(
        1.0, "--d-min", help="Minimum interatomic distance used in MC acceptance"
    ),
    width: float = typer.Option(
        0.1, "--width", help="Width of the error function for MC acceptance"
    ),
    n_iter: int = typer.Option(10, "--n-iter", help="Number of Monte Carlo cycles"),
    max_disp: float = typer.Option(
        2.0, "--max-disp", help="Maximum allowed displacement magnitude for any atom"
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for generating structures"
    ),
    output_format: str = typer.Option(
        "vasp",
        "--format",
        "-f",
        help="Output format: one of vasp, cif, qe, xyz",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help=(
            "Optional filename prefix. If set, all structures are named "
            "'<prefix><index>.<ext>' with zero-padded running index "
            "and no additional labels."
        ),
    ),
    min_volume: float | None = typer.Option(
        0.9,
        "--min-volume",
        help=(
            "Minimum volume ratio for applying volumetric strain when --eps is enabled."
        ),
    ),
    max_volume: float | None = typer.Option(
        1.05,
        "--max-volume",
        help=(
            "Maximum volume ratio for applying volumetric strain when --eps is enabled."
        ),
    ),
    eps: bool = typer.Option(
        False,
        "--eps/--no-eps",
        help=(
            "Enable applying a random volumetric strain to each "
            "generated structure, with volume ratio drawn uniformly "
            "from [min_volume, max_volume]."
        ),
    ),
):
    """Generate Monte Carlo rattled structures.

    Structures are generated using the ``mc_rattle`` algorithm with the
    specified ``rattle_std`` and ``d_min`` parameters. The input
    structure is read from ``sposcar``. Optionally a random volumetric
    strain can be applied to each generated structure.
    """

    # normalize and validate output format
    fmt = output_format.lower()
    valid_formats = {"vasp", "cif", "qe", "xyz"}
    if fmt not in valid_formats:
        raise typer.BadParameter(
            f"Invalid format '{output_format}'. Must be one of: {', '.join(sorted(valid_formats))}."
        )

    # map CLI format keyword to ASE writer format string and file extension
    ase_format_map = {
        "vasp": ("vasp", "vasp"),
        "cif": ("cif", "cif"),
        "qe": ("espresso-in", "in"),
        "xyz": ("xyz", "xyz"),
    }
    ase_format, file_ext = ase_format_map[fmt]

    atoms = read_atoms(sposcar)
    reference_positions = atoms.get_positions().copy()

    # validate volumetric strain options
    if eps:
        if min_volume is None or max_volume is None:
            raise typer.BadParameter(
                "When --eps is enabled you must also specify both --min-volume and --max-volume."
            )
        if min_volume <= 0 or max_volume <= 0:
            raise typer.BadParameter("min_volume and max_volume must be positive.")
        if min_volume > max_volume:
            raise typer.BadParameter("min_volume cannot be larger than max_volume.")

    # determine zero-padding width for indices (1-based)
    total_structures = n_structures
    if total_structures > 0:
        idx_width = len(str(total_structures))
    else:
        idx_width = 1
    # use 1-based indices so the first file is ...001 when total_structures >= 10
    global_index = 1

    # generate MC-rattled structures
    structures = generate_mc_rattled_structures(
        atoms=atoms,
        n_structures=n_structures,
        rattle_std=rattle_std,
        d_min=d_min,
        seed=seed,
        width=width,
        n_iter=n_iter,
        max_disp=max_disp,
    )

    for i, structure in enumerate(structures):
        # apply random volumetric strain if requested
        if eps:
            volume_ratio = np.random.uniform(min_volume, max_volume)
            scale = volume_ratio ** (1.0 / 3.0)
            structure.set_cell(structure.get_cell() * scale, scale_atoms=True)

        if prefix is not None:
            # prefixed filenames: <prefix><1..N> with zero-padding
            filename = f"{prefix}{global_index:0{idx_width}d}.{file_ext}"
            global_index += 1
        else:
            # default filenames: mc_rattle_<1..N> with zero-padding
            index = i + 1
            filename = f"mc_rattle_{index:0{idx_width}d}.{file_ext}"
        write(filename, structure, format=ase_format)

    # use rattle_std as a label in the distribution plot filename
    plot_distributions(structures, reference_positions, rattle_std)
