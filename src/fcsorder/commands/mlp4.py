#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party imports
import numpy as np
import typer
from rich.progress import track

# Local imports
from fcsorder.core import fourthorder_core
from fcsorder.core.fourthorder_core import prepare_calculation4
from fcsorder.calc.calculators import (
    make_dp,
    make_mtp,
    make_nep,
    make_polymp,
    make_tace,
)
from fcsorder.io.io_abstraction import get_atoms, read_atoms
from fcsorder.core.domain.common import (
    H,
    build_unpermutation,
    move_three_atoms,
    normalize_SPOSCAR,
    write_ifcs4,
)


def calculate_phonon_force_constants_4th(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    calculation,
    is_write: bool = False,
    poscar_path: str = "POSCAR",
):
    """
    Core function to calculate 4-phonon force constants.

    Args:
        na, nb, nc: Supercell size
        cutoff: Cutoff value
        calculation: Calculator object for force calculations
        is_write: Whether to save intermediate files

    Returns:
        None (writes FORCE_CONSTANTS_4TH file)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation4(
        na, nb, nc, cutoff, poscar_path
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    typer.echo(f"Found {wedge.nlist} quartet equivalence classes")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred

    typer.echo(f"Total DFT runs needed: {nruns}")

    # Write sposcar positions and forces to 4TH.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("4TH.SPOSCAR.xyz", format="extxyz")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = f"4TH.POSCAR.{{:0{width}d}}.xyz"
    p = build_unpermutation(sposcar)
    typer.echo("Computing an irreducible set of anharmonic force constants")
    phipart = np.zeros((3, nirred, ntot))
    for i, e in enumerate(track(list6, description="Processing quartets")):
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
            atoms = get_atoms(dsposcar, calculation)
            f = atoms.get_forces()[p, :]
            # Accumulate directly into phipart (equivalent to the original two-pass computation)
            phipart[:, i, :] -= isign * jsign * ksign * f.T
            filename = namepattern.format(number)
            if is_write:
                atoms.write(filename, format="extxyz")
    phipart /= 8000.0 * H * H * H
    typer.echo("Reconstructing the full array")
    phifull = fourthorder_core.reconstruct_ifcs(phipart, wedge, list6, poscar, sposcar)
    typer.echo("Writing the constants to FORCE_CONSTANTS_4TH")
    write_ifcs4(
        phifull, poscar, sposcar, dmin, nequi, shifts, frange, "FORCE_CONSTANTS_4TH"
    )


# Create the main app
app = typer.Typer(
    help="Calculate 4-phonon force constants using machine learning potentials."
)


@app.command()
def nep(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="NEP potential file path (e.g. 'nep.txt')",
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        "-w",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", "-g", help="Use GPU calculator for faster computation"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using NEP (Neural Evolution Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: NEP potential file path
        is_write: Whether to save intermediate files
        is_gpu: Use GPU calculator for faster computation
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    typer.echo(f"Initializing NEP calculator with potential: {potential}")
    try:
        calc = make_nep(potential, is_gpu=is_gpu)
        typer.echo(
            "Using GPU calculator for NEP" if is_gpu else "Using CPU calculator for NEP"
        )
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)


@app.command()
def tace(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    model_path: str = typer.Option(
        ..., exists=True, help="Path to the TACE model checkpoint (.pt/.pth/.ckpt)"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Compute device, e.g., 'cpu' or 'cuda'"
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        "-t",
        help="Tensor dtype: 'float32' | 'float64' | None (string 'None' to disable)",
    ),
    level: int = typer.Option(0, "--level", "-l", help="Fidelity level for TACE model"),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using TACE model.
    """
    # Normalize dtype option
    dtype_opt = None if dtype.lower() == "none" else dtype

    typer.echo(f"Initializing TACE calculator with model: {model_path}")
    try:
        calc = make_tace(
            model_path=model_path,
            device=device,
            dtype=dtype_opt,
            level=level,
        )
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)


@app.command()
def dp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="DeepMD potential file path (e.g. 'model.pb')",
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        "-w",
        help="Whether to save intermediate files during the calculation process",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using Deep Potential (DP) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: Deep Potential model file path
        is_write: Whether to save intermediate files
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # DP calculator initialization
    typer.echo(f"Initializing DP calculator with potential: {potential}")
    try:
        calc = make_dp(potential)
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)


@app.command()
def hiphive(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="Hiphive potential file path (e.g. 'potential.fcp')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        "-w",
        help="Whether to save intermediate files during the calculation process",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using hiphive force constant potential.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
                    Note: The supercell size must be greater than or equal to the size used
                    for training the fcp potential. It cannot be smaller.
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: Hiphive potential file path
        is_write: Whether to save intermediate files
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # Hiphive calculator initialization
    typer.echo(f"Using hiphive calculator with potential: {potential}")
    try:
        from hiphive import ForceConstantPotential
        from hiphive.calculators import ForceConstantCalculator

        fcp = ForceConstantPotential.read(potential)
        # Create a dummy atoms object to get force constants
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation4(na, nb, nc, cutoff, poscar)
        )
        atoms = get_atoms(normalize_SPOSCAR(sposcar))
        force_constants = fcp.get_force_constants(atoms)
        calc = ForceConstantCalculator(force_constants)
    except ImportError:
        typer.echo("hiphive not found, please install it first")
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)


@app.command()
def ploymp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., "--potential", "-p", exists=True, help="PolyMLP potential file path"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        "-w",
        help="Whether to save intermediate files during the calculation process",
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using PolyMLP (Polynomial Machine Learning Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)
        potential: PolyMLP potential file path
        is_write: Whether to save intermediate files
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # PolyMLP calculator initialization
    typer.echo(f"Using ploymp calculator with potential: {potential}")
    try:
        calc = make_polymp(potential)
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)


@app.command()
def mtp2(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ...,
        "--potential",
        "-p",
        exists=True,
        help="MTP potential file path (e.g. 'pot.mtp')",
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        "-w",
        help="Whether to save intermediate files during the calculation process",
    ),
    mtp_exe: str = typer.Option(
        "mlp", "--mtp-exe", "-x", help="Path to MLP executable, default is 'mlp'"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'",
        exists=True,
    ),
):
    """
    Calculate 4-phonon force constants using MTP (Moment Tensor Potential) model.

    Args:
        na, nb, nc: Supercell size, corresponding to expansion times in a, b, c directions\n
        cutoff: Cutoff distance, negative values for nearest neighbors, positive values for distance (in nm)\n
        potential: MTP potential file path\n
        is_write: Whether to save intermediate files\n
        mtp_exe: Path to MLP executable\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Read atoms to get unique elements
    atoms = read_atoms(poscar, in_format="auto")
    unique_elements = list(dict.fromkeys(atoms.get_chemical_symbols()))

    # MTP calculator initialization
    typer.echo(f"Initializing MTP calculator with potential: {potential}")
    try:
        calc = make_mtp(potential, mtp_exe=mtp_exe, unique_elements=unique_elements)
        typer.echo(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(na, nb, nc, cutoff, calc, is_write, poscar)
