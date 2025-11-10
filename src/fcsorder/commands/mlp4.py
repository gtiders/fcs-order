#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys

# Third-party imports
import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator
from rich.progress import track

# Local imports
from ..core import fourthorder_core  # type: ignore
from ..utils.order_common import (
    H,
    build_unpermutation,
    normalize_SPOSCAR,
    move_three_atoms,
    write_ifcs4,
)
from ..utils.prepare_calculation import prepare_calculation4
from ..utils.atoms import get_atoms
from ..utils.calculators import make_nep, make_dp, make_polymp, make_mtp, make_tace


def calculate_phonon_force_constants_4th(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    calculation,
    is_write: bool = False,
    is_sparse: bool = False,
    poscar_path: str = "POSCAR",
):
    """
    Core function to calculate 4-phonon force constants.

    Args:
        na, nb, nc: Supercell size
        cutoff: Cutoff value
        calculation: Calculator object for force calculations
        is_write: Whether to save intermediate files
        is_sparse: Use sparse tensor method

    Returns:
        None (writes FORCE_CONSTANTS_4TH file)
    """
    poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = prepare_calculation4(
        na, nb, nc, cutoff, poscar_path
    )
    natoms = len(poscar["types"])
    ntot = natoms * na * nb * nc
    wedge = fourthorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts, frange)
    typer.print(f"Found {wedge.nlist} quartet equivalence classes")
    list6 = wedge.build_list4()
    nirred = len(list6)
    nruns = 8 * nirred

    typer.print(f"Total DFT runs needed: {nruns}")

    # Write sposcar positions and forces to 4TH.SPOSCAR.extxyz file
    atoms = get_atoms(normalize_SPOSCAR(sposcar), calculation)
    atoms.get_forces()
    atoms.write("4TH.SPOSCAR.xyz", format="extxyz")
    width = len(str(8 * (len(list6) + 1)))
    namepattern = f"4TH.POSCAR.{{:0{width}d}}.xyz"
    p = build_unpermutation(sposcar)
    typer.print("Computing an irreducible set of anharmonic force constants")
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
    typer.print("Reconstructing the full array")
    phifull = fourthorder_core.reconstruct_ifcs(
        phipart, wedge, list6, poscar, sposcar, is_sparse
    )
    typer.print("Writing the constants to FORCE_CONSTANTS_4TH")
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
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="NEP potential file path (e.g. 'nep.txt')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
    is_gpu: bool = typer.Option(
        False, "--is-gpu", help="Use GPU calculator for faster computation"
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
        is_sparse: Use sparse tensor method for memory efficiency
        is_gpu: Use GPU calculator for faster computation
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    typer.print(f"Initializing NEP calculator with potential: {potential}")
    try:
        calc = make_nep(potential, is_gpu=is_gpu)
        typer.print(
            "Using GPU calculator for NEP" if is_gpu else "Using CPU calculator for NEP"
        )
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )


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
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
    device: str = typer.Option("cuda", help="Compute device, e.g., 'cpu' or 'cuda'"),
    dtype: str = typer.Option(
        "float32",
        help="Tensor dtype: 'float32' | 'float64' | None (string 'None' to disable)",
    ),
    level: int = typer.Option(0, help="Fidelity level for TACE model"),
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

    typer.print(f"Initializing TACE calculator with model: {model_path}")
    try:
        calc = make_tace(
            model_path=model_path,
            device=device,
            dtype=dtype_opt,
            level=level,
        )
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )


@app.command()
def dp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="DeepMD potential file path (e.g. 'model.pb')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
    poscar: str = typer.Option(
        "POSCAR",
        "--poscar",
        help="Path to a structure file parsable by ASE. Default: 'POSCAR'",
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
        is_sparse: Use sparse tensor method for memory efficiency
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # DP calculator initialization
    typer.print(f"Initializing DP calculator with potential: {potential}")
    try:
        calc = make_dp(potential)
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )


@app.command()
def hiphive(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="Hiphive potential file path (e.g. 'potential.fcp')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
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
        is_sparse: Use sparse tensor method for memory efficiency
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # Hiphive calculator initialization
    typer.print(f"Using hiphive calculator with potential: {potential}")
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
        typer.print("hiphive not found, please install it first")
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )


@app.command()
def ploymp(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(..., exists=True, help="PolyMLP potential file path"),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
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
        is_sparse: Use sparse tensor method for memory efficiency
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'
    """
    # PolyMLP calculator initialization
    typer.print(f"Using ploymp calculator with potential: {potential}")
    try:
        calc = make_polymp(potential)
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )


@app.command()
def mtp2(
    na: int,
    nb: int,
    nc: int,
    cutoff: str = typer.Option(
        ...,
        help="Cutoff value (negative for nearest neighbors, positive for distance in nm)",
    ),
    potential: str = typer.Option(
        ..., exists=True, help="MTP potential file path (e.g. 'pot.mtp')"
    ),
    is_write: bool = typer.Option(
        False,
        "--is-write",
        help="Whether to save intermediate files during the calculation process",
    ),
    is_sparse: bool = typer.Option(
        False, "--is-sparse", help="Use sparse tensor method for memory efficiency"
    ),
    mtp_exe: str = typer.Option(
        "mlp", "--mtp-exe", help="Path to MLP executable, default is 'mlp'"
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
        is_sparse: Use sparse tensor method for memory efficiency\n
        mtp_exe: Path to MLP executable\n
        poscar: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'\n
    """
    # Read atoms to get unique elements
    from ase.io import read

    atoms = read(poscar)
    unique_elements = list(dict.fromkeys(atoms.get_chemical_symbols()))

    # MTP calculator initialization
    typer.print(f"Initializing MTP calculator with potential: {potential}")
    try:
        calc = make_mtp(potential, mtp_exe=mtp_exe, unique_elements=unique_elements)
        typer.print(f"Using MTP calculator with elements: {unique_elements}")
    except ImportError as e:
        typer.print(str(e))
        raise typer.Exit(code=1)

    calculate_phonon_force_constants_4th(
        na, nb, nc, cutoff, calc, is_write, is_sparse, poscar
    )
