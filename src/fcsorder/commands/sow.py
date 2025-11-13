#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import typer

from fcsorder.core import thirdorder_core, fourthorder_core
from fcsorder.core.thirdorder_core import prepare_calculation3
from fcsorder.core.fourthorder_core import prepare_calculation4
from fcsorder.io.io_abstraction import write_structure
from fcsorder.core.domain.common import (
    H,
    move_three_atoms,
    move_two_atoms,
    normalize_SPOSCAR,
)


def sow(
    na: int,
    nb: int,
    nc: int,
    cutoff: str,
    order: int = 3,
    poscar_path: str = "POSCAR",
    out_format: str = "poscar",
    out_dir: str = ".",
    name_template: str | None = None,
    undisplaced_name: str | None = None,
):
    """
    Generate displaced POSCAR files for 3-phonon or 4-phonon calculations.

    Args:
        na, nb, nc: Supercell dimensions along a, b, c directions.
        cutoff: Cutoff distance (negative for nearest neighbors, positive for distance in nm).
        order: 3 for third-order (3-phonon), 4 for fourth-order (4-phonon).
        poscar_path: Path to a structure file parsable by ASE (e.g., VASP POSCAR, CIF, XYZ). Default: 'POSCAR'.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    ofmt = out_format.lower()
    is_vasp = ofmt in ("poscar", "vasp")
    ext = "poscar" if is_vasp else ofmt

    if order == 3:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation3(na, nb, nc, cutoff, poscar_path)
        )
        wedge = thirdorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.echo(f"Found {wedge.nlist} triplet equivalence classes")
        list4 = wedge.build_list4()
        nirred = len(list4)
        nruns = 4 * nirred
        typer.echo(f"Total DFT runs needed: {nruns}")

        if undisplaced_name:
            ctx = {
                "order": "3RD",
                "phase": "structure",
                "ext": ext,
                "width": len(str(4 * (len(list4) + 1))),
                "index": 0,
                "index_padded": f"{0:0{len(str(4 * (len(list4) + 1)))}d}",
            }
            filename = undisplaced_name.format(**ctx)
            filepath = os.path.join(out_dir, filename)
            typer.echo(
                f"Writing undisplaced coordinates to {os.path.basename(filepath)}"
            )
            write_structure(normalize_SPOSCAR(sposcar), filepath, ofmt)
        else:
            if is_vasp:
                typer.echo("Writing undisplaced coordinates to 3RD.SPOSCAR")
                write_structure(
                    normalize_SPOSCAR(sposcar),
                    os.path.join(out_dir, "3RD.SPOSCAR"),
                    ofmt,
                )
            else:
                undisplaced = os.path.join(out_dir, f"3RD.structure.{ext}")
                typer.echo(
                    f"Writing undisplaced coordinates to {os.path.basename(undisplaced)}"
                )
                write_structure(normalize_SPOSCAR(sposcar), undisplaced, ofmt)
        width = len(str(4 * (len(list4) + 1)))
        if name_template:
            typer.echo("Writing displaced coordinates using custom template")
            # name_template can use placeholders
            # {order}, {phase}, {index}, {index_padded}, {width}, {ext}
            namepattern = None
        else:
            if is_vasp:
                namepattern = f"3RD.POSCAR.{{:0{width}d}}"
                typer.echo("Writing displaced coordinates to 3RD.POSCAR.* files")
            else:
                namepattern = f"3RD.disp.{{:0{width}d}}.{ext}"
                typer.echo(f"Writing displaced coordinates to 3RD.disp.*.{ext} files")
        for i, e in enumerate(list4):
            for n in range(4):
                isign = (-1) ** (n // 2)
                jsign = -((-1) ** (n % 2))
                number = nirred * n + i + 1
                dsposcar = normalize_SPOSCAR(
                    move_two_atoms(
                        sposcar, e[1], e[3], isign * H, e[0], e[2], jsign * H
                    )
                )
                if name_template:
                    ctx = {
                        "order": "3RD",
                        "phase": "disp",
                        "index": number,
                        "width": width,
                        "index_padded": f"{number:0{width}d}",
                        "ext": ext,
                    }
                    filename = name_template.format(**ctx)
                    filepath = os.path.join(out_dir, filename)
                else:
                    filename = namepattern.format(number)
                    filepath = os.path.join(out_dir, filename)
                write_structure(dsposcar, filepath, ofmt)
        return

    if order == 4:
        poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh = (
            prepare_calculation4(na, nb, nc, cutoff, poscar_path)
        )
        wedge = fourthorder_core.Wedge(
            poscar, sposcar, symops, dmin, nequi, shifts, frange
        )
        typer.echo(f"Found {wedge.nlist} quartet equivalence classes")
        list6 = wedge.build_list4()
        nirred = len(list6)
        nruns = 8 * nirred
        typer.echo(f"Total DFT runs needed: {nruns}")
        if undisplaced_name:
            ctx = {
                "order": "4TH",
                "phase": "structure",
                "ext": ext,
                "width": len(str(8 * (len(list6) + 1))),
                "index": 0,
                "index_padded": f"{0:0{len(str(8 * (len(list6) + 1)))}d}",
            }
            filename = undisplaced_name.format(**ctx)
            filepath = os.path.join(out_dir, filename)
            typer.echo(
                f"Writing undisplaced coordinates to {os.path.basename(filepath)}"
            )
            write_structure(normalize_SPOSCAR(sposcar), filepath, ofmt)
        else:
            if is_vasp:
                typer.echo("Writing undisplaced coordinates to 4TH.SPOSCAR")
                write_structure(
                    normalize_SPOSCAR(sposcar),
                    os.path.join(out_dir, "4TH.SPOSCAR"),
                    ofmt,
                )
            else:
                undisplaced = os.path.join(out_dir, f"4TH.structure.{ext}")
                typer.echo(
                    f"Writing undisplaced coordinates to {os.path.basename(undisplaced)}"
                )
                write_structure(normalize_SPOSCAR(sposcar), undisplaced, ofmt)
        width = len(str(8 * (len(list6) + 1)))
        if name_template:
            typer.echo("Writing displaced coordinates using custom template")
            namepattern = None
        else:
            if is_vasp:
                namepattern = f"4TH.POSCAR.{{:0{width}d}}"
                typer.echo("Writing displaced coordinates to 4TH.POSCAR.* files")
            else:
                namepattern = f"4TH.disp.{{:0{width}d}}.{ext}"
                typer.echo(f"Writing displaced coordinates to 4TH.disp.*.{ext} files")
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
                if name_template:
                    ctx = {
                        "order": "4TH",
                        "phase": "disp",
                        "index": number,
                        "width": width,
                        "index_padded": f"{number:0{width}d}",
                        "ext": ext,
                    }
                    filename = name_template.format(**ctx)
                    filepath = os.path.join(out_dir, filename)
                else:
                    filename = namepattern.format(number)
                    filepath = os.path.join(out_dir, filename)
                write_structure(dsposcar, filepath, ofmt)
        return

    raise typer.BadParameter("--order must be either 3 or 4")
