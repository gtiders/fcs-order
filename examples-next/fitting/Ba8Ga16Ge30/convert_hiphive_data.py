#!/usr/bin/env python3
"""Convert the official hiPhive Ba8Ga16Ge30 DFT snapshots to MLFCS input."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ase.db import connect
from ase.io import read, write

from mlfcs import StructureRelation

DATABASES = (
    "mc_rattle_std0.042_vdW-DF-cx.db",
    "mc_rattle_based_md_T300_vdW-DF-cx.db",
    "mc_rattle_based_md_T650_vdW-DF-cx.db",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="hiPhive BaGaGe_clathrate/ directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "input")
    args = parser.parse_args()

    structures = args.source / "dft_calculations" / "structures"
    revision = subprocess.run(
        ["git", "-C", str(args.source.parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reference = read(structures / "POSCAR_groundstate_vdW-DF-cx")
    snapshots = []
    counts = {}
    for name in DATABASES:
        frames = [row.toatoms() for row in connect(structures / name).select()]
        counts[name] = len(frames)
        snapshots.extend(frames)

    # This clathrate's 54-atom cubic cell is its primitive calculation frame.
    # The explicit relation also rejects accidental cell or atom-order changes.
    relation = StructureRelation.from_atoms(reference, reference)
    for snapshot in snapshots:
        relation.displacement(snapshot)
        snapshot.new_array("forces", snapshot.get_forces())
        snapshot.calc = None

    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "primitive.vasp", reference, format="vasp", direct=True, sort=False)
    write(args.output / "reference.vasp", reference, format="vasp", direct=True, sort=False)
    write(args.output / "training.extxyz", snapshots, format="extxyz")
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "upstream": "https://gitlab.com/materials-modeling/hiphive-examples",
                "material": "Ba8Ga16Ge30",
                "upstream_revision": revision,
                "reference": "dft_calculations/structures/POSCAR_groundstate_vdW-DF-cx",
                "databases": counts,
                "frames": len(snapshots),
                "source_atom_order_preserved": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(f"wrote {len(snapshots)} snapshots to {args.output}")


if __name__ == "__main__":
    main()
