"""Complete external VASP sow/collect/reap example.

The script does not launch or configure VASP. It creates ordered POSCAR files,
collects forces from completed vasprun.xml files, and reconstructs FC3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

from mlfcs import ForceConstantCalculation

MANIFEST = "mlfcs-plan.json"
FORCE_ARCHIVE = "forces.npz"


def calculation_from_values(
    primitive_path: Path,
    *,
    supercell: tuple[int, int, int],
    cutoff: float,
    displacement: float,
    verbose: bool = True,
) -> ForceConstantCalculation:
    return ForceConstantCalculation(
        read(primitive_path),
        order=3,
        supercell=supercell,
        cutoff=cutoff,
        displacement=displacement,
        jax_platform="cpu",
        verbose=verbose,
    )


def sow(arguments: argparse.Namespace) -> None:
    workspace = arguments.workspace.resolve()
    structures = workspace / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    primitive_target = workspace / "POSCAR-unitcell"
    primitive = read(arguments.primitive)
    write(primitive_target, primitive, format="vasp", direct=True, vasp5=True)

    calculation = calculation_from_values(
        primitive_target,
        supercell=tuple(arguments.supercell),
        cutoff=arguments.cutoff,
        displacement=arguments.displacement,
    )
    configurations = calculation.sow(atom_order="grouped")
    width = max(3, len(str(len(configurations))))
    filenames = []
    for configuration_id, atoms in enumerate(configurations):
        filename = f"POSCAR-{configuration_id + 1:0{width}d}"
        write(structures / filename, atoms, format="vasp", direct=True, vasp5=True)
        filenames.append(filename)

    manifest = {
        "schema": 1,
        "order": 3,
        "supercell": list(arguments.supercell),
        "cutoff": arguments.cutoff,
        "displacement": arguments.displacement,
        "atom_order": "grouped",
        "plan_hash": calculation.plan.hash,
        "configuration_count": len(configurations),
        "filenames": filenames,
    }
    (workspace / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(configurations)} ordered structures to {structures}")
    print(f"Plan manifest: {workspace / MANIFEST}")


def load_manifest(workspace: Path) -> dict[str, object]:
    manifest = json.loads((workspace / MANIFEST).read_text())
    if manifest.get("schema") != 1 or manifest.get("order") != 3:
        raise ValueError("unsupported or invalid MLFCS plan manifest")
    return manifest


def collect(arguments: argparse.Namespace) -> None:
    workspace = arguments.workspace.resolve()
    manifest = load_manifest(workspace)
    calculations = arguments.calculations.resolve()
    forces = []
    missing = []
    for filename in manifest["filenames"]:
        vasprun = calculations / filename / "vasprun.xml"
        if not vasprun.is_file():
            missing.append(str(vasprun))
            continue
        atoms = read(vasprun, index=-1)
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} vasprun.xml files:\n{preview}")

    values = np.asarray(forces, dtype=float)
    expected_count = int(manifest["configuration_count"])
    if values.shape[0] != expected_count:
        raise ValueError(f"expected {expected_count} force sets, got {values.shape[0]}")
    np.savez_compressed(
        workspace / FORCE_ARCHIVE,
        forces=values,
        configuration_ids=np.arange(expected_count, dtype=int),
        plan_hash=np.asarray(manifest["plan_hash"]),
        atom_order=np.asarray(manifest["atom_order"]),
    )
    print(f"Collected {expected_count} force sets in sow order")
    print(f"Force archive: {workspace / FORCE_ARCHIVE}")


def reap(arguments: argparse.Namespace) -> None:
    workspace = arguments.workspace.resolve()
    manifest = load_manifest(workspace)
    calculation = calculation_from_values(
        workspace / "POSCAR-unitcell",
        supercell=tuple(int(value) for value in manifest["supercell"]),
        cutoff=float(manifest["cutoff"]),
        displacement=float(manifest["displacement"]),
    )
    archive = np.load(workspace / FORCE_ARCHIVE)
    expected_ids = np.arange(int(manifest["configuration_count"]), dtype=int)
    if not np.array_equal(archive["configuration_ids"], expected_ids):
        raise ValueError("force archive configuration IDs are not the exact sow order")
    archive_hash = str(archive["plan_hash"])
    if archive_hash != manifest["plan_hash"]:
        raise ValueError("force archive and manifest plan hashes differ")

    result = calculation.reap(
        archive["forces"],
        atom_order=str(archive["atom_order"]),
        plan_hash=archive_hash,
        acoustic_sum_rule=not arguments.no_asr,
    )
    output = arguments.output.resolve()
    result.write(
        output,
        format=arguments.format,
        order=3,
        compatibility=arguments.compatibility,
    )
    print(f"Wrote order-3 force constants to {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    sow_parser = commands.add_parser("sow", help="write ordered POSCAR-xxx structures")
    sow_parser.add_argument("primitive", type=Path)
    sow_parser.add_argument("workspace", type=Path)
    sow_parser.add_argument("--supercell", type=int, nargs=3, default=(3, 3, 3))
    sow_parser.add_argument("--cutoff", type=float, default=-6)
    sow_parser.add_argument("--displacement", type=float, default=0.01)
    sow_parser.set_defaults(handler=sow)

    collect_parser = commands.add_parser("collect", help="collect ordered vasprun.xml forces")
    collect_parser.add_argument("workspace", type=Path)
    collect_parser.add_argument("calculations", type=Path)
    collect_parser.set_defaults(handler=collect)

    reap_parser = commands.add_parser("reap", help="reconstruct and export FC3")
    reap_parser.add_argument("workspace", type=Path)
    reap_parser.add_argument("output", type=Path)
    reap_parser.add_argument(
        "--format",
        choices=("hdf5", "numpy", "phono3py_hdf5", "shengbte"),
        default="shengbte",
    )
    reap_parser.add_argument("--compatibility", choices=("thirdorder",))
    reap_parser.add_argument("--no-asr", action="store_true")
    reap_parser.set_defaults(handler=reap)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
