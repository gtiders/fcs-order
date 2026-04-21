"""Shared CLI helpers for mlfcs command-line entrypoints."""

from __future__ import annotations

import argparse


def build_sow_reap_parser(
    description: str,
    symprec_default: float,
    hstep_default: float,
    forces_help: str,
) -> argparse.ArgumentParser:
    """Create a parser for the shared sow/reap CLI shape."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "command",
        choices=["sow", "reap", "interfaces"],
        help="Sub-command: sow, reap, or interfaces",
    )
    parser.add_argument("na", type=int, nargs="?", help="Supercell A")
    parser.add_argument("nb", type=int, nargs="?", help="Supercell B")
    parser.add_argument("nc", type=int, nargs="?", help="Supercell C")
    parser.add_argument(
        "--cutoff",
        required=False,
        type=str,
        help="Cutoff in nm (positive) or neighbor index (negative integer), e.g. --cutoff -3",
    )
    parser.add_argument(
        "-i", "--input", default="POSCAR", help="Input structure file (default: POSCAR)"
    )
    parser.add_argument(
        "--interface",
        default="vasp",
        help=(
            "Input interface for structure parsing (default: vasp). "
            "Use explicit values such as vasp, abacus, qe, cp2k, aims."
        ),
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=symprec_default,
        help=f"Symmetry precision (default: {symprec_default})",
    )
    parser.add_argument(
        "--hstep",
        type=float,
        default=hstep_default,
        help=f"Displacement step size in nm (default: {hstep_default})",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="vasp",
        choices=["vasp", "same"],
        help=(
            "[Sow] Output format: vasp (multiple files), "
            "or same (write using --interface)."
        ),
    )
    parser.add_argument("--forces", nargs="+", help=forces_help)
    parser.add_argument(
        "--forces-interface",
        default=None,
        help=(
            "Interface used to parse force files in reap. "
            "Default: use --interface value."
        ),
    )
    return parser


def print_cli_interface_capabilities(tool_name: str, interfaces: list[str]) -> None:
    """Print CLI interface capabilities for the discovered phonopy interfaces."""
    print(f"{tool_name} CLI interface capabilities")
    print("")
    print("Columns:")
    print("  read   = structure read via --interface")
    print("  write  = sow --format same via phonopy writer")
    print("  forces = reap force parsing via --forces-interface")
    print("")
    print(f"{'interface':<16} {'read':<6} {'write':<6} {'forces':<7} notes")
    print(f"{'-' * 16} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 24}")

    for name in interfaces:
        note = ""
        if name == "cp2k":
            note = "writer may need template info"
        print(f"{name:<16} {'yes':<6} {'yes':<6} {'yes':<7} {note}")

    print("")
    print("Extra CLI formats:")
    print("  sow --format vasp : always available")
    print("")
    print("Use '--interface <name>' for structure, '--forces-interface <name>' for reap.")
