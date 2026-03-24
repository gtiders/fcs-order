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
        "command", choices=["sow", "reap"], help="Sub-command: sow or reap"
    )
    parser.add_argument("na", type=int, help="Supercell A")
    parser.add_argument("nb", type=int, help="Supercell B")
    parser.add_argument("nc", type=int, help="Supercell C")
    parser.add_argument(
        "--cutoff",
        required=True,
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
        choices=["vasp", "xyz", "same"],
        help=(
            "[Sow] Output format: vasp (multiple files), xyz (single file), "
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
