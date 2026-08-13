from __future__ import annotations

from pathlib import Path

from ase import Atoms

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def conventional_cell() -> Atoms:
    """Return the eight-atom conventional KCl cell used upstream."""
    return Atoms(
        symbols=["K"] * 4 + ["Cl"] * 4,
        scaled_positions=[
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ],
        cell=[6.292, 6.292, 6.292],
        pbc=True,
    )
