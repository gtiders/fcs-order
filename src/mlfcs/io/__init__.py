from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlfcs.model import ForceConstants

Writer = Callable[..., None]


def write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
) -> None:
    normalized = format.casefold().replace("-", "_")
    if normalized == "hdf5":
        from mlfcs.io.hdf5 import write_hdf5

        write_hdf5(target, force_constants)
        return
    if normalized in {"numpy", "npz"}:
        from mlfcs.io.numpy import write_numpy

        write_numpy(target, force_constants)
        return
    if normalized == "shengbte":
        from mlfcs.io.shengbte import write_shengbte

        selected_order = order if order is not None else max(force_constants.arrays)
        cutoff = force_constants.metadata.get("cutoff_angstrom")
        if cutoff is None:
            raise ValueError("cutoff_angstrom metadata is required for ShengBTE output")
        write_shengbte(
            target,
            force_constants.arrays[selected_order],
            force_constants.supercell,
            cutoff=float(cutoff),
        )
        return
    raise ValueError(f"unknown force-constant format: {format!r}")


__all__ = ["write_force_constants"]
