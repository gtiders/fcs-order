#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Sequence


def make_nep(potential: str, is_gpu: bool = False):
    """Create a NEP calculator via calorine (CPU/GPU)."""
    try:
        from calorine.calculators import CPUNEP, GPUNEP  # type: ignore
    except ImportError as e:
        raise ImportError("calorine not found, please install it first") from e
    return GPUNEP(potential) if is_gpu else CPUNEP(potential)


def make_dp(potential: str):
    """Create a DeepMD calculator."""
    try:
        from deepmd.calculator import DP  # type: ignore
    except ImportError as e:
        raise ImportError("deepmd not found, please install it first") from e
    return DP(model=potential)


def make_polymp(potential: str):
    """Create a PolyMLP ASE calculator."""
    try:
        from pypolymlp.calculator.utils.ase_calculator import (  # type: ignore
            PolymlpASECalculator,
        )
    except ImportError as e:
        raise ImportError("pypolymlp not found, please install it first") from e
    return PolymlpASECalculator(pot=potential)


def make_mtp(
    potential: str,
    mtp_exe: str = "mlp",
    unique_elements: Optional[Sequence[str]] = None,
):
    """Create an internal MTP calculator wrapper consistent with existing usage."""
    try:
        # Local import to avoid hard dependency for users who don't need MTP
        from fcsorder.calc.mtp2calc import MTP
    except Exception as e:
        raise ImportError(f"Error importing MTP: {e}") from e
    return MTP(
        mtp_path=potential, mtp_exe=mtp_exe, unique_elements=list(unique_elements or [])
    )


def make_hiphive_force_constant_calculator_from_supercell(
    potential: str, supercell
) -> "Calculator":
    """Create a hiphive ForceConstantCalculator using a provided ASE supercell."""
    try:
        from hiphive import ForceConstantPotential  # type: ignore
        from hiphive.calculators import ForceConstantCalculator  # type: ignore
    except ImportError as e:
        raise ImportError("hiphive not found, please install it first") from e
    fcp = ForceConstantPotential.read(potential)
    force_constants = fcp.get_force_constants(supercell)
    return ForceConstantCalculator(force_constants)


def make_tace(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[str] = "float32",
    extra_compute_first_derivative=None,
    extra_compute_second_derivative=None,
    level: int = 0,
):
    """Create a TACECalculator from tace.interface.ase.calculator."""
    try:
        from tace.interface.ase.calculator import TACECalculator  # type: ignore
    except ImportError as e:
        raise ImportError("tace not found, please install it first") from e
    return TACECalculator(
        model_path=model_path,
        device=device,
        dtype=dtype,
        extra_compute_first_derivative=extra_compute_first_derivative,
        extra_compute_second_derivative=extra_compute_second_derivative,
        level=level,
    )
