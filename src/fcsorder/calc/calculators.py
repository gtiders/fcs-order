#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculator factory for creating various ML/physics-based calculators."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import typer
from ase.calculators.calculator import Calculator


class CalculatorConfig:
    """Configuration metadata for a calculator type.

    Stores parameter specifications for a registered calculator.
    """

    def __init__(
        self,
        name: str,
        required_params: dict[str, str],
        optional_params: dict[str, tuple[Any, str]],
    ) -> None:
        """Initialize calculator configuration.

        Args:
            name: Calculator type name.
            required_params: Dict of {param_name: description}.
            optional_params: Dict of {param_name: (default_value, description)}.
        """
        self._name = name
        self._required_params = required_params
        self._optional_params = optional_params

    @property
    def name(self) -> str:
        """Get calculator type name."""
        return self._name

    @property
    def required_params(self) -> dict[str, str]:
        """Get required parameters."""
        return self._required_params

    @property
    def optional_params(self) -> dict[str, tuple[Any, str]]:
        """Get optional parameters."""
        return self._optional_params


class CalculatorFactory:
    """Factory for creating ASE calculators from various backends."""

    _registry: dict[str, Callable[..., Calculator]] = {}
    _configs: dict[str, CalculatorConfig] = {}

    @classmethod
    def register(
        cls, name: str, config: Optional[CalculatorConfig] = None
    ) -> Callable:
        """Decorator to register a calculator builder function.

        Args:
            name: Calculator type name.
            config: Optional configuration metadata.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable[..., Calculator]) -> Callable[..., Calculator]:
            cls._registry[name.lower()] = func
            if config:
                cls._configs[name.lower()] = config
            return func
        return decorator

    @classmethod
    def create(cls, calculator_type: str, **kwargs) -> Calculator:
        """Create a calculator by type name.

        Args:
            calculator_type: Type of calculator (nep, dp, polymlp, mtp, hiphive, tace, mace).
            **kwargs: Arguments passed to the specific calculator builder.

        Returns:
            ASE Calculator instance.

        Raises:
            ValueError: If calculator type is not registered.
        """
        calc_type_lower = calculator_type.lower()
        if calc_type_lower not in cls._registry:
            available_types = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown calculator type '{calculator_type}'. "
                f"Available types: {available_types}"
            )
        return cls._registry[calc_type_lower](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered calculator types.

        Returns:
            Sorted list of calculator type names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_config(cls, calculator_type: str) -> Optional[CalculatorConfig]:
        """Get configuration metadata for a calculator type.

        Args:
            calculator_type: Calculator type name.

        Returns:
            CalculatorConfig instance or None if not found.
        """
        return cls._configs.get(calculator_type.lower())


@CalculatorFactory.register(
    "nep",
    CalculatorConfig(
        name="nep",
        required_params={"potential": "Path to NEP model file"},
        optional_params={"device": ("cpu", "Compute device (cpu or cuda)")},
    ),
)
def _make_nep(potential: str, device: str = "cpu", **kwargs) -> Calculator:
    """Create a NEP calculator via calorine (CPU/GPU).

    Args:
        potential: Path to NEP model file.
        device: Compute device ("cpu" or "cuda").
        **kwargs: Additional arguments.

    Returns:
        NEP calculator instance.

    Raises:
        ImportError: If calorine is not installed.
    """
    try:
        from calorine.calculators import CPUNEP, GPUNEP  # type: ignore
    except ImportError as e:
        raise ImportError("calorine not found, please install it first") from e
    return GPUNEP(potential) if device.lower() == "cuda" else CPUNEP(potential)


@CalculatorFactory.register(
    "dp",
    CalculatorConfig(
        name="dp",
        required_params={"potential": "Path to DeepMD model file"},
        optional_params={},
    ),
)
def _make_dp(potential: str, **kwargs) -> Calculator:
    """Create a DeepMD calculator."""
    try:
        from deepmd.calculator import DP  # type: ignore
    except ImportError as e:
        raise ImportError("deepmd not found, please install it first") from e
    return DP(model=potential)


@CalculatorFactory.register(
    "polymlp",
    CalculatorConfig(
        name="polymlp",
        required_params={"potential": "Path to PolyMLP model file"},
        optional_params={},
    ),
)
def _make_polymlp(potential: str, **kwargs) -> Calculator:
    """Create a PolyMLP ASE calculator."""
    try:
        from pypolymlp.calculator.utils.ase_calculator import (  # type: ignore
            PolymlpASECalculator,
        )
    except ImportError as e:
        raise ImportError("pypolymlp not found, please install it first") from e
    return PolymlpASECalculator(pot=potential)


@CalculatorFactory.register(
    "mtp",
    CalculatorConfig(
        name="mtp",
        required_params={"potential": "Path to MTP model file"},
        optional_params={
            "mtp_exe": ("mlp", "MLP executable name"),
            "unique_elements": (None, "List of unique elements"),
        },
    ),
)
def _make_mtp(
    potential: str,
    mtp_exe: str = "mlp",
    unique_elements: Optional[Sequence[str]] = None,
    **kwargs,
) -> Calculator:
    """
    Create an internal MTP calculator wrapper.
    
    Warning:
        This MTP calculator only supports mlp2.x versions.
    """
    try:
        from fcsorder.calc.mtpcalc import MTP
    except Exception as e:
        raise ImportError(f"Error importing MTP: {e}") from e
    
    # Print warning message in red
    typer.secho(
        "⚠️  Warning: This MTP calculator only supports mlp2.x versions",
        fg=typer.colors.RED,
        bold=True,
    )
    
    return MTP(
        mtp_path=potential,
        mtp_exe=mtp_exe,
        unique_elements=list(unique_elements or []),
    )


@CalculatorFactory.register(
    "hiphive",
    CalculatorConfig(
        name="hiphive",
        required_params={
            "potential": "Path to hiphive ForceConstantPotential file",
            "supercell": "ASE Atoms object for supercell",
        },
        optional_params={},
    ),
)
def _make_hiphive(potential: str, supercell: Any = None, **kwargs) -> Calculator:
    """Create a hiphive ForceConstantCalculator from a supercell."""
    if supercell is None:
        raise ValueError("supercell is required for hiphive calculator")
    try:
        from hiphive import ForceConstantPotential  # type: ignore
        from hiphive.calculators import ForceConstantCalculator  # type: ignore
    except ImportError as e:
        raise ImportError("hiphive not found, please install it first") from e
    fcp = ForceConstantPotential.read(potential)
    force_constants = fcp.get_force_constants(supercell)
    return ForceConstantCalculator(force_constants)


@CalculatorFactory.register(
    "tace",
    CalculatorConfig(
        name="tace",
        required_params={"model_path": "Path to TACE model checkpoint"},
        optional_params={
            "device": ("cpu", "Compute device (cpu/cuda)"),
            "dtype": ("float64", "Tensor dtype (float32/float64)"),
            "level": (0, "Fidelity level"),
        },
    ),
)
def _make_tace(
    model_path: str,
    device: str = "cpu",
    dtype: Optional[str] = "float64",
    extra_compute_first_derivative: Optional[bool] = None,
    extra_compute_second_derivative: Optional[bool] = None,
    level: int = 0,
    **kwargs,
) -> Calculator:
    """Create a TACECalculator."""
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


@CalculatorFactory.register(
    "mace",
    CalculatorConfig(
        name="mace",
        required_params={"model_path": "Path to MACE model file"},
        optional_params={
            "device": ("cpu", "Compute device (cpu/cuda)"),
            "default_dtype": ("float64", "Default tensor dtype"),
        },
    ),
)
def _make_mace(
    model_path: str,
    device: str = "cpu",
    default_dtype: str = "float64",
    **kwargs,
) -> Calculator:
    """Create a MACE calculator."""
    try:
        from mace.calculators import MACECalculator  # type: ignore
    except ImportError as e:
        raise ImportError("mace not found, please install it first") from e
    return MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=default_dtype,
    )
