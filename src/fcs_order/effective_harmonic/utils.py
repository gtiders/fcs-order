"""
Utility functions for effective harmonic calculations.
"""

import sys
from typing import Tuple, Any


def check_hiphive_imports() -> Tuple[Any, ...]:
    """
    Check if hiphive and trainstation are available, raise error if not.
    
    Returns
    -------
    tuple
        Tuple containing (ForceConstantPotential, ClusterSpace, StructureContainer, 
                         ForceConstantCalculator, Optimizer, prepare_structures)
    """
    try:
        from hiphive import ForceConstantPotential, ClusterSpace, StructureContainer
        from hiphive.calculators import ForceConstantCalculator
        from hiphive.utilities import prepare_structures
        from trainstation import Optimizer

        return (
            ForceConstantPotential,
            ClusterSpace,
            StructureContainer,
            ForceConstantCalculator,
            Optimizer,
            prepare_structures,
        )
    except ImportError as e:
        print("hiphive and trainstation are required for hiphive calculator")
        print(f"Import error: {e}")
        sys.exit(1)


def parse_temperature_string(temperature_string: str) -> list[int]:
    """
    Parse temperature string into list of integers.
    
    Parameters
    ----------
    temperature_string : str
        Comma-separated temperature values (e.g., '2000,1000,300')
        
    Returns
    -------
    list[int]
        List of temperature values
    """
    return [int(temp.strip()) for temp in temperature_string.split(",")]


def validate_calc_potential_pair(calc: str, potential: str) -> None:
    """
    Validate that calculator and potential are provided together.
    
    Parameters
    ----------
    calc : str
        Calculator type
    potential : str
        Potential file path
        
    Raises
    ------
    click.BadParameter
        If only one of calc or potential is provided
    """
    import click
    if (calc is not None and potential is None) or (calc is None and potential is not None):
        raise click.BadParameter("--calc and --potential must be provided together")