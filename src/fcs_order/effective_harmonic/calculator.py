"""
Calculator initialization and management for effective harmonic calculations.
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def initialize_calculator(
    calc_type: str, 
    potential_file: str, 
    atoms: "Atoms"
) -> "Calculator":
    """
    Initialize ASE calculator based on calculator type and potential file.

    Parameters
    ----------
    calc_type : str
        Calculator type ('nep', 'dp', 'hiphive', 'ploymp')
    potential_file : str
        Path to potential file
    atoms : ase.Atoms
        Atoms object for hiphive calculator

    Returns
    -------
    calculator : ase.calculators.calculator.Calculator
        Initialized ASE calculator

    Raises
    ------
    SystemExit
        If required package is not installed or unknown calculator type
    """
    from .utils import check_hiphive_imports
    
    calc_type = calc_type.lower()

    if calc_type == "nep":
        print(f"Using NEP calculator with potential: {potential_file}")
        try:
            from calorine.calculators import CPUNEP
            return CPUNEP(potential_file)
        except ImportError:
            print("calorine not found, please install it first")
            sys.exit(1)

    elif calc_type == "dp":
        print(f"Using DP calculator with potential: {potential_file}")
        try:
            from deepmd.calculator import DP
            return DP(model=potential_file)
        except ImportError:
            print("deepmd not found, please install it first")
            sys.exit(1)

    elif calc_type == "hiphive":
        print(f"Using hiphive calculator with potential: {potential_file}")
        (
            ForceConstantPotential,
            ClusterSpace,
            StructureContainer,
            ForceConstantCalculator,
            Optimizer,
            _,
        ) = check_hiphive_imports()
        try:
            fcp = ForceConstantPotential.read(potential_file)
            force_constants = fcp.get_force_constants(atoms)
            return ForceConstantCalculator(force_constants)
        except ImportError:
            print("hiphive not found, please install it first")
            sys.exit(1)

    elif calc_type == "ploymp":
        print(f"Using ploymp calculator with potential: {potential_file}")
        try:
            from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator
            return PolymlpASECalculator(pot=potential_file)
        except ImportError:
            print("pypolymlp not found, please install it first")
            sys.exit(1)
    else:
        print(f"Unknown calculator type: {calc_type}")
        sys.exit(1)