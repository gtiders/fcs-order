"""
核心计算层

包含对称性分析、Wedge约简、IFC重构等核心算法
支持二阶、三阶、四阶力常数计算
"""

from anharmonic.core.symmetry import SymmetryAnalyzer, SymmetryOperations
from anharmonic.core.wedge import TripletWedge, Wedge
from anharmonic.core.wedge4 import QuartetWedge, Wedge4
from anharmonic.core.reconstruction import IFCReconstructor, reconstruct_ifcs
from anharmonic.core.reconstruction4 import IFC4Reconstructor, reconstruct_ifcs4
from anharmonic.core.gaussian import gaussian_elimination, gaussian
from anharmonic.core.utils import (
    calculate_distances,
    calculate_cutoff_range,
    displace_two_atoms,
    generate_supercell,
    parse_cutoff,
    validate_supercell_size,
    displace_three_atoms,
    # 向后兼容别名
    calc_dists,
    calc_frange,
    move_two_atoms,
    move_three_atoms,
    gen_SPOSCAR,
)

__all__ = [
    # 现代化 API - 三阶
    "SymmetryAnalyzer",
    "TripletWedge",
    "IFCReconstructor",
    "gaussian_elimination",
    "calculate_distances",
    "calculate_cutoff_range",
    "displace_two_atoms",
    "displace_three_atoms",
    "generate_supercell",
    "parse_cutoff",
    "validate_supercell_size",
    # 现代化 API - 四阶
    "QuartetWedge",
    "IFC4Reconstructor",
    # 向后兼容别名
    "SymmetryOperations",
    "Wedge",
    "Wedge4",
    "reconstruct_ifcs",
    "reconstruct_ifcs4",
    "gaussian",
    "calc_dists",
    "calc_frange",
    "move_two_atoms",
    "move_three_atoms",
    "gen_SPOSCAR",
]
