"""
力常数数据模型

定义力常数相关的数据类
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import sparse
    from anharmonic.models.structure import CrystalStructure, SupercellStructure


@dataclass
class ThirdOrderIFC:
    """
    三阶力常数容器
    
    Attributes:
        data: 稀疏力常数矩阵 (3, 3, 3, natoms, ntot, ntot)
        primitive_structure: 原胞结构
        supercell_structure: 超胞结构
        cutoff_range: 截断距离 (nm)
        distance_matrix: 原子间距离矩阵
        equivalent_count: 等价原子数目矩阵
        shift_vectors: 位移矢量
    """
    data: "sparse.COO"
    primitive_structure: "CrystalStructure"
    supercell_structure: "SupercellStructure"
    cutoff_range: float
    distance_matrix: NDArray[np.float64]
    equivalent_count: NDArray[np.intc]
    shift_vectors: NDArray[np.intc]
    
    @property
    def num_primitive_atoms(self) -> int:
        """原胞原子数"""
        return self.primitive_structure.num_atoms
    
    @property
    def num_supercell_atoms(self) -> int:
        """超胞原子数"""
        return self.supercell_structure.num_atoms
