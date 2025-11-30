"""
晶体结构数据模型

定义原胞和超胞结构的数据类
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class CrystalStructure:
    """
    晶体结构数据类
    
    Attributes:
        lattice_vectors: 晶格矢量 (3, 3), 单位 nm, 列向量形式
        positions: 原子分数坐标 (3, N)
        elements: 元素符号列表（按种类分组）
        atom_counts: 每种元素的原子数量
        atom_types: 每个原子对应的元素类型索引
    """
    lattice_vectors: NDArray[np.float64]
    positions: NDArray[np.float64]
    elements: list[str]
    atom_counts: NDArray[np.int64]
    atom_types: list[int]
    
    @property
    def num_atoms(self) -> int:
        """原子总数"""
        return self.positions.shape[1]
    
    @property
    def num_species(self) -> int:
        """元素种类数"""
        return len(self.elements)
    
    def to_atoms(self, calculator=None):
        """
        转换为 ASE Atoms 对象
        
        Args:
            calculator: ASE 计算器（可选）
            
        Returns:
            ASE Atoms 对象
        """
        from ase import Atoms
        
        symbols = np.repeat(self.elements, self.atom_counts).tolist()
        
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=self.positions.T,
            cell=self.lattice_vectors.T * 10.0,  # nm -> Angstrom
            pbc=True,
        )
        
        if calculator is not None:
            atoms.calc = calculator
        
        return atoms


@dataclass
class SupercellStructure(CrystalStructure):
    """
    超胞结构数据类
    
    继承自 CrystalStructure, 添加超胞特有属性
    
    Attributes:
        grid_size: 超胞扩展倍数 (na, nb, nc)
        primitive: 对应的原胞结构引用
    """
    grid_size: tuple[int, int, int] = (1, 1, 1)
    primitive: Optional[CrystalStructure] = None
    
    @property
    def na(self) -> int:
        return self.grid_size[0]
    
    @property
    def nb(self) -> int:
        return self.grid_size[1]
    
    @property
    def nc(self) -> int:
        return self.grid_size[2]
    
    @property
    def num_primitive_atoms(self) -> int:
        """原胞中的原子数"""
        if self.primitive is not None:
            return self.primitive.num_atoms
        return self.num_atoms // (self.na * self.nb * self.nc)
