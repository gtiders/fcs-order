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
        lattice_vectors: 晶格矢量 (3, 3)，单位 nm，列向量形式
        positions: 原子分数坐标 (3, N)
        elements: 元素符号列表（按种类分组）
        atom_counts: 每种元素的原子数量
        atom_types: 每个原子对应的元素类型索引
    """
    lattice_vectors: NDArray[np.float64]
    positions: NDArray[np.float64]
    elements: list[str]
    atom_counts: NDArray[np.intc]
    atom_types: list[int]
    
    @property
    def num_atoms(self) -> int:
        """原子总数"""
        return self.positions.shape[1]
    
    @property
    def num_species(self) -> int:
        """元素种类数"""
        return len(self.elements)
    
    def to_dict(self) -> dict:
        """转换为旧版字典格式（兼容性）"""
        return {
            "lattvec": self.lattice_vectors,
            "positions": self.positions,
            "elements": self.elements,
            "numbers": self.atom_counts,
            "types": self.atom_types,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CrystalStructure":
        """从旧版字典格式创建"""
        return cls(
            lattice_vectors=data["lattvec"],
            positions=data["positions"],
            elements=data["elements"],
            atom_counts=data["numbers"],
            atom_types=data["types"],
        )


@dataclass
class SupercellStructure(CrystalStructure):
    """
    超胞结构数据类
    
    继承自 CrystalStructure，添加超胞特有属性
    
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
    
    def to_dict(self) -> dict:
        """转换为旧版字典格式（兼容性）"""
        result = super().to_dict()
        result["na"] = self.na
        result["nb"] = self.nb
        result["nc"] = self.nc
        return result
    
    @classmethod
    def from_dict(cls, data: dict, primitive: Optional[CrystalStructure] = None) -> "SupercellStructure":
        """从旧版字典格式创建"""
        return cls(
            lattice_vectors=data["lattvec"],
            positions=data["positions"],
            elements=data["elements"],
            atom_counts=data["numbers"],
            atom_types=data["types"],
            grid_size=(data.get("na", 1), data.get("nb", 1), data.get("nc", 1)),
            primitive=primitive,
        )
