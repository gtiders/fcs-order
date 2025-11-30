"""
结构读取模块

支持多种格式的晶体结构文件读取
"""

from pathlib import Path
from typing import Union

from ase import Atoms
from ase.io import read as ase_read

from anharmonic.models.structure import CrystalStructure


class StructureReader:
    """
    晶体结构读取器
    
    支持通过 ASE 读取多种格式的结构文件
    """
    
    @staticmethod
    def read(
        file_path: Union[str, Path],
        format: str = "auto",
    ) -> CrystalStructure:
        """
        读取结构文件并转换为 CrystalStructure
        
        Args:
            file_path: 结构文件路径
            format: 文件格式，"auto" 表示自动检测
            
        Returns:
            CrystalStructure 对象
        """
        fmt = None if format == "auto" else format
        atoms = ase_read(str(file_path), format=fmt)
        return StructureReader.from_ase_atoms(atoms)
    
    @staticmethod
    def from_ase_atoms(atoms: Atoms) -> CrystalStructure:
        """
        从 ASE Atoms 对象创建 CrystalStructure
        
        Args:
            atoms: ASE Atoms 对象
            
        Returns:
            CrystalStructure 对象
        """
        import numpy as np
        
        # 晶格矢量：ASE 使用 Angstrom，内部使用 nm
        # ASE cell 是行向量，内部使用列向量
        lattice_vectors = 0.1 * atoms.get_cell().T
        
        # 分数坐标：转置为 (3, N) 格式
        positions = np.asarray(atoms.get_scaled_positions()).T
        
        # 解析元素和原子数
        chemical_symbols = atoms.get_chemical_symbols()
        elements, atom_counts = StructureReader._group_elements(chemical_symbols)
        
        # 原子类型索引
        atom_types = np.repeat(
            range(len(atom_counts)),
            atom_counts
        ).tolist()
        
        return CrystalStructure(
            lattice_vectors=lattice_vectors,
            positions=positions,
            elements=elements,
            atom_counts=np.array(atom_counts, dtype=np.intc),
            atom_types=atom_types,
        )
    
    @staticmethod
    def _group_elements(symbols: list[str]) -> tuple[list[str], list[int]]:
        """
        将连续相同的元素符号分组
        
        Args:
            symbols: 元素符号列表
            
        Returns:
            (元素列表, 每种元素的原子数)
        """
        if not symbols:
            return [], []
        
        elements = []
        counts = []
        current_element = symbols[0]
        current_count = 1
        
        for symbol in symbols[1:]:
            if symbol == current_element:
                current_count += 1
            else:
                elements.append(current_element)
                counts.append(current_count)
                current_element = symbol
                current_count = 1
        
        elements.append(current_element)
        counts.append(current_count)
        
        return elements, counts
