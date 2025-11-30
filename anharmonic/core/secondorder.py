"""
二阶力常数计算模块

基于 phonopy 实现二阶谐性力常数的计算
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    PHONOPY_AVAILABLE = True
except ModuleNotFoundError:
    PHONOPY_AVAILABLE = False
    Phonopy = None
    PhonopyAtoms = None


class SecondOrderCalculator:
    """
    二阶力常数计算器
    
    封装 phonopy 进行二阶谐性力常数的计算。
    """
    
    @staticmethod
    def calculate(
        structure: Atoms,
        calculator: "Calculator",
        supercell_matrix: np.ndarray | list[int],
        phonopy_kwargs: Dict[str, Any] | None = None,
        displacement_kwargs: Dict[str, Any] | None = None,
    ) -> "Phonopy":
        """
        计算二阶力常数
        
        Args:
            structure: 原胞结构 (ASE Atoms)
            calculator: ASE 计算器
            supercell_matrix: 超胞矩阵 (3,) 或 (3, 3)
            phonopy_kwargs: 传递给 Phonopy 的参数
            displacement_kwargs: 传递给 generate_displacements 的参数
            
        Returns:
            Phonopy 对象，包含力常数矩阵
        """
        if not PHONOPY_AVAILABLE:
            raise ModuleNotFoundError(
                "phonopy (https://pypi.org/project/phonopy/) 是必需的依赖。"
                "请使用 `pip install phonopy` 安装。"
            )
        
        phonopy_kwargs = phonopy_kwargs or {}
        displacement_kwargs = displacement_kwargs or {}
        
        # 转换为 phonopy 格式
        structure_ph = ase_to_phonopy(structure)
        structure_ph.masses = structure.get_masses()
        
        # 准备超胞
        phonon = Phonopy(structure_ph, supercell_matrix, **phonopy_kwargs)
        phonon.generate_displacements(**displacement_kwargs)
        
        # 计算力
        forces = []
        for supercell_ph in phonon.supercells_with_displacements:
            supercell_ase = phonopy_to_ase(supercell_ph)
            supercell_ase.calc = calculator
            forces.append(supercell_ase.get_forces().copy())
        
        # 生成力常数
        phonon.forces = forces
        phonon.produce_force_constants()
        
        return phonon
    
    @staticmethod
    def get_force_constants(phonon: "Phonopy") -> np.ndarray:
        """
        从 Phonopy 对象获取力常数矩阵
        
        Args:
            phonon: Phonopy 对象
            
        Returns:
            力常数矩阵 (natoms, natoms, 3, 3)
        """
        return phonon.force_constants


def ase_to_phonopy(atoms: Atoms, **kwargs) -> "PhonopyAtoms":
    """将 ASE Atoms 转换为 PhonopyAtoms"""
    if not PHONOPY_AVAILABLE:
        raise ModuleNotFoundError("phonopy 未安装")
    
    return PhonopyAtoms(
        numbers=atoms.numbers,
        cell=atoms.cell,
        positions=atoms.positions,
        **kwargs
    )


def phonopy_to_ase(atoms: "PhonopyAtoms", **kwargs) -> Atoms:
    """将 PhonopyAtoms 转换为 ASE Atoms"""
    return Atoms(
        cell=atoms.cell,
        numbers=atoms.numbers,
        positions=atoms.positions,
        pbc=True,
        **kwargs
    )


def build_supercell(
    primitive: Atoms,
    supercell_matrix: list[int] | np.ndarray,
) -> Atoms:
    """
    从原胞构建超胞
    
    Args:
        primitive: 原胞结构 (ASE Atoms)
        supercell_matrix: 超胞矩阵 (3,) 或 (9,) 或 (3, 3)
        
    Returns:
        超胞结构 (ASE Atoms)
    """
    if not PHONOPY_AVAILABLE:
        raise ModuleNotFoundError("phonopy 未安装")
    
    if len(supercell_matrix) == 3:
        na, nb, nc = supercell_matrix
        matrix = np.array([[na, 0, 0], [0, nb, 0], [0, 0, nc]])
    elif len(supercell_matrix) == 9:
        matrix = np.array(supercell_matrix).reshape(3, 3)
    else:
        matrix = np.array(supercell_matrix)
    
    structure_ph = ase_to_phonopy(primitive)
    phonon = Phonopy(structure_ph, matrix)
    return phonopy_to_ase(phonon.supercell)


# 向后兼容的函数
get_force_constants = SecondOrderCalculator.calculate
