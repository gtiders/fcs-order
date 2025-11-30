"""
对称性分析模块

封装 spglib 进行晶体对称性分析
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

import spglib

if TYPE_CHECKING:
    from anharmonic.models.structure import CrystalStructure, SupercellStructure


class SymmetryAnalyzer:
    """
    晶体对称性分析器
    
    封装 spglib 提供的对称性分析功能，计算空间群、
    对称操作（旋转和平移）及其笛卡尔坐标表示。
    
    Attributes:
        space_group_symbol: 国际符号表示的空间群
        num_symmetry_operations: 对称操作的数量
        rotations: 分数坐标下的旋转矩阵 (nsyms, 3, 3)
        translations: 分数坐标下的平移向量 (nsyms, 3)
        cartesian_rotations: 笛卡尔坐标下的旋转矩阵 (nsyms, 3, 3)
        cartesian_translations: 笛卡尔坐标下的平移向量 (nsyms, 3)
    """
    
    def __init__(
        self,
        lattice_vectors: NDArray[np.float64],
        atom_types: list[int],
        positions: NDArray[np.float64],
        symmetry_precision: float = 1e-5,
    ):
        """
        初始化对称性分析器
        
        Args:
            lattice_vectors: 晶格矢量 (3, 3)，列向量形式
            atom_types: 原子类型索引列表
            positions: 原子分数坐标 (natoms, 3)
            symmetry_precision: 对称性搜索容差
        """
        self._lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self._atom_types = np.array(atom_types, dtype=np.intc)
        self._positions = np.array(positions, dtype=np.float64)
        self._symmetry_precision = symmetry_precision
        
        self.num_atoms = self._positions.shape[0]
        
        # 验证输入
        if self._positions.shape[0] != self.num_atoms or self._positions.shape[1] != 3:
            raise ValueError("positions 必须是 (natoms, 3) 的数组")
        if self._lattice_vectors.shape != (3, 3):
            raise ValueError("lattice_vectors 必须是 (3, 3) 的矩阵")
        
        # 初始化对称性数据
        self._origin_shift: NDArray[np.float64]
        self._transformation_matrix: NDArray[np.float64]
        self._rotations: NDArray[np.float64]
        self._translations: NDArray[np.float64]
        self._cartesian_rotations: NDArray[np.float64]
        self._cartesian_translations: NDArray[np.float64]
        
        self._analyze_symmetry()
    
    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        """晶格矢量 (3, 3)"""
        return np.asarray(self._lattice_vectors)
    
    @property
    def positions(self) -> NDArray[np.float64]:
        """原子分数坐标 (natoms, 3)"""
        return np.asarray(self._positions)
    
    @property
    def atom_types(self) -> NDArray[np.intc]:
        """原子类型索引"""
        return np.asarray(self._atom_types)
    
    @property
    def origin_shift(self) -> NDArray[np.float64]:
        """原点偏移"""
        return np.asarray(self._origin_shift)
    
    @property
    def transformation_matrix(self) -> NDArray[np.float64]:
        """变换矩阵"""
        return np.asarray(self._transformation_matrix)
    
    @property
    def rotations(self) -> NDArray[np.float64]:
        """分数坐标下的旋转矩阵 (nsyms, 3, 3)"""
        return np.asarray(self._rotations)
    
    @property
    def translations(self) -> NDArray[np.float64]:
        """分数坐标下的平移向量 (nsyms, 3)"""
        return np.asarray(self._translations)
    
    @property
    def cartesian_rotations(self) -> NDArray[np.float64]:
        """笛卡尔坐标下的旋转矩阵 (nsyms, 3, 3)"""
        return np.asarray(self._cartesian_rotations)
    
    @property
    def cartesian_translations(self) -> NDArray[np.float64]:
        """笛卡尔坐标下的平移向量 (nsyms, 3)"""
        return np.asarray(self._cartesian_translations)
    
    def _analyze_symmetry(self) -> None:
        """使用 spglib 分析晶体对称性"""
        # spglib 使用行向量形式的晶格矢量
        lattice_for_spglib = np.asarray(self._lattice_vectors).T
        
        dataset = spglib.get_symmetry_dataset(
            (lattice_for_spglib, self._positions, self._atom_types),
            symprec=self._symmetry_precision,
        )
        
        if dataset is None:
            raise MemoryError("spglib 无法分析对称性")
        
        self.space_group_symbol = dataset.international.strip()
        self._origin_shift = np.array(dataset.origin_shift, dtype=np.float64)
        self._transformation_matrix = np.array(dataset.transformation_matrix, dtype=np.float64)
        self.num_symmetry_operations = len(dataset.rotations)
        self._rotations = np.array(dataset.rotations, dtype=np.float64)
        self._translations = np.array(dataset.translations, dtype=np.float64)
        
        # 计算笛卡尔坐标下的对称操作
        self._compute_cartesian_operations()
    
    def _compute_cartesian_operations(self) -> None:
        """将对称操作从分数坐标转换为笛卡尔坐标"""
        self._cartesian_rotations = np.empty_like(self._rotations)
        self._cartesian_translations = np.empty_like(self._translations)
        
        inverse_lattice = scipy.linalg.inv(self._lattice_vectors)
        
        for i in range(self.num_symmetry_operations):
            # R_cart = L @ R_frac @ L^(-1)
            self._cartesian_rotations[i] = np.dot(
                self._lattice_vectors,
                np.dot(self._rotations[i], inverse_lattice)
            )
            # t_cart = L @ t_frac
            self._cartesian_translations[i] = np.dot(
                self._lattice_vectors,
                self._translations[i]
            )
    
    def _apply_all_symmetry_operations(
        self,
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        对单个位置应用所有对称操作
        
        Args:
            position: 笛卡尔坐标 (3,)
            
        Returns:
            变换后的位置 (3, nsyms)
        """
        position = np.asarray(position, dtype=np.float64)
        transformed = np.zeros((3, self.num_symmetry_operations), dtype=np.float64)
        
        for i in range(self.num_symmetry_operations):
            transformed[:, i] = (
                np.dot(self._cartesian_rotations[i], position)
                + self._cartesian_translations[i]
            )
        
        return transformed
    
    def map_supercell_atoms(
        self,
        supercell: SupercellStructure | dict,
    ) -> NDArray[np.intc]:
        """
        建立超胞原子在对称操作下的映射关系
        
        对于超胞中的每个原子，计算其在每个对称操作下映射到的原子索引。
        
        Args:
            supercell: 超胞结构对象或字典
            
        Returns:
            映射数组 (nsyms, ntot)，其中 result[isym, i] 表示
            原子 i 在对称操作 isym 下映射到的原子索引
        """
        # 支持字典格式（向后兼容）
        if isinstance(supercell, dict):
            positions = supercell["positions"]
            lattice = supercell["lattvec"]
            grid_size = np.array(
                [supercell["na"], supercell["nb"], supercell["nc"]],
                dtype=np.intc
            )
        else:
            positions = supercell.positions
            lattice = supercell.lattice_vectors
            grid_size = np.array(supercell.grid_size, dtype=np.intc)
        
        num_supercell_atoms = positions.shape[1]
        num_primitive_atoms = num_supercell_atoms // (grid_size[0] * grid_size[1] * grid_size[2])
        
        # 计算原胞原子的笛卡尔坐标
        primitive_cartesian = np.empty((3, num_primitive_atoms), dtype=np.float64)
        for atom_idx in range(num_primitive_atoms):
            for coord in range(3):
                primitive_cartesian[coord, atom_idx] = (
                    self._positions[atom_idx, 0] * self._lattice_vectors[coord, 0]
                    + self._positions[atom_idx, 1] * self._lattice_vectors[coord, 1]
                    + self._positions[atom_idx, 2] * self._lattice_vectors[coord, 2]
                )
        
        # 构建映射数组
        atom_mapping = np.empty(
            (self.num_symmetry_operations, num_supercell_atoms),
            dtype=np.intc
        )
        
        lu_factorization = scipy.linalg.lu_factor(self._lattice_vectors)
        
        for atom_idx in range(num_supercell_atoms):
            # 计算当前原子的笛卡尔坐标
            cartesian_pos = np.empty(3, dtype=np.float64)
            for coord in range(3):
                cartesian_pos[coord] = (
                    positions[0, atom_idx] * lattice[coord, 0]
                    + positions[1, atom_idx] * lattice[coord, 1]
                    + positions[2, atom_idx] * lattice[coord, 2]
                )
            
            # 应用所有对称操作
            transformed_positions = self._apply_all_symmetry_operations(cartesian_pos)
            
            # 为每个对称操作找到对应的原子
            for sym_idx in range(self.num_symmetry_operations):
                found = False
                
                for prim_atom_idx in range(num_primitive_atoms):
                    # 计算变换后位置与原胞原子的差异
                    diff = np.empty(3, dtype=np.float64)
                    for coord in range(3):
                        diff[coord] = (
                            transformed_positions[coord, sym_idx]
                            - primitive_cartesian[coord, prim_atom_idx]
                        )
                    
                    # 求解晶格向量的线性组合
                    fractional_diff = scipy.linalg.lu_solve(lu_factorization, diff)
                    
                    # 四舍五入到最近整数
                    cell_indices = np.empty(3, dtype=np.intc)
                    for coord in range(3):
                        cell_indices[coord] = int(round(fractional_diff[coord]))
                    
                    # 检查是否为有效的晶格平移
                    error = (
                        abs(cell_indices[0] - fractional_diff[0])
                        + abs(cell_indices[1] - fractional_diff[1])
                        + abs(cell_indices[2] - fractional_diff[2])
                    )
                    
                    # 应用周期性边界条件
                    for coord in range(3):
                        cell_indices[coord] = cell_indices[coord] % grid_size[coord]
                    
                    if error < 1e-4:
                        # 计算超胞中的原子索引
                        atom_mapping[sym_idx, atom_idx] = (
                            cell_indices[0]
                            + (cell_indices[1] + cell_indices[2] * grid_size[1]) * grid_size[0]
                        ) * num_primitive_atoms + prim_atom_idx
                        found = True
                        break
                
                if not found:
                    sys.exit(
                        f"错误：对称操作 {sym_idx}，原子 {atom_idx} 找不到等价原子"
                    )
        
        return atom_mapping
    
    @classmethod
    def from_structure(
        cls,
        structure: CrystalStructure,
        symmetry_precision: float = 1e-5,
    ) -> SymmetryAnalyzer:
        """
        从 CrystalStructure 创建对称性分析器
        
        Args:
            structure: 晶体结构对象
            symmetry_precision: 对称性搜索容差
            
        Returns:
            SymmetryAnalyzer 实例
        """
        return cls(
            lattice_vectors=structure.lattice_vectors,
            atom_types=structure.atom_types,
            positions=structure.positions.T,  # 转换为 (natoms, 3)
            symmetry_precision=symmetry_precision,
        )


# 向后兼容的别名
SymmetryOperations = SymmetryAnalyzer
