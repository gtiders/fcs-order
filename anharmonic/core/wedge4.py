"""
四阶 Wedge 约简模块

实现四阶力常数的不可约集合约简，
通过对称性分析找出独立的力常数分量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from rich.progress import track

from anharmonic.core.gaussian import gaussian_elimination
from anharmonic.core.jit_kernels import (
    jit_supercell_index_to_cell_atom,
    jit_cell_atom_to_supercell_index,
)
from anharmonic.core.jit_kernels4 import (
    jit_quartet_in_list,
    jit_quartets_equal,
    jit_build_transformation_array4,
)

if TYPE_CHECKING:
    from anharmonic.core.symmetry import SymmetryAnalyzer
    from anharmonic.models.structure import CrystalStructure, SupercellStructure

# 四元组的24种置换
QUARTET_PERMUTATIONS = np.array([
    [0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [0, 3, 1, 2],
    [0, 3, 2, 1], [0, 2, 3, 1], [1, 0, 2, 3], [1, 0, 3, 2],
    [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0],
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1],
    [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
], dtype=np.intc)


class QuartetWedge:
    """
    四阶力常数不可约集合约简器
    
    通过对称性分析和置换对称性，找出四阶力常数的
    不可约子集，减少需要计算的力常数数量。
    
    Attributes:
        num_quartet_classes: 不等价四元组类的数量
        quartet_list: 代表性四元组列表 (4, nlist)
        equivalent_count: 每个类的等价四元组数量 (nlist,)
        equivalent_quartets: 等价四元组 (4, max_equiv, nlist)
        independent_basis_count: 每个类的独立基数量 (nlist,)
        independent_basis: 独立基索引 (81, nlist)
        transformation_array: 变换数组 (81, 81, max_equiv, nlist)
    """
    
    def __init__(
        self,
        primitive: CrystalStructure | dict,
        supercell: SupercellStructure | dict,
        symmetry: SymmetryAnalyzer,
        distance_matrix: NDArray[np.float64],
        equivalent_count_matrix: NDArray[np.intc],
        shift_vectors: NDArray[np.intc],
        cutoff_range: float,
    ):
        """
        初始化四阶 Wedge 约简器
        
        Args:
            primitive: 原胞结构
            supercell: 超胞结构
            symmetry: 对称性分析器
            distance_matrix: 原子间距离矩阵 (natoms, ntot)
            equivalent_count_matrix: 等价原子数目矩阵 (natoms, ntot)
            shift_vectors: 位移矢量 (natoms, ntot, max_equiv)
            cutoff_range: 截断距离 (nm)
        """
        if isinstance(primitive, dict):
            self._primitive = primitive
        else:
            self._primitive = primitive.to_dict()
        
        if isinstance(supercell, dict):
            self._supercell = supercell
        else:
            self._supercell = supercell.to_dict()
        
        self._symmetry = symmetry
        self._distance_matrix = distance_matrix
        self._equivalent_count_matrix = equivalent_count_matrix
        self._shift_vectors = shift_vectors
        self._cutoff_range = cutoff_range
        
        self._init_storage()
        self._init_rotation_matrices()
        self._reduce()
    
    def _init_storage(self) -> None:
        """初始化存储数组"""
        initial_size = 32
        max_equiv = 24 * self._symmetry.num_symmetry_operations
        
        self.equivalent_count = np.empty(initial_size, dtype=np.intc)
        self.equivalent_quartets = np.empty((4, max_equiv, initial_size), dtype=np.intc)
        self.transformation = np.empty((81, 81, max_equiv, initial_size), dtype=np.float64)
        self.transformation_array = np.empty((81, 81, max_equiv, initial_size), dtype=np.float64)
        self.transformation_auxiliary = np.empty((81, 81, initial_size), dtype=np.float64)
        self.independent_basis_count = np.empty(initial_size, dtype=np.intc)
        self.independent_basis = np.empty((81, initial_size), dtype=np.intc)
        self.quartet_list = np.empty((4, initial_size), dtype=np.intc)
        
        all_initial_size = 512
        self._all_quartets = np.empty((4, all_initial_size), dtype=np.intc)
        
        self._alloc_size = initial_size
        self._all_alloc_size = all_initial_size
    
    def _expand_storage(self) -> None:
        """扩展存储数组"""
        self._alloc_size *= 2
        self.equivalent_count = np.concatenate([self.equivalent_count, self.equivalent_count])
        self.equivalent_quartets = np.concatenate([self.equivalent_quartets, self.equivalent_quartets], axis=-1)
        self.transformation = np.concatenate([self.transformation, self.transformation], axis=-1)
        self.transformation_array = np.concatenate([self.transformation_array, self.transformation_array], axis=-1)
        self.transformation_auxiliary = np.concatenate([self.transformation_auxiliary, self.transformation_auxiliary], axis=-1)
        self.independent_basis_count = np.concatenate([self.independent_basis_count, self.independent_basis_count])
        self.independent_basis = np.concatenate([self.independent_basis, self.independent_basis], axis=-1)
        self.quartet_list = np.concatenate([self.quartet_list, self.quartet_list], axis=-1)
    
    def _expand_all_quartets(self) -> None:
        """扩展已处理四元组存储"""
        self._all_alloc_size *= 2
        self._all_quartets = np.concatenate([self._all_quartets, self._all_quartets], axis=-1)
    
    def _init_rotation_matrices(self) -> None:
        """初始化旋转矩阵（81x81 基变换）"""
        num_syms = self._symmetry.num_symmetry_operations
        cartesian_rotations = np.transpose(self._symmetry.cartesian_rotations, (1, 2, 0))
        
        # 24种置换 × num_syms对称操作 × 81×81基变换
        self._rotation_matrices = np.empty((24, num_syms, 81, 81), dtype=np.float64)
        
        basis = np.empty(4, dtype=np.intc)
        for perm_idx in range(24):
            for sym_idx in range(num_syms):
                for i_prime in range(3):
                    for j_prime in range(3):
                        for k_prime in range(3):
                            for l_prime in range(3):
                                index_prime = ((i_prime * 3 + j_prime) * 3 + k_prime) * 3 + l_prime
                                for i in range(3):
                                    basis[0] = i
                                    for j in range(3):
                                        basis[1] = j
                                        for k in range(3):
                                            basis[2] = k
                                            for l in range(3):
                                                basis[3] = l
                                                index = ((i * 3 + j) * 3 + k) * 3 + l
                                                i_perm = basis[QUARTET_PERMUTATIONS[perm_idx, 0]]
                                                j_perm = basis[QUARTET_PERMUTATIONS[perm_idx, 1]]
                                                k_perm = basis[QUARTET_PERMUTATIONS[perm_idx, 2]]
                                                l_perm = basis[QUARTET_PERMUTATIONS[perm_idx, 3]]
                                                self._rotation_matrices[perm_idx, sym_idx, index_prime, index] = (
                                                    cartesian_rotations[i_prime, i_perm, sym_idx]
                                                    * cartesian_rotations[j_prime, j_perm, sym_idx]
                                                    * cartesian_rotations[k_prime, k_perm, sym_idx]
                                                    * cartesian_rotations[l_prime, l_perm, sym_idx]
                                                )
        
        # 构建用于检测恒等变换的矩阵
        self._rotation_identity_diff = self._rotation_matrices.copy()
        self._nonzero_flags = np.zeros((24, num_syms, 81), dtype=np.intc)
        
        for perm_idx in range(24):
            for sym_idx in range(num_syms):
                for index_prime in range(81):
                    self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index_prime] -= 1.0
                    for index in range(81):
                        if np.abs(self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index]) > 1e-12:
                            self._nonzero_flags[perm_idx, sym_idx, index_prime] = 1
                        else:
                            self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index] = 0.0
    
    def _reduce(self) -> None:
        """执行不可约约简"""
        cutoff_squared = self._cutoff_range * self._cutoff_range
        
        grid_size = np.array([
            self._supercell["na"],
            self._supercell["nb"],
            self._supercell["nc"],
        ], dtype=np.intc)
        
        num_syms = self._symmetry.num_symmetry_operations
        num_primitive_atoms = len(self._primitive["types"])
        num_supercell_atoms = len(self._supercell["types"])
        
        lattice = self._supercell["lattvec"]
        cartesian_positions = np.dot(lattice, self._supercell["positions"])
        
        # 构建27个邻居位移
        shifts_27 = np.empty((27, 3), dtype=np.intc)
        idx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shifts_27[idx] = [i, j, k]
                    idx += 1
        
        symmetry_mapping = self._symmetry.map_supercell_atoms(self._supercell)
        cell_indices, atom_indices = jit_supercell_index_to_cell_atom(grid_size, num_primitive_atoms)
        
        self.num_quartet_classes = 0
        self._num_all_quartets = 0
        
        quartet = np.empty(4, dtype=np.intc)
        quartet_permuted = np.empty(4, dtype=np.intc)
        quartet_transformed = np.empty(4, dtype=np.intc)
        equiv_list = np.empty((4, num_syms * 24), dtype=np.intc)
        coefficients = np.empty((24 * num_syms * 81, 81), dtype=np.float64)
        
        # 扫描所有原子四元组
        for atom_i in track(range(num_primitive_atoms), description="扫描原子四元组"):
            for atom_j in range(num_supercell_atoms):
                dist_ij = self._distance_matrix[atom_i, atom_j]
                if dist_ij >= self._cutoff_range:
                    continue
                
                for atom_k in range(num_supercell_atoms):
                    dist_ik = self._distance_matrix[atom_i, atom_k]
                    if dist_ik >= self._cutoff_range:
                        continue
                    
                    for atom_l in range(num_supercell_atoms):
                        dist_il = self._distance_matrix[atom_i, atom_l]
                        if dist_il >= self._cutoff_range:
                            continue
                        
                        # 检查是否已处理过
                        quartet[0] = atom_i
                        quartet[1] = atom_j
                        quartet[2] = atom_k
                        quartet[3] = atom_l
                        
                        if jit_quartet_in_list(quartet, self._all_quartets, self._num_all_quartets):
                            continue
                        
                        # 新的不等价四元组类
                        self.num_quartet_classes += 1
                        if self.num_quartet_classes >= self._alloc_size:
                            self._expand_storage()
                        
                        class_idx = self.num_quartet_classes - 1
                        self.quartet_list[:, class_idx] = quartet
                        self.equivalent_count[class_idx] = 0
                        coefficients[:, :] = 0.0
                        num_nonzero = 0
                        
                        # 扫描所有置换和对称操作
                        for perm_idx in range(24):
                            for p in range(4):
                                quartet_permuted[p] = quartet[QUARTET_PERMUTATIONS[perm_idx, p]]
                            
                            for sym_idx in range(num_syms):
                                for p in range(4):
                                    quartet_transformed[p] = symmetry_mapping[sym_idx, quartet_permuted[p]]
                                
                                vec1 = cell_indices[:, symmetry_mapping[sym_idx, quartet_permuted[0]]].copy()
                                vec2 = cell_indices[:, symmetry_mapping[sym_idx, quartet_permuted[1]]].copy()
                                vec3 = cell_indices[:, symmetry_mapping[sym_idx, quartet_permuted[2]]].copy()
                                vec4 = cell_indices[:, symmetry_mapping[sym_idx, quartet_permuted[3]]].copy()
                                
                                # 将第一个原子移到原点晶胞
                                if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                    vec4 = (vec4 - vec1) % grid_size
                                    vec3 = (vec3 - vec1) % grid_size
                                    vec2 = (vec2 - vec1) % grid_size
                                    vec1[:] = 0
                                    species = [atom_indices[symmetry_mapping[sym_idx, quartet_permuted[p]]] for p in range(4)]
                                    quartet_transformed[0] = jit_cell_atom_to_supercell_index(vec1, species[0], grid_size, num_primitive_atoms)
                                    quartet_transformed[1] = jit_cell_atom_to_supercell_index(vec2, species[1], grid_size, num_primitive_atoms)
                                    quartet_transformed[2] = jit_cell_atom_to_supercell_index(vec3, species[2], grid_size, num_primitive_atoms)
                                    quartet_transformed[3] = jit_cell_atom_to_supercell_index(vec4, species[3], grid_size, num_primitive_atoms)
                                
                                # 检查是否为新的等价四元组
                                is_new = (perm_idx == 0 and sym_idx == 0) or not (
                                    jit_quartets_equal(quartet_transformed, quartet) or
                                    jit_quartet_in_list(quartet_transformed, equiv_list, self.equivalent_count[class_idx])
                                )
                                
                                if is_new:
                                    equiv_idx = self.equivalent_count[class_idx]
                                    self.equivalent_count[class_idx] += 1
                                    
                                    for coord in range(4):
                                        equiv_list[coord, equiv_idx] = quartet_transformed[coord]
                                        self.equivalent_quartets[coord, equiv_idx, class_idx] = quartet_transformed[coord]
                                    
                                    self._num_all_quartets += 1
                                    if self._num_all_quartets >= self._all_alloc_size:
                                        self._expand_all_quartets()
                                    for coord in range(4):
                                        self._all_quartets[coord, self._num_all_quartets - 1] = quartet_transformed[coord]
                                    
                                    for basis_i in range(81):
                                        for basis_j in range(81):
                                            self.transformation[basis_i, basis_j, equiv_idx, class_idx] = \
                                                self._rotation_matrices[perm_idx, sym_idx, basis_i, basis_j]
                                
                                # 如果变换回原四元组，记录系数
                                if jit_quartets_equal(quartet_transformed, quartet):
                                    for index_prime in range(81):
                                        if self._nonzero_flags[perm_idx, sym_idx, index_prime]:
                                            for index in range(81):
                                                coefficients[num_nonzero, index] = \
                                                    self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index]
                                            num_nonzero += 1
                        
                        # 高斯消元确定独立基
                        coeff_reduced = np.zeros((max(num_nonzero, 81), 81), dtype=np.float64)
                        for i in range(num_nonzero):
                            for j in range(81):
                                coeff_reduced[i, j] = coefficients[i, j]
                        
                        b, independent = gaussian_elimination(coeff_reduced)
                        
                        for i in range(81):
                            for j in range(81):
                                self.transformation_auxiliary[i, j, class_idx] = b[i, j]
                        
                        self.independent_basis_count[class_idx] = len(independent)
                        for i in range(len(independent)):
                            self.independent_basis[i, class_idx] = independent[i]
        
        # 构建最终变换数组
        self.transformation_array[:, :, :, :] = 0.0
        jit_build_transformation_array4(
            self.transformation,
            self.transformation_auxiliary,
            self.equivalent_count,
            self.independent_basis_count,
            self.num_quartet_classes,
            self.transformation_array,
        )
    
    def get_irreducible_displacements(self) -> list[tuple[int, int, int, int, int, int]]:
        """
        获取不可约位移列表
        
        Returns:
            每个元素为 (atom_i, atom_j, atom_k, coord_i, coord_j, coord_k)
        """
        displacement_list = []
        
        for class_idx in range(self.num_quartet_classes):
            for basis_idx in range(self.independent_basis_count[class_idx]):
                basis_value = self.independent_basis[basis_idx, class_idx]
                coord_l = basis_value // 27
                coord_m = (basis_value % 27) // 9
                coord_n = (basis_value % 9) // 3
                
                atom_i = self.quartet_list[0, class_idx]
                atom_j = self.quartet_list[1, class_idx]
                atom_k = self.quartet_list[2, class_idx]
                
                displacement_list.append((atom_k, atom_j, atom_i, coord_n, coord_m, coord_l))
        
        # 去重
        unique_displacements = []
        for item in displacement_list:
            if item not in unique_displacements:
                unique_displacements.append(item)
        
        return unique_displacements
    
    # 向后兼容属性
    @property
    def nlist(self) -> int:
        return self.num_quartet_classes
    
    @property
    def nequi(self) -> NDArray[np.intc]:
        return self.equivalent_count
    
    @property
    def allequilist(self) -> NDArray[np.intc]:
        return self.equivalent_quartets
    
    @property
    def nindependentbasis(self) -> NDArray[np.intc]:
        return self.independent_basis_count
    
    @property
    def independentbasis(self) -> NDArray[np.intc]:
        return self.independent_basis
    
    @property
    def llist(self) -> NDArray[np.intc]:
        return self.quartet_list
    
    @property
    def transformationarray(self) -> NDArray[np.float64]:
        return self.transformation_array
    
    def build_list4(self) -> list[tuple[int, int, int, int, int, int]]:
        """向后兼容的别名"""
        return self.get_irreducible_displacements()


# 向后兼容别名
Wedge4 = QuartetWedge
