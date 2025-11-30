"""
Wedge 约简模块

实现三阶力常数的不可约集合约简，
通过对称性分析找出独立的力常数分量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from rich.progress import track

from anharmonic.core.gaussian import gaussian_elimination
from anharmonic.core.jit_kernels import (
    jit_triplet_in_list,
    jit_triplets_equal,
    jit_supercell_index_to_cell_atom,
    jit_cell_atom_to_supercell_index,
    jit_build_transformation_array,
)

if TYPE_CHECKING:
    from anharmonic.core.symmetry import SymmetryAnalyzer
    from anharmonic.models.structure import CrystalStructure, SupercellStructure

# 三元组的6种置换
TRIPLET_PERMUTATIONS = np.array([
    [0, 1, 2],
    [1, 0, 2],
    [2, 1, 0],
    [0, 2, 1],
    [1, 2, 0],
    [2, 0, 1],
], dtype=np.int64)


class TripletWedge:
    """
    三阶力常数不可约集合约简器
    
    通过对称性分析和置换对称性，找出三阶力常数的
    不可约子集，减少需要计算的力常数数量。
    
    Attributes:
        num_triplet_classes: 不等价三元组类的数量
        triplet_list: 代表性三元组列表 (3, nlist)
        equivalent_count: 每个类的等价三元组数量 (nlist,)
        equivalent_triplets: 等价三元组 (3, max_equiv, nlist)
        independent_basis_count: 每个类的独立基数量 (nlist,)
        independent_basis: 独立基索引 (27, nlist)
        transformation_array: 变换数组 (27, 27, max_equiv, nlist)
    """
    
    def __init__(
        self,
        primitive: CrystalStructure,
        supercell: SupercellStructure,
        symmetry: SymmetryAnalyzer,
        distance_matrix: NDArray[np.float64],
        equivalent_count_matrix: NDArray[np.int64],
        shift_vectors: NDArray[np.int64],
        cutoff_range: float,
    ):
        """
        初始化 Wedge 约简器
        
        Args:
            primitive: 原胞结构
            supercell: 超胞结构
            symmetry: 对称性分析器
            distance_matrix: 原子间距离矩阵 (natoms, ntot)
            equivalent_count_matrix: 等价原子数目矩阵 (natoms, ntot)
            shift_vectors: 位移矢量 (natoms, ntot, max_equiv)
            cutoff_range: 截断距离 (nm)
        """
        self._primitive = primitive
        self._supercell = supercell
        
        self._symmetry = symmetry
        self._distance_matrix = distance_matrix
        self._equivalent_count_matrix = equivalent_count_matrix
        self._shift_vectors = shift_vectors
        self._cutoff_range = cutoff_range
        
        # 初始化存储数组
        self._init_storage()
        
        # 初始化旋转矩阵
        self._init_rotation_matrices()
        
        # 执行约简
        self._reduce()
    
    def _init_storage(self) -> None:
        """初始化存储数组"""
        initial_size = 16
        max_equiv = 6 * self._symmetry.num_symmetry_operations
        
        self.equivalent_count = np.empty(initial_size, dtype=np.int64)
        self.equivalent_triplets = np.empty((3, max_equiv, initial_size), dtype=np.int64)
        self.transformation = np.empty((27, 27, max_equiv, initial_size), dtype=np.float64)
        self.transformation_array = np.empty((27, 27, max_equiv, initial_size), dtype=np.float64)
        self.transformation_auxiliary = np.empty((27, 27, initial_size), dtype=np.float64)
        self.independent_basis_count = np.empty(initial_size, dtype=np.int64)
        self.independent_basis = np.empty((27, initial_size), dtype=np.int64)
        self.triplet_list = np.empty((3, initial_size), dtype=np.int64)
        
        # 存储所有已处理的三元组
        all_initial_size = 512
        self._all_triplets = np.empty((3, all_initial_size), dtype=np.int64)
        
        self._alloc_size = initial_size
        self._all_alloc_size = all_initial_size
    
    def _expand_storage(self) -> None:
        """扩展存储数组"""
        self._alloc_size *= 2
        self.equivalent_count = np.concatenate([self.equivalent_count, self.equivalent_count])
        self.equivalent_triplets = np.concatenate([self.equivalent_triplets, self.equivalent_triplets], axis=-1)
        self.transformation = np.concatenate([self.transformation, self.transformation], axis=-1)
        self.transformation_array = np.concatenate([self.transformation_array, self.transformation_array], axis=-1)
        self.transformation_auxiliary = np.concatenate([self.transformation_auxiliary, self.transformation_auxiliary], axis=-1)
        self.independent_basis_count = np.concatenate([self.independent_basis_count, self.independent_basis_count])
        self.independent_basis = np.concatenate([self.independent_basis, self.independent_basis], axis=-1)
        self.triplet_list = np.concatenate([self.triplet_list, self.triplet_list], axis=-1)
    
    def _expand_all_triplets(self) -> None:
        """扩展已处理三元组存储"""
        self._all_alloc_size *= 2
        self._all_triplets = np.concatenate([self._all_triplets, self._all_triplets], axis=-1)
    
    def _init_rotation_matrices(self) -> None:
        """初始化旋转矩阵"""
        num_syms = self._symmetry.num_symmetry_operations
        cartesian_rotations = np.transpose(self._symmetry.cartesian_rotations, (1, 2, 0))
        
        # 6种置换 × num_syms对称操作 × 27×27基变换
        self._rotation_matrices = np.empty((6, num_syms, 27, 27), dtype=np.float64)
        
        basis = np.empty(3, dtype=np.int64)
        for perm_idx in range(6):
            for sym_idx in range(num_syms):
                for i_prime in range(3):
                    for j_prime in range(3):
                        for k_prime in range(3):
                            index_prime = (i_prime * 3 + j_prime) * 3 + k_prime
                            for i in range(3):
                                basis[0] = i
                                for j in range(3):
                                    basis[1] = j
                                    for k in range(3):
                                        basis[2] = k
                                        index = i * 9 + j * 3 + k
                                        i_perm = basis[TRIPLET_PERMUTATIONS[perm_idx, 0]]
                                        j_perm = basis[TRIPLET_PERMUTATIONS[perm_idx, 1]]
                                        k_perm = basis[TRIPLET_PERMUTATIONS[perm_idx, 2]]
                                        self._rotation_matrices[perm_idx, sym_idx, index_prime, index] = (
                                            cartesian_rotations[i_prime, i_perm, sym_idx]
                                            * cartesian_rotations[j_prime, j_perm, sym_idx]
                                            * cartesian_rotations[k_prime, k_perm, sym_idx]
                                        )
        
        # 构建用于检测恒等变换的矩阵
        self._rotation_identity_diff = self._rotation_matrices.copy()
        self._nonzero_flags = np.zeros((6, num_syms, 27), dtype=np.int64)
        
        for perm_idx in range(6):
            for sym_idx in range(num_syms):
                for index_prime in range(27):
                    self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index_prime] -= 1.0
                    for index in range(27):
                        if np.abs(self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index]) > 1e-12:
                            self._nonzero_flags[perm_idx, sym_idx, index_prime] = 1
                        else:
                            self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index] = 0.0
    
    def _reduce(self) -> None:
        """执行不可约约简"""
        cutoff_squared = self._cutoff_range * self._cutoff_range
        
        grid_size = np.array([
            self._supercell.na,
            self._supercell.nb,
            self._supercell.nc,
        ], dtype=np.int64)
        
        num_syms = self._symmetry.num_symmetry_operations
        num_primitive_atoms = self._primitive.num_atoms
        num_supercell_atoms = self._supercell.num_atoms
        
        lattice = self._supercell.lattice_vectors
        cartesian_positions = np.dot(lattice, self._supercell.positions)
        
        # 构建27个邻居位移
        shifts_27 = np.empty((27, 3), dtype=np.int64)
        idx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shifts_27[idx] = [i, j, k]
                    idx += 1
        
        # 获取对称操作的原子映射
        symmetry_mapping = self._symmetry.map_supercell_atoms(self._supercell)
        cell_indices, atom_indices = jit_supercell_index_to_cell_atom(grid_size, num_primitive_atoms)
        
        self.num_triplet_classes = 0
        self._num_all_triplets = 0
        
        triplet = np.empty(3, dtype=np.int64)
        triplet_permuted = np.empty(3, dtype=np.int64)
        triplet_transformed = np.empty(3, dtype=np.int64)
        shift_i_all = np.empty((3, 27), dtype=np.int64)
        shift_j_all = np.empty((3, 27), dtype=np.int64)
        equiv_list = np.empty((3, num_syms * 6), dtype=np.int64)
        coefficients = np.empty((6 * num_syms * 27, 27), dtype=np.float64)
        
        # 扫描所有原子三元组
        for atom_i in track(range(num_primitive_atoms), description="扫描原子三元组"):
            for atom_j in track(range(num_supercell_atoms), description=f"处理原子 {atom_i}"):
                dist_ij = self._distance_matrix[atom_i, atom_j]
                if dist_ij >= self._cutoff_range:
                    continue
                
                num_equiv_ij = self._equivalent_count_matrix[atom_i, atom_j]
                for k in range(num_equiv_ij):
                    shift_i_all[:, k] = shifts_27[self._shift_vectors[atom_i, atom_j, k], :]
                
                for atom_k in range(num_supercell_atoms):
                    dist_ik = self._distance_matrix[atom_i, atom_k]
                    if dist_ik >= self._cutoff_range:
                        continue
                    
                    num_equiv_ik = self._equivalent_count_matrix[atom_i, atom_k]
                    for k in range(num_equiv_ik):
                        shift_j_all[:, k] = shifts_27[self._shift_vectors[atom_i, atom_k, k], :]
                    
                    # 检查 j-k 距离
                    dist_jk_min_squared = np.inf
                    for idx_i in range(num_equiv_ij):
                        cart_j = np.array([
                            shift_i_all[0, idx_i] * lattice[0, 0] + shift_i_all[1, idx_i] * lattice[0, 1] + shift_i_all[2, idx_i] * lattice[0, 2] + cartesian_positions[0, atom_j],
                            shift_i_all[0, idx_i] * lattice[1, 0] + shift_i_all[1, idx_i] * lattice[1, 1] + shift_i_all[2, idx_i] * lattice[1, 2] + cartesian_positions[1, atom_j],
                            shift_i_all[0, idx_i] * lattice[2, 0] + shift_i_all[1, idx_i] * lattice[2, 1] + shift_i_all[2, idx_i] * lattice[2, 2] + cartesian_positions[2, atom_j],
                        ])
                        for idx_j in range(num_equiv_ik):
                            cart_k = np.array([
                                shift_j_all[0, idx_j] * lattice[0, 0] + shift_j_all[1, idx_j] * lattice[0, 1] + shift_j_all[2, idx_j] * lattice[0, 2] + cartesian_positions[0, atom_k],
                                shift_j_all[0, idx_j] * lattice[1, 0] + shift_j_all[1, idx_j] * lattice[1, 1] + shift_j_all[2, idx_j] * lattice[1, 2] + cartesian_positions[1, atom_k],
                                shift_j_all[0, idx_j] * lattice[2, 0] + shift_j_all[1, idx_j] * lattice[2, 1] + shift_j_all[2, idx_j] * lattice[2, 2] + cartesian_positions[2, atom_k],
                            ])
                            dist_squared = np.sum((cart_j - cart_k) ** 2)
                            if dist_squared < dist_jk_min_squared:
                                dist_jk_min_squared = dist_squared
                    
                    if dist_jk_min_squared >= cutoff_squared:
                        continue
                    
                    # 检查是否已处理过
                    triplet[0] = atom_i
                    triplet[1] = atom_j
                    triplet[2] = atom_k
                    
                    if jit_triplet_in_list(triplet, self._all_triplets, self._num_all_triplets):
                        continue
                    
                    # 新的不等价三元组类
                    self.num_triplet_classes += 1
                    if self.num_triplet_classes >= self._alloc_size:
                        self._expand_storage()
                    
                    class_idx = self.num_triplet_classes - 1
                    self.triplet_list[0, class_idx] = atom_i
                    self.triplet_list[1, class_idx] = atom_j
                    self.triplet_list[2, class_idx] = atom_k
                    self.equivalent_count[class_idx] = 0
                    coefficients[:, :] = 0.0
                    num_nonzero = 0
                    
                    # 扫描所有置换和对称操作
                    for perm_idx in range(6):
                        triplet_permuted[0] = triplet[TRIPLET_PERMUTATIONS[perm_idx, 0]]
                        triplet_permuted[1] = triplet[TRIPLET_PERMUTATIONS[perm_idx, 1]]
                        triplet_permuted[2] = triplet[TRIPLET_PERMUTATIONS[perm_idx, 2]]
                        
                        for sym_idx in range(num_syms):
                            triplet_transformed[0] = symmetry_mapping[sym_idx, triplet_permuted[0]]
                            triplet_transformed[1] = symmetry_mapping[sym_idx, triplet_permuted[1]]
                            triplet_transformed[2] = symmetry_mapping[sym_idx, triplet_permuted[2]]
                            
                            vec1 = cell_indices[:, symmetry_mapping[sym_idx, triplet_permuted[0]]].copy()
                            vec2 = cell_indices[:, symmetry_mapping[sym_idx, triplet_permuted[1]]].copy()
                            vec3 = cell_indices[:, symmetry_mapping[sym_idx, triplet_permuted[2]]].copy()
                            
                            # 将第一个原子移到原点晶胞
                            if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                vec3 = (vec3 - vec1) % grid_size
                                vec2 = (vec2 - vec1) % grid_size
                                vec1[:] = 0
                                species1 = atom_indices[symmetry_mapping[sym_idx, triplet_permuted[0]]]
                                species2 = atom_indices[symmetry_mapping[sym_idx, triplet_permuted[1]]]
                                species3 = atom_indices[symmetry_mapping[sym_idx, triplet_permuted[2]]]
                                triplet_transformed[0] = jit_cell_atom_to_supercell_index(vec1, species1, grid_size, num_primitive_atoms)
                                triplet_transformed[1] = jit_cell_atom_to_supercell_index(vec2, species2, grid_size, num_primitive_atoms)
                                triplet_transformed[2] = jit_cell_atom_to_supercell_index(vec3, species3, grid_size, num_primitive_atoms)
                            
                            # 检查是否为新的等价三元组
                            is_new = (perm_idx == 0 and sym_idx == 0) or not (
                                jit_triplets_equal(triplet_transformed, triplet) or
                                jit_triplet_in_list(triplet_transformed, equiv_list, self.equivalent_count[class_idx])
                            )
                            
                            if is_new:
                                equiv_idx = self.equivalent_count[class_idx]
                                self.equivalent_count[class_idx] += 1
                                
                                for coord in range(3):
                                    equiv_list[coord, equiv_idx] = triplet_transformed[coord]
                                    self.equivalent_triplets[coord, equiv_idx, class_idx] = triplet_transformed[coord]
                                
                                # 添加到已处理列表
                                self._num_all_triplets += 1
                                if self._num_all_triplets >= self._all_alloc_size:
                                    self._expand_all_triplets()
                                for coord in range(3):
                                    self._all_triplets[coord, self._num_all_triplets - 1] = triplet_transformed[coord]
                                
                                # 存储变换矩阵
                                for basis_i in range(27):
                                    for basis_j in range(27):
                                        self.transformation[basis_i, basis_j, equiv_idx, class_idx] = \
                                            self._rotation_matrices[perm_idx, sym_idx, basis_i, basis_j]
                            
                            # 如果变换回原三元组，记录系数
                            if jit_triplets_equal(triplet_transformed, triplet):
                                for index_prime in range(27):
                                    if self._nonzero_flags[perm_idx, sym_idx, index_prime]:
                                        for index in range(27):
                                            coefficients[num_nonzero, index] = \
                                                self._rotation_identity_diff[perm_idx, sym_idx, index_prime, index]
                                        num_nonzero += 1
                    
                    # 高斯消元确定独立基
                    coeff_reduced = np.zeros((max(num_nonzero, 27), 27), dtype=np.float64)
                    for i in range(num_nonzero):
                        for j in range(27):
                            coeff_reduced[i, j] = coefficients[i, j]
                    
                    b, independent = gaussian_elimination(coeff_reduced)
                    
                    for i in range(27):
                        for j in range(27):
                            self.transformation_auxiliary[i, j, class_idx] = b[i, j]
                    
                    self.independent_basis_count[class_idx] = len(independent)
                    for i in range(len(independent)):
                        self.independent_basis[i, class_idx] = independent[i]
        
        # 构建最终变换数组
        self.transformation_array[:, :, :, :] = 0.0
        jit_build_transformation_array(
            self.transformation,
            self.transformation_auxiliary,
            self.equivalent_count,
            self.independent_basis_count,
            self.num_triplet_classes,
            self.transformation_array,
        )
    
    def get_irreducible_displacements(self) -> list[tuple[int, int, int, int]]:
        """
        获取不可约位移列表
        
        Returns:
            每个元素为 (atom_i, atom_j, coord_i, coord_j)
        """
        displacement_list = []
        
        for class_idx in range(self.num_triplet_classes):
            for basis_idx in range(self.independent_basis_count[class_idx]):
                basis_value = self.independent_basis[basis_idx, class_idx]
                coord_l = basis_value // 9
                coord_m = (basis_value % 9) // 3
                # coord_n = basis_value % 3  # 保留用于调试
                
                atom_i = self.triplet_list[0, class_idx]
                atom_j = self.triplet_list[1, class_idx]
                
                displacement_list.append((atom_i, atom_j, coord_l, coord_m))
        
        # 去重
        unique_displacements = []
        for item in displacement_list:
            four_numbers = (item[0], item[1], item[2], item[3])
            if four_numbers not in unique_displacements:
                unique_displacements.append(four_numbers)
        
        return unique_displacements
    
    # 向后兼容的属性名
    @property
    def nlist(self) -> int:
        return self.num_triplet_classes
    
    @property
    def nequi(self) -> NDArray[np.int64]:
        return self.equivalent_count
    
    @property
    def allequilist(self) -> NDArray[np.int64]:
        return self.equivalent_triplets
    
    @property
    def nindependentbasis(self) -> NDArray[np.int64]:
        return self.independent_basis_count
    
    @property
    def independentbasis(self) -> NDArray[np.int64]:
        return self.independent_basis
    
    @property
    def llist(self) -> NDArray[np.int64]:
        return self.triplet_list
    
    @property
    def transformationarray(self) -> NDArray[np.float64]:
        return self.transformation_array
    
