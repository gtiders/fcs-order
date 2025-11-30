"""
IFC 重构模块

从不可约力常数重构完整的三阶力常数矩阵
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sparse
from numpy.typing import NDArray
from rich.progress import track

from anharmonic.core.jit_kernels import (
    jit_build_sparse_ifc_indices,
    jit_build_full_ifc_coordinates,
)

if TYPE_CHECKING:
    from anharmonic.core.wedge import TripletWedge


class IFCReconstructor:
    """
    三阶力常数重构器
    
    从不可约集合的力常数值重构完整的三阶力常数矩阵，
    并应用声子平移规则确保动量守恒。
    """
    
    @staticmethod
    def reconstruct(
        partial_force_constants: NDArray[np.float64],
        wedge: TripletWedge,
        irreducible_displacements: list[tuple[int, int, int, int]],
        primitive: dict,
        supercell: dict,
    ) -> sparse.COO:
        """
        从部分力常数重构完整的三阶力常数矩阵
        
        Args:
            partial_force_constants: 部分力常数 (3, nirred, ntot)
            wedge: Wedge 约简器对象
            irreducible_displacements: 不可约位移列表
            primitive: 原胞结构字典
            supercell: 超胞结构字典
            
        Returns:
            完整的三阶力常数矩阵 (3, 3, 3, natoms, ntot, ntot)
        """
        num_triplet_classes = wedge.num_triplet_classes
        num_primitive_atoms = len(primitive["types"])
        num_supercell_atoms = len(supercell["types"])
        
        # 构建初始稀疏矩阵
        initial_ifc = sparse.zeros(
            (3, 3, 3, num_primitive_atoms, num_supercell_atoms, num_supercell_atoms),
            format="dok"
        )
        
        # 累积独立基数量
        accumulated_independent = np.insert(
            np.cumsum(wedge.independent_basis_count[:num_triplet_classes], dtype=np.intc),
            0,
            np.zeros(1, dtype=np.intc),
        )
        total_independent = accumulated_independent[-1]
        
        # 填充初始力常数
        num_displacements = len(irreducible_displacements)
        for disp_idx in track(range(num_displacements), description="处理位移"):
            atom_i, atom_j, coord_l, coord_m = irreducible_displacements[disp_idx]
            initial_ifc[coord_l, coord_m, :, atom_i, atom_j, :] = partial_force_constants[:, disp_idx, :]
        
        initial_ifc = initial_ifc.to_coo()
        
        # 提取独立力常数值
        force_constant_values = []
        for class_idx in track(range(num_triplet_classes), description="提取力常数"):
            for basis_idx in range(wedge.independent_basis_count[class_idx]):
                basis_value = wedge.independent_basis[basis_idx, class_idx]
                coord_l = basis_value // 9
                coord_m = (basis_value % 9) // 3
                coord_n = basis_value % 3
                
                force_constant_values.append(
                    initial_ifc[
                        coord_l,
                        coord_m,
                        coord_n,
                        wedge.triplet_list[0, class_idx],
                        wedge.triplet_list[1, class_idx],
                        wedge.triplet_list[2, class_idx],
                    ]
                )
        
        force_constant_array = np.array(force_constant_values, dtype=np.float64)
        
        # 构建三元组类索引和等价索引数组
        triplet_class_indices = -np.ones(
            (num_primitive_atoms, num_supercell_atoms, num_supercell_atoms),
            dtype=np.intc
        )
        equivalent_indices = -np.ones(
            (num_primitive_atoms, num_supercell_atoms, num_supercell_atoms),
            dtype=np.intc
        )
        
        for class_idx in range(num_triplet_classes):
            for equiv_idx in range(wedge.equivalent_count[class_idx]):
                atom_i = wedge.equivalent_triplets[0, equiv_idx, class_idx]
                atom_j = wedge.equivalent_triplets[1, equiv_idx, class_idx]
                atom_k = wedge.equivalent_triplets[2, equiv_idx, class_idx]
                triplet_class_indices[atom_i, atom_j, atom_k] = class_idx
                equivalent_indices[atom_i, atom_j, atom_k] = equiv_idx
        
        # 构建稀疏系数矩阵
        num_rows = total_independent
        num_cols = num_primitive_atoms * num_supercell_atoms * 27
        
        row_indices, col_indices, values = jit_build_sparse_ifc_indices(
            triplet_class_indices,
            equivalent_indices,
            accumulated_independent,
            num_primitive_atoms,
            num_supercell_atoms,
            num_triplet_classes,
            wedge.transformation_array,
        )
        
        coefficient_matrix = scipy.sparse.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_rows, num_cols)
        ).tocsr()
        
        # 应用声子平移规则
        force_constant_array = IFCReconstructor._apply_acoustic_sum_rule(
            force_constant_array,
            coefficient_matrix,
        )
        
        # 构建完整力常数矩阵
        (
            coord_0_list,
            coord_1_list,
            coord_2_list,
            coord_3_list,
            coord_4_list,
            coord_5_list,
            values_list,
        ) = jit_build_full_ifc_coordinates(
            num_triplet_classes,
            wedge.equivalent_count,
            wedge.independent_basis_count,
            wedge.equivalent_triplets,
            accumulated_independent,
            wedge.transformation_array,
            force_constant_array,
        )
        
        coordinates = np.array([
            np.array(coord_0_list, dtype=np.intp),
            np.array(coord_1_list, dtype=np.intp),
            np.array(coord_2_list, dtype=np.intp),
            np.array(coord_3_list, dtype=np.intp),
            np.array(coord_4_list, dtype=np.intp),
            np.array(coord_5_list, dtype=np.intp),
        ], dtype=np.intp)
        
        full_ifc = sparse.COO(
            coordinates,
            np.array(values_list, dtype=np.float64),
            shape=(3, 3, 3, num_primitive_atoms, num_supercell_atoms, num_supercell_atoms),
            has_duplicates=True,
        )
        
        return full_ifc
    
    @staticmethod
    def _apply_acoustic_sum_rule(
        force_constants: NDArray[np.float64],
        coefficient_matrix: scipy.sparse.csr_matrix,
    ) -> NDArray[np.float64]:
        """
        应用声子平移规则（Acoustic Sum Rule）
        
        确保力常数满足动量守恒条件。
        
        Args:
            force_constants: 力常数值数组
            coefficient_matrix: 稀疏系数矩阵
            
        Returns:
            修正后的力常数数组
        """
        # 构建对角矩阵
        diagonal_matrix = scipy.sparse.spdiags(
            force_constants,
            [0],
            force_constants.size,
            force_constants.size,
            format="csr"
        )
        
        # 计算补偿
        scaled_matrix = diagonal_matrix.dot(coefficient_matrix)
        ones = np.ones_like(force_constants)
        multiplier = -scipy.sparse.linalg.lsqr(scaled_matrix, ones)[0]
        compensation = diagonal_matrix.dot(scaled_matrix.dot(multiplier))
        
        return force_constants + compensation


# 向后兼容的函数
def reconstruct_ifcs(
    phipart: NDArray[np.float64],
    wedge: TripletWedge,
    list4: list[tuple[int, int, int, int]],
    poscar: dict,
    sposcar: dict,
) -> sparse.COO:
    """
    向后兼容的重构函数
    
    从不可约力常数重构完整的三阶力常数矩阵。
    
    Args:
        phipart: 部分力常数 (3, nirred, ntot)
        wedge: Wedge 约简器对象
        list4: 不可约位移列表
        poscar: 原胞结构字典
        sposcar: 超胞结构字典
        
    Returns:
        完整的三阶力常数矩阵
    """
    return IFCReconstructor.reconstruct(
        partial_force_constants=phipart,
        wedge=wedge,
        irreducible_displacements=list4,
        primitive=poscar,
        supercell=sposcar,
    )
