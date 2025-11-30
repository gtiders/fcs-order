"""
四阶 IFC 重构模块

从不可约力常数重构完整的四阶力常数矩阵
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sparse
from numpy.typing import NDArray
from rich.progress import track

from anharmonic.core.jit_kernels4 import (
    jit_build_sparse_ifc4_indices,
    jit_build_full_ifc4_coordinates,
)

if TYPE_CHECKING:
    from anharmonic.core.wedge4 import QuartetWedge


class IFC4Reconstructor:
    """
    四阶力常数重构器
    
    从不可约集合的力常数值重构完整的四阶力常数矩阵，
    并应用声子平移规则确保动量守恒。
    """
    
    @staticmethod
    def reconstruct(
        partial_force_constants: NDArray[np.float64],
        wedge: QuartetWedge,
        irreducible_displacements: list[tuple[int, int, int, int, int, int]],
        primitive: dict,
        supercell: dict,
    ) -> sparse.COO:
        """
        从部分力常数重构完整的四阶力常数矩阵
        
        Args:
            partial_force_constants: 部分力常数 (3, nirred, ntot)
            wedge: QuartetWedge 约简器对象
            irreducible_displacements: 不可约位移列表
            primitive: 原胞结构字典
            supercell: 超胞结构字典
            
        Returns:
            完整的四阶力常数矩阵 (3, 3, 3, 3, natoms, ntot, ntot, ntot)
        """
        num_quartet_classes = wedge.num_quartet_classes
        num_primitive_atoms = len(primitive["types"])
        num_supercell_atoms = len(supercell["types"])
        
        # 构建初始稀疏矩阵
        initial_ifc = sparse.zeros(
            (3, 3, 3, 3, num_primitive_atoms, num_supercell_atoms, num_supercell_atoms, num_supercell_atoms),
            format="dok"
        )
        
        # 累积独立基数量
        accumulated_independent = np.insert(
            np.cumsum(wedge.independent_basis_count[:num_quartet_classes], dtype=np.intc),
            0,
            np.zeros(1, dtype=np.intc),
        )
        
        # 填充初始力常数
        num_displacements = len(irreducible_displacements)
        for disp_idx in track(range(num_displacements), description="处理位移"):
            atom_k, atom_j, atom_i, coord_n, coord_m, coord_l = irreducible_displacements[disp_idx]
            initial_ifc[coord_l, coord_m, coord_n, :, atom_i, atom_j, atom_k, :] = partial_force_constants[:, disp_idx, :]
        
        initial_ifc = initial_ifc.to_coo()
        
        # 提取独立力常数值
        force_constant_values = []
        for class_idx in track(range(num_quartet_classes), description="提取力常数"):
            for basis_idx in range(wedge.independent_basis_count[class_idx]):
                basis_value = wedge.independent_basis[basis_idx, class_idx]
                coord_l = basis_value // 27
                coord_m = (basis_value % 27) // 9
                coord_n = (basis_value % 9) // 3
                coord_o = basis_value % 3
                
                force_constant_values.append(
                    initial_ifc[
                        coord_l,
                        coord_m,
                        coord_n,
                        coord_o,
                        wedge.quartet_list[0, class_idx],
                        wedge.quartet_list[1, class_idx],
                        wedge.quartet_list[2, class_idx],
                        wedge.quartet_list[3, class_idx],
                    ]
                )
        
        force_constant_array = np.array(force_constant_values, dtype=np.float64)
        
        # 构建完整力常数矩阵
        (
            coord_0_list, coord_1_list, coord_2_list, coord_3_list,
            coord_4_list, coord_5_list, coord_6_list, coord_7_list,
            values_list,
        ) = jit_build_full_ifc4_coordinates(
            num_quartet_classes,
            wedge.equivalent_count,
            wedge.independent_basis_count,
            wedge.equivalent_quartets,
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
            np.array(coord_6_list, dtype=np.intp),
            np.array(coord_7_list, dtype=np.intp),
        ], dtype=np.intp)
        
        full_ifc = sparse.COO(
            coordinates,
            np.array(values_list, dtype=np.float64),
            shape=(3, 3, 3, 3, num_primitive_atoms, num_supercell_atoms, num_supercell_atoms, num_supercell_atoms),
            has_duplicates=True,
        )
        
        return full_ifc


# 向后兼容的函数
def reconstruct_ifcs4(
    phipart: NDArray[np.float64],
    wedge: QuartetWedge,
    list6: list[tuple[int, int, int, int, int, int]],
    poscar: dict,
    sposcar: dict,
) -> sparse.COO:
    """向后兼容的重构函数"""
    return IFC4Reconstructor.reconstruct(
        partial_force_constants=phipart,
        wedge=wedge,
        irreducible_displacements=list6,
        primitive=poscar,
        supercell=sposcar,
    )
