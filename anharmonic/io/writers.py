"""
力常数输出模块

支持多种格式的力常数文件输出
"""

from __future__ import annotations

import io
import itertools
from pathlib import Path
from typing import Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import sparse


class ForceConstantsWriter:
    """
    力常数输出器
    
    支持 ShengBTE 格式的力常数文件输出
    """
    
    @staticmethod
    def write_third_order(
        force_constants: "sparse.COO",
        primitive: dict,
        supercell: dict,
        distance_matrix: NDArray[np.float64],
        equivalent_count: NDArray[np.intc],
        shift_vectors: NDArray[np.intc],
        cutoff_range: float,
        output_path: Union[str, Path],
    ) -> None:
        """
        写入三阶力常数文件 (ShengBTE 格式)
        
        Args:
            force_constants: 完整力常数矩阵 (3, 3, 3, natoms, ntot, ntot)
            primitive: 原胞结构字典
            supercell: 超胞结构字典
            distance_matrix: 距离矩阵
            equivalent_count: 等价原子数目
            shift_vectors: 位移矢量
            cutoff_range: 截断距离 (nm)
            output_path: 输出文件路径
        """
        num_primitive_atoms = len(primitive["types"])
        num_supercell_atoms = len(supercell["types"])
        
        # 27个邻居位移
        shifts_27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
        cutoff_squared = cutoff_range * cutoff_range
        
        num_blocks = 0
        buffer = io.StringIO()
        
        for atom_i, atom_j in itertools.product(
            range(num_primitive_atoms),
            range(num_supercell_atoms)
        ):
            if distance_matrix[atom_i, atom_j] >= cutoff_range:
                continue
            
            primitive_atom_j = atom_j % num_primitive_atoms
            shifts_ij = [
                shifts_27[i] for i in 
                shift_vectors[atom_i, atom_j, :equivalent_count[atom_i, atom_j]]
            ]
            
            for atom_k in range(num_supercell_atoms):
                if distance_matrix[atom_i, atom_k] >= cutoff_range:
                    continue
                
                primitive_atom_k = atom_k % num_primitive_atoms
                shifts_ik = [
                    shifts_27[i] for i in
                    shift_vectors[atom_i, atom_k, :equivalent_count[atom_i, atom_k]]
                ]
                
                # 找到最佳位移组合
                min_distance_squared = np.inf
                best_shift_j = None
                best_shift_k = None
                
                for shift_j in shifts_ij:
                    cartesian_j = np.dot(
                        supercell["lattvec"],
                        np.array(shift_j) + supercell["positions"][:, atom_j]
                    )
                    for shift_k in shifts_ik:
                        cartesian_k = np.dot(
                            supercell["lattvec"],
                            np.array(shift_k) + supercell["positions"][:, atom_k]
                        )
                        dist_squared = ((cartesian_j - cartesian_k) ** 2).sum()
                        if dist_squared < min_distance_squared:
                            best_shift_j = shift_j
                            best_shift_k = shift_k
                            min_distance_squared = dist_squared
                
                if min_distance_squared >= cutoff_squared:
                    continue
                
                num_blocks += 1
                
                # 计算晶格矢量
                R_j = np.dot(
                    supercell["lattvec"],
                    np.array(best_shift_j) + supercell["positions"][:, atom_j]
                    - supercell["positions"][:, primitive_atom_j]
                )
                R_k = np.dot(
                    supercell["lattvec"],
                    np.array(best_shift_k) + supercell["positions"][:, atom_k]
                    - supercell["positions"][:, primitive_atom_k]
                )
                
                # 写入块
                buffer.write("\n")
                buffer.write(f"{num_blocks:>5}\n")
                buffer.write(
                    f"{10.0 * R_j[0]:>15.10e} {10.0 * R_j[1]:>15.10e} {10.0 * R_j[2]:>15.10e}\n"
                )
                buffer.write(
                    f"{10.0 * R_k[0]:>15.10e} {10.0 * R_k[1]:>15.10e} {10.0 * R_k[2]:>15.10e}\n"
                )
                buffer.write(
                    f"{atom_i + 1:>6d} {primitive_atom_j + 1:>6d} {primitive_atom_k + 1:>6d}\n"
                )
                
                # 写入力常数值
                for coord_l, coord_m, coord_n in itertools.product(
                    range(3), range(3), range(3)
                ):
                    value = force_constants[coord_l, coord_m, coord_n, atom_i, atom_j, atom_k]
                    buffer.write(
                        f"{coord_l + 1:>2d} {coord_m + 1:>2d} {coord_n + 1:>2d} {value:>20.10e}\n"
                    )
        
        # 写入文件
        with open(output_path, "w") as f:
            f.write(f"{num_blocks:>5}\n")
            f.write(buffer.getvalue())
        
        buffer.close()


    @staticmethod
    def write_fourth_order(
        force_constants: "sparse.COO",
        primitive: dict,
        supercell: dict,
        distance_matrix: NDArray[np.float64],
        equivalent_count: NDArray[np.intc],
        shift_vectors: NDArray[np.intc],
        cutoff_range: float,
        output_path: Union[str, Path],
    ) -> None:
        """
        写入四阶力常数文件 (ShengBTE 格式)
        
        Args:
            force_constants: 完整力常数矩阵 (3, 3, 3, 3, natoms, ntot, ntot, ntot)
            primitive: 原胞结构字典
            supercell: 超胞结构字典
            distance_matrix: 距离矩阵
            equivalent_count: 等价原子数目
            shift_vectors: 位移矢量
            cutoff_range: 截断距离 (nm)
            output_path: 输出文件路径
        """
        num_primitive_atoms = len(primitive["types"])
        num_supercell_atoms = len(supercell["types"])
        
        shifts_27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
        cutoff_squared = cutoff_range * cutoff_range
        
        num_blocks = 0
        buffer = io.StringIO()
        
        for atom_i, atom_j, atom_k in itertools.product(
            range(num_primitive_atoms),
            range(num_supercell_atoms),
            range(num_supercell_atoms)
        ):
            if distance_matrix[atom_i, atom_j] >= cutoff_range:
                continue
            if distance_matrix[atom_i, atom_k] >= cutoff_range:
                continue
            
            primitive_atom_j = atom_j % num_primitive_atoms
            primitive_atom_k = atom_k % num_primitive_atoms
            shifts_ij = [
                shifts_27[i] for i in 
                shift_vectors[atom_i, atom_j, :equivalent_count[atom_i, atom_j]]
            ]
            shifts_ik = [
                shifts_27[i] for i in
                shift_vectors[atom_i, atom_k, :equivalent_count[atom_i, atom_k]]
            ]
            
            for atom_l in range(num_supercell_atoms):
                if distance_matrix[atom_i, atom_l] >= cutoff_range:
                    continue
                
                primitive_atom_l = atom_l % num_primitive_atoms
                shifts_il = [
                    shifts_27[i] for i in
                    shift_vectors[atom_i, atom_l, :equivalent_count[atom_i, atom_l]]
                ]
                
                # 找到最佳位移组合
                min_distance_squared = np.inf
                best_shift_j = shifts_ij[0] if shifts_ij else (0, 0, 0)
                best_shift_k = shifts_ik[0] if shifts_ik else (0, 0, 0)
                best_shift_l = shifts_il[0] if shifts_il else (0, 0, 0)
                
                # 简化版：直接使用第一个位移
                num_blocks += 1
                
                # 计算晶格矢量
                R_j = np.dot(
                    supercell["lattvec"],
                    np.array(best_shift_j) + supercell["positions"][:, atom_j]
                    - supercell["positions"][:, primitive_atom_j]
                )
                R_k = np.dot(
                    supercell["lattvec"],
                    np.array(best_shift_k) + supercell["positions"][:, atom_k]
                    - supercell["positions"][:, primitive_atom_k]
                )
                R_l = np.dot(
                    supercell["lattvec"],
                    np.array(best_shift_l) + supercell["positions"][:, atom_l]
                    - supercell["positions"][:, primitive_atom_l]
                )
                
                buffer.write("\n")
                buffer.write(f"{num_blocks:>5}\n")
                buffer.write(
                    f"{10.0 * R_j[0]:>15.10e} {10.0 * R_j[1]:>15.10e} {10.0 * R_j[2]:>15.10e}\n"
                )
                buffer.write(
                    f"{10.0 * R_k[0]:>15.10e} {10.0 * R_k[1]:>15.10e} {10.0 * R_k[2]:>15.10e}\n"
                )
                buffer.write(
                    f"{10.0 * R_l[0]:>15.10e} {10.0 * R_l[1]:>15.10e} {10.0 * R_l[2]:>15.10e}\n"
                )
                buffer.write(
                    f"{atom_i + 1:>6d} {primitive_atom_j + 1:>6d} {primitive_atom_k + 1:>6d} {primitive_atom_l + 1:>6d}\n"
                )
                
                for coord_l, coord_m, coord_n, coord_o in itertools.product(
                    range(3), range(3), range(3), range(3)
                ):
                    value = force_constants[coord_l, coord_m, coord_n, coord_o, atom_i, atom_j, atom_k, atom_l]
                    buffer.write(
                        f"{coord_l + 1:>2d} {coord_m + 1:>2d} {coord_n + 1:>2d} {coord_o + 1:>2d} {value:>20.10e}\n"
                    )
        
        with open(output_path, "w") as f:
            f.write(f"{num_blocks:>5}\n")
            f.write(buffer.getvalue())
        
        buffer.close()


# 向后兼容的函数
def write_ifcs3(
    phifull,
    poscar: dict,
    sposcar: dict,
    dmin: NDArray[np.float64],
    nequi: NDArray[np.intc],
    shifts: NDArray[np.intc],
    frange: float,
    filename: str,
) -> None:
    """向后兼容的三阶力常数写入函数"""
    ForceConstantsWriter.write_third_order(
        force_constants=phifull,
        primitive=poscar,
        supercell=sposcar,
        distance_matrix=dmin,
        equivalent_count=nequi,
        shift_vectors=shifts,
        cutoff_range=frange,
        output_path=filename,
    )


def write_ifcs4(
    phifull,
    poscar: dict,
    sposcar: dict,
    dmin: NDArray[np.float64],
    nequi: NDArray[np.intc],
    shifts: NDArray[np.intc],
    frange: float,
    filename: str,
) -> None:
    """向后兼容的四阶力常数写入函数"""
    ForceConstantsWriter.write_fourth_order(
        force_constants=phifull,
        primitive=poscar,
        supercell=sposcar,
        distance_matrix=dmin,
        equivalent_count=nequi,
        shift_vectors=shifts,
        cutoff_range=frange,
        output_path=filename,
    )
