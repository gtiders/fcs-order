"""
工具函数模块

提供距离计算、原子位移、截断范围计算等辅助功能
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance
from numpy.typing import NDArray
import typer

if TYPE_CHECKING:
    from anharmonic.models.structure import CrystalStructure, SupercellStructure


def calculate_distances(
    supercell: SupercellStructure,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    计算超胞中所有原子对之间的距离
    
    Args:
        supercell: 超胞结构
        
    Returns:
        distance_matrix: 最小距离矩阵 (ntot, ntot)
        equivalent_count: 等价原子数目 (ntot, ntot)
        shift_vectors: 位移矢量索引 (ntot, ntot, max_equiv)
    """
    num_atoms = supercell.positions.shape[1]
    cartesian_positions = np.dot(supercell.lattice_vectors, supercell.positions)
    
    # 计算27个邻居镜像的距离
    distance_squared = np.empty((27, num_atoms, num_atoms))
    
    for idx, (shift_a, shift_b, shift_c) in enumerate(
        itertools.product(range(-1, 2), range(-1, 2), range(-1, 2))
    ):
        shifted_positions = np.dot(
            supercell.lattice_vectors,
            (supercell.positions.T + [shift_a, shift_b, shift_c]).T
        )
        distance_squared[idx, :, :] = scipy.spatial.distance.cdist(
            cartesian_positions.T,
            shifted_positions.T,
            "sqeuclidean"
        )
    
    # 找到最小距离
    min_distance_squared = distance_squared.min(axis=0)
    distance_matrix = np.sqrt(min_distance_squared)
    
    # 找到等价的镜像（距离相同）
    is_equivalent = np.abs(distance_squared - min_distance_squared) < 1e-4
    equivalent_count = is_equivalent.sum(axis=0, dtype=np.int64)
    max_equivalent = equivalent_count.max()
    
    # 构建位移矢量索引
    sorting = np.argsort(np.logical_not(is_equivalent), axis=0)
    shift_vectors = np.transpose(
        sorting[:max_equivalent, :, :],
        (1, 2, 0)
    ).astype(np.int64)
    
    return distance_matrix, equivalent_count, shift_vectors


def calculate_cutoff_range(
    primitive: CrystalStructure,
    supercell: SupercellStructure,
    neighbor_order: int,
    distance_matrix: NDArray[np.float64],
) -> float:
    """
    计算第 n 近邻的截断距离
    
    Args:
        primitive: 原胞结构
        supercell: 超胞结构
        neighbor_order: 近邻阶数
        distance_matrix: 距离矩阵
        
    Returns:
        截断距离 (nm)
    """
    num_primitive_atoms = primitive.num_atoms
    cutoff_distances = []
    warned = False
    
    for atom_idx in range(num_primitive_atoms):
        distances = sorted(distance_matrix[atom_idx, :].tolist())
        
        # 去除重复距离
        unique_distances = []
        for dist in distances:
            is_duplicate = False
            for existing in unique_distances:
                if np.allclose(existing, dist):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_distances.append(dist)
        
        try:
            # 取第 n 和 n+1 近邻距离的平均值
            cutoff_distances.append(
                0.5 * (unique_distances[neighbor_order] + unique_distances[neighbor_order + 1])
            )
        except IndexError:
            if not warned:
                typer.secho(
                    "警告: 超胞太小，无法找到第 n 近邻",
                    fg=typer.colors.RED,
                    err=True,
                )
                warned = True
            cutoff_distances.append(1.1 * max(unique_distances))
    
    return max(cutoff_distances)


def displace_two_atoms(
    structure: SupercellStructure,
    atom_i: int,
    coord_i: int,
    displacement_i: float,
    atom_j: int,
    coord_j: int,
    displacement_j: float,
) -> SupercellStructure:
    """
    对结构中的两个原子进行位移
    
    Args:
        structure: 超胞结构
        atom_i: 第一个原子索引
        coord_i: 第一个原子的位移坐标方向 (0, 1, 2)
        displacement_i: 第一个原子的位移量 (nm)
        atom_j: 第二个原子索引
        coord_j: 第二个原子的位移坐标方向 (0, 1, 2)
        displacement_j: 第二个原子的位移量 (nm)
        
    Returns:
        位移后的超胞结构
    """
    from anharmonic.models.structure import SupercellStructure
    
    new_positions = structure.positions.copy()
    
    # 位移第一个原子
    displacement = np.zeros(3)
    displacement[coord_i] = displacement_i
    new_positions[:, atom_i] += scipy.linalg.solve(
        structure.lattice_vectors,
        displacement
    )
    
    # 位移第二个原子
    displacement[:] = 0.0
    displacement[coord_j] = displacement_j
    new_positions[:, atom_j] += scipy.linalg.solve(
        structure.lattice_vectors,
        displacement
    )
    
    return SupercellStructure(
        lattice_vectors=structure.lattice_vectors.copy(),
        positions=new_positions,
        elements=structure.elements.copy(),
        atom_counts=structure.atom_counts.copy(),
        atom_types=structure.atom_types.copy(),
        grid_size=structure.grid_size,
        primitive=structure.primitive,
    )


def generate_supercell(
    primitive: CrystalStructure,
    na: int,
    nb: int,
    nc: int,
) -> SupercellStructure:
    """
    生成超胞结构
    
    Args:
        primitive: 原胞结构
        na, nb, nc: 三个方向的扩展倍数
        
    Returns:
        超胞结构
    """
    from anharmonic.models.structure import SupercellStructure
    
    # 扩展晶格矢量
    lattice_vectors = primitive.lattice_vectors.copy()
    lattice_vectors[:, 0] *= na
    lattice_vectors[:, 1] *= nb
    lattice_vectors[:, 2] *= nc
    
    atom_counts = na * nb * nc * primitive.atom_counts
    
    num_primitive_atoms = primitive.num_atoms
    num_supercell_atoms = num_primitive_atoms * na * nb * nc
    positions = np.empty((3, num_supercell_atoms))
    
    # 生成超胞原子位置
    for pos_idx, (ic, ib, ia, atom_idx) in enumerate(
        itertools.product(
            range(nc), range(nb), range(na), range(num_primitive_atoms)
        )
    ):
        positions[:, pos_idx] = (
            primitive.positions[:, atom_idx] + [ia, ib, ic]
        ) / [na, nb, nc]
    
    # 生成原子类型列表
    atom_types: list[int] = []
    for _ in range(na * nb * nc):
        atom_types.extend(primitive.atom_types)
    
    return SupercellStructure(
        lattice_vectors=lattice_vectors,
        positions=positions,
        elements=primitive.elements.copy(),
        atom_counts=atom_counts,
        atom_types=atom_types,
        grid_size=(na, nb, nc),
        primitive=primitive,
    )


def parse_cutoff(cutoff_str: str) -> tuple[int | None, float | None]:
    """
    解析截断参数
    
    Args:
        cutoff_str: 截断参数字符串
            - 负整数: 表示第 n 近邻 (如 "-3" 表示第3近邻)
            - 正数: 表示截断距离 (nm)
            
    Returns:
        (neighbor_order, cutoff_range)
        - 如果是近邻数: (n, None)
        - 如果是距离: (None, distance)
    """
    if cutoff_str.startswith("-"):
        try:
            neighbor_order = -int(cutoff_str)
        except ValueError:
            raise ValueError(f"无效的截断参数: {cutoff_str}")
        if neighbor_order == 0:
            raise ValueError("近邻阶数不能为0")
        return neighbor_order, None
    else:
        try:
            cutoff_range = float(cutoff_str)
        except ValueError:
            raise ValueError(f"无效的截断参数: {cutoff_str}")
        if cutoff_range == 0.0:
            raise ValueError("截断距离不能为0")
        return None, cutoff_range


def validate_supercell_size(na: int, nb: int, nc: int) -> None:
    """验证超胞尺寸"""
    if min(na, nb, nc) < 1:
        raise ValueError("超胞尺寸 (na, nb, nc) 必须是正整数")
