"""
JIT 编译内核函数

集中管理所有 numba JIT 编译的高性能计算函数
命名规范: jit_<功能描述>
"""

from numba import jit
from numba.typed import List
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# 索引转换函数
# ============================================================================

@jit(nopython=True)
def jit_cell_atom_to_supercell_index(
    cell_indices: NDArray[np.int64],
    atom_index: int,
    grid_size: NDArray[np.int64],
    num_primitive_atoms: int,
) -> int:
    """
    将晶胞索引和原子索引转换为超胞中的原子索引
    
    Args:
        cell_indices: 晶胞索引 (3,) [ia, ib, ic]
        atom_index: 原胞中的原子索引
        grid_size: 超胞网格大小 (3,) [na, nb, nc]
        num_primitive_atoms: 原胞中的原子数
        
    Returns:
        超胞中的原子索引
    """
    return (
        cell_indices[0] 
        + (cell_indices[1] + cell_indices[2] * grid_size[1]) * grid_size[0]
    ) * num_primitive_atoms + atom_index


@jit(nopython=True)
def jit_supercell_index_to_cell_atom(
    grid_size: NDArray[np.int64],
    num_primitive_atoms: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    创建从超胞原子索引到晶胞索引和原子索引的映射
    
    Args:
        grid_size: 超胞网格大小 (3,) [na, nb, nc]
        num_primitive_atoms: 原胞中的原子数
        
    Returns:
        cell_indices: 晶胞索引数组 (3, ntot)
        atom_indices: 原子索引数组 (ntot,)
    """
    num_supercell_atoms = grid_size[0] * grid_size[1] * grid_size[2] * num_primitive_atoms
    cell_indices = np.empty((3, num_supercell_atoms), dtype=np.int64)
    atom_indices = np.empty(num_supercell_atoms, dtype=np.int64)
    
    for idx in range(num_supercell_atoms):
        remainder, atom_indices[idx] = divmod(idx, num_primitive_atoms)
        remainder, cell_indices[0, idx] = divmod(remainder, grid_size[0])
        cell_indices[2, idx], cell_indices[1, idx] = divmod(remainder, grid_size[1])
    
    return cell_indices, atom_indices


# ============================================================================
# 三元组操作函数
# ============================================================================

@jit(nopython=True)
def jit_triplet_in_list(
    triplet: NDArray[np.int64],
    triplet_list: NDArray[np.int64],
    list_length: int,
) -> bool:
    """
    检查三元组是否在列表中
    
    Args:
        triplet: 要检查的三元组 (3,)
        triplet_list: 三元组列表 (3, N)
        list_length: 列表中有效三元组的数量
        
    Returns:
        如果找到则返回 True
    """
    for i in range(list_length):
        if (
            triplet[0] == triplet_list[0, i]
            and triplet[1] == triplet_list[1, i]
            and triplet[2] == triplet_list[2, i]
        ):
            return True
    return False


@jit(nopython=True)
def jit_triplets_equal(
    triplet1: NDArray[np.int64],
    triplet2: NDArray[np.int64],
) -> bool:
    """
    比较两个三元组是否相等
    
    Args:
        triplet1: 第一个三元组 (3,)
        triplet2: 第二个三元组 (3,)
        
    Returns:
        如果相等则返回 True
    """
    for i in range(3):
        if triplet1[i] != triplet2[i]:
            return False
    return True


# ============================================================================
# 变换数组构建函数
# ============================================================================

@jit(nopython=True)
def jit_build_transformation_array(
    transformation: NDArray[np.float64],
    transformation_auxiliary: NDArray[np.float64],
    equivalent_count: NDArray[np.int64],
    independent_basis_count: NDArray[np.int64],
    num_triplet_classes: int,
    output_array: NDArray[np.float64],
) -> None:
    """
    构建三阶力常数的变换数组
    
    计算:
        output[k, l, j, i] = sum_{aux} transformation[k, aux, j, i] 
                                     * transformation_auxiliary[aux, l, i]
    
    Args:
        transformation: 变换矩阵 (27, 27, max_equiv, nlist)
        transformation_auxiliary: 辅助变换矩阵 (27, 27, nlist)
        equivalent_count: 每个三元组类的等价数量 (nlist,)
        independent_basis_count: 每个三元组类的独立基数量 (nlist,)
        num_triplet_classes: 三元组类的数量
        output_array: 输出数组 (27, 27, max_equiv, nlist)
    """
    for triplet_idx in range(num_triplet_classes):
        num_equiv = equivalent_count[triplet_idx]
        num_independent = independent_basis_count[triplet_idx]
        
        for equiv_idx in range(num_equiv):
            for basis_prime in range(27):
                for independent_idx in range(num_independent):
                    value = 0.0
                    for aux_idx in range(27):
                        value += (
                            transformation[basis_prime, aux_idx, equiv_idx, triplet_idx]
                            * transformation_auxiliary[aux_idx, independent_idx, triplet_idx]
                        )
                    # 过滤掉极小值
                    if value != 0.0 and abs(value) < 1e-12:
                        value = 0.0
                    output_array[basis_prime, independent_idx, equiv_idx, triplet_idx] = value


# ============================================================================
# IFC 重构函数
# ============================================================================

@jit(nopython=True)
def jit_build_sparse_ifc_indices(
    triplet_class_indices: NDArray[np.int64],
    equivalent_indices: NDArray[np.int64],
    accumulated_independent: NDArray[np.int64],
    num_primitive_atoms: int,
    num_supercell_atoms: int,
    num_triplet_classes: int,
    transformation_array: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    构建稀疏三阶力常数矩阵的 COO 格式索引和值
    
    Args:
        triplet_class_indices: 三元组类索引 (natoms, ntot, ntot)
        equivalent_indices: 等价索引 (natoms, ntot, ntot)
        accumulated_independent: 累积独立基数量 (nlist + 1,)
        num_primitive_atoms: 原胞原子数
        num_supercell_atoms: 超胞原子数
        num_triplet_classes: 三元组类数量
        transformation_array: 变换数组 (27, 27, max_equiv, nlist)
        
    Returns:
        row_indices: 行索引
        col_indices: 列索引
        values: 非零值
    """
    # 第一遍：计算总元素数
    total_count = 0
    for atom_i in range(num_primitive_atoms):
        for atom_j in range(num_supercell_atoms):
            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        for atom_k in range(num_supercell_atoms):
                            for class_idx in range(num_triplet_classes):
                                if triplet_class_indices[atom_i, atom_j, atom_k] == class_idx:
                                    total_count += (
                                        accumulated_independent[class_idx + 1]
                                        - accumulated_independent[class_idx]
                                    )
    
    # 分配数组
    row_indices = np.empty(total_count, dtype=np.int64)
    col_indices = np.empty(total_count, dtype=np.int64)
    values = np.empty(total_count, dtype=np.float64)
    
    # 第二遍：填充数组
    count = 0
    col_index = 0
    
    for atom_i in range(num_primitive_atoms):
        for atom_j in range(num_supercell_atoms):
            tri_basis_index = 0
            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        for atom_k in range(num_supercell_atoms):
                            for class_idx in range(num_triplet_classes):
                                if triplet_class_indices[atom_i, atom_j, atom_k] == class_idx:
                                    start = accumulated_independent[class_idx]
                                    end = accumulated_independent[class_idx + 1]
                                    for s in range(start, end):
                                        independent_idx = s - start
                                        row_indices[count] = s
                                        col_indices[count] = col_index
                                        values[count] = transformation_array[
                                            tri_basis_index,
                                            independent_idx,
                                            equivalent_indices[atom_i, atom_j, atom_k],
                                            class_idx,
                                        ]
                                        count += 1
                        tri_basis_index += 1
                        col_index += 1
    
    return row_indices, col_indices, values


@jit(nopython=True)
def jit_build_full_ifc_coordinates(
    num_triplet_classes: int,
    equivalent_count: NDArray[np.int64],
    independent_basis_count: NDArray[np.int64],
    equivalent_triplets: NDArray[np.int64],
    accumulated_independent: NDArray[np.int64],
    transformation_array: NDArray[np.float64],
    force_constant_values: NDArray[np.float64],
) -> tuple:
    """
    构建完整三阶力常数的坐标和值列表
    
    Args:
        num_triplet_classes: 三元组类数量
        equivalent_count: 等价数量数组 (nlist,)
        independent_basis_count: 独立基数量数组 (nlist,)
        equivalent_triplets: 等价三元组数组 (3, max_equiv, nlist)
        accumulated_independent: 累积独立基数量 (nlist + 1,)
        transformation_array: 变换数组 (27, 27, max_equiv, nlist)
        force_constant_values: 力常数值数组 (total_independent,)
        
    Returns:
        六个坐标列表和一个值列表
    """
    coord_0 = List()  # l
    coord_1 = List()  # m
    coord_2 = List()  # n
    coord_3 = List()  # atom_i
    coord_4 = List()  # atom_j
    coord_5 = List()  # atom_k
    values = List()
    
    for class_idx in range(num_triplet_classes):
        num_equiv = equivalent_count[class_idx]
        num_independent = independent_basis_count[class_idx]
        
        for equiv_idx in range(num_equiv):
            atom_i = equivalent_triplets[0, equiv_idx, class_idx]
            atom_j = equivalent_triplets[1, equiv_idx, class_idx]
            atom_k = equivalent_triplets[2, equiv_idx, class_idx]
            
            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        tri_basis_index = (coord_l * 3 + coord_m) * 3 + coord_n
                        
                        for independent_idx in range(num_independent):
                            value = (
                                transformation_array[tri_basis_index, independent_idx, equiv_idx, class_idx]
                                * force_constant_values[accumulated_independent[class_idx] + independent_idx]
                            )
                            
                            if value == 0.0:
                                continue
                            
                            coord_0.append(coord_l)
                            coord_1.append(coord_m)
                            coord_2.append(coord_n)
                            coord_3.append(atom_i)
                            coord_4.append(atom_j)
                            coord_5.append(atom_k)
                            values.append(value)
    
    return coord_0, coord_1, coord_2, coord_3, coord_4, coord_5, values
