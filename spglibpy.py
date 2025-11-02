#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用spglib分析POSCAR文件的对称性
"""

import numpy as np
import spglib
from ase.io import read


def analyze_symmetry(poscar_file="POSCAR", symprec=1e-5):
    """
    使用spglib分析POSCAR文件的对称性

    Parameters:
    -----------
    poscar_file : str
        POSCAR文件路径
    symprec : float
        对称性判断精度
    """

    # 读取POSCAR文件
    print(f"读取POSCAR文件: {poscar_file}")
    atoms = read(poscar_file)

    # 获取晶格和原子位置信息
    lattice = atoms.get_cell().array
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    print("\n" + "=" * 60)
    print("晶体结构信息")
    print("=" * 60)
    print(f"晶胞参数 (Å):")
    for i, vec in enumerate(lattice):
        print(f"  a{i + 1}: [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}]")

    print(f"\n原子种类和数量:")
    unique_numbers, counts = np.unique(numbers, return_counts=True)
    for num, count in zip(unique_numbers, counts):
        element = atoms.get_chemical_symbols()[np.where(numbers == num)[0][0]]
        print(f"  {element}: {count} 个原子")

    print(f"\n总原子数: {len(atoms)}")

    # 使用spglib分析对称性
    print("\n" + "=" * 60)
    print("对称性分析结果")
    print("=" * 60)

    # 获取空间群信息
    dataset = spglib.get_symmetry_dataset(
        (lattice, positions, numbers), symprec=symprec
    )

    if dataset is None:
        print("错误: 无法获取对称性数据")
        return

    print(f"空间群编号: {dataset['number']}")
    print(f"国际符号: {dataset['international']}")
    print(f"Hall符号: {dataset['hall']}")
    print(f"点群: {dataset['pointgroup']}")

    print(f"\n对称操作数量: {len(dataset['rotations'])}")
    print(f"Wyckoff位置数量: {len(dataset['wyckoffs'])}")

    # 显示Wyckoff位置
    print(f"\nWyckoff位置:")
    unique_wyckoffs = np.unique(dataset["wyckoffs"])
    for wyckoff in unique_wyckoffs:
        count = np.sum(dataset["wyckoffs"] == wyckoff)
        print(f"  {wyckoff}: {count} 个原子")

    # 显示等效原子
    print(f"\n等效原子映射:")
    for i, equiv in enumerate(dataset["equivalent_atoms"]):
        if i == equiv:
            equiv_atoms = np.where(dataset["equivalent_atoms"] == i)[0]
            element = atoms.get_chemical_symbols()[i]
            print(f"  {element}{i + 1} -> 等效原子: {[j + 1 for j in equiv_atoms]}")

    # 显示变换矩阵和原点偏移
    print(f"\n变换矩阵:")
    for i in range(3):
        print(
            f"  [{dataset['transformation_matrix'][i][0]:6.3f}, "
            f"{dataset['transformation_matrix'][i][1]:6.3f}, "
            f"{dataset['transformation_matrix'][i][2]:6.3f}]"
        )

    print(f"\n原点偏移:")
    print(
        f"  [{dataset['origin_shift'][0]:6.3f}, "
        f"{dataset['origin_shift'][1]:6.3f}, "
        f"{dataset['origin_shift'][2]:6.3f}]"
    )

    # 检查晶体系统
    crystal_system = get_crystal_system(dataset["number"])
    print(f"\n晶体系统: {crystal_system}")

    # 显示前几个对称操作
    print(f"\n前5个对称操作:")
    for i in range(min(5, len(dataset["rotations"]))):
        print(f"操作 {i + 1}:")
        print("  旋转矩阵:")
        for j in range(3):
            print(
                f"    [{dataset['rotations'][i][j][0]:2.0f}, "
                f"{dataset['rotations'][i][j][1]:2.0f}, "
                f"{dataset['rotations'][i][j][2]:2.0f}]"
            )
        print(
            f"  平移向量: [{dataset['translations'][i][0]:6.3f}, "
            f"{dataset['translations'][i][1]:6.3f}, "
            f"{dataset['translations'][i][2]:6.3f}]"
        )


def get_crystal_system(spacegroup_number):
    """
    根据空间群编号获取晶体系统
    """
    if 1 <= spacegroup_number <= 2:
        return "三斜"
    elif 3 <= spacegroup_number <= 15:
        return "单斜"
    elif 16 <= spacegroup_number <= 74:
        return "正交"
    elif 75 <= spacegroup_number <= 142:
        return "四方"
    elif 143 <= spacegroup_number <= 167:
        return "三角"
    elif 168 <= spacegroup_number <= 194:
        return "六角"
    elif 195 <= spacegroup_number <= 230:
        return "立方"
    else:
        return "未知"


def find_primitive_cell(poscar_file="POSCAR", symprec=1e-5):
    """
    寻找原胞
    """
    print("\n" + "=" * 60)
    print("原胞分析")
    print("=" * 60)

    atoms = read(poscar_file)
    lattice = atoms.get_cell().array
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    # 寻找原胞
    primitive_lattice, primitive_positions, primitive_numbers = spglib.find_primitive(
        (lattice, positions, numbers), symprec=symprec
    )

    if primitive_lattice is not None:
        print("原胞晶格参数 (Å):")
        for i, vec in enumerate(primitive_lattice):
            print(f"  a{i + 1}: [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}]")

        print(f"\n原胞原子数: {len(primitive_numbers)}")
        unique_numbers, counts = np.unique(primitive_numbers, return_counts=True)
        for num, count in zip(unique_numbers, counts):
            element = atoms.get_chemical_symbols()[np.where(numbers == num)[0][0]]
            print(f"  {element}: {count} 个原子")
    else:
        print("无法找到原胞")


if __name__ == "__main__":
    # 分析对称性
    analyze_symmetry()

    # 寻找原胞
    find_primitive_cell()

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)
