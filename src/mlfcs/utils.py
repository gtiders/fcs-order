import numpy as np
import ase.io
import sys
from typing import List
from ase import Atoms


def read_forces_from_files(files: List[str]) -> np.ndarray:
    """从文件读取力数据。

    使用 ASE 的 get_forces() 方法获取力。该方法会自动检查 atoms.arrays 中是否
    已有力数据（例如从 extxyz 文件读取的 'forces' 属性）。

    注意: 如果文件不包含力信息且没有设置 calculator，会抛出错误。

    Args:
        files: 文件路径列表 (.xyz, .extxyz, 或其他 ASE 支持的格式)

    Returns:
        numpy.ndarray: 力数组，形状为 (n_frames, n_atoms, 3)

    Raises:
        RuntimeError: 如果无法获取力（文件无力信息且无 calculator）
    """
    forces = []
    print("Reading force files...")

    for i, f in enumerate(files):
        try:
            # 读取所有帧
            atoms_list = ase.io.read(f, index=":")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

            for frame_idx, atoms in enumerate(atoms_list):
                try:
                    # get_forces() 会检查 arrays 中是否有 forces，有则返回
                    f_arr = atoms.get_forces()
                    forces.append(f_arr)
                except RuntimeError as e:
                    # 没有力数据且没有 calculator
                    raise RuntimeError(
                        f"File '{f}' frame {frame_idx}: cannot get forces. "
                        f"Make sure the file contains 'forces' property (extxyz format). "
                        f"Available arrays: {list(atoms.arrays.keys())}. "
                        f"Original error: {e}"
                    )

        except RuntimeError:
            raise  # Re-raise as-is
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            sys.exit(1)

        if (i + 1) % 10 == 0:
            print(f"- Read {i + 1}/{len(files)} files")

    print(f"- Finished reading. Total frames: {len(forces)}")
    return np.array(forces)


def calculate_forces_with_calculator(
    atoms_list: List[Atoms],
    calculator,
    verbose: bool = True,
) -> np.ndarray:
    """使用 ASE Calculator 计算结构列表的力。

    Args:
        atoms_list: ASE Atoms 对象列表
        calculator: ASE Calculator 对象
        verbose: 是否打印进度

    Returns:
        numpy.ndarray: 力数组，形状为 (n_frames, n_atoms, 3)
    """
    forces = []
    n_total = len(atoms_list)

    if verbose:
        print(f"Calculating forces for {n_total} structures...")

    for i, atoms in enumerate(atoms_list):
        # 创建副本并设置计算器
        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator

        # 计算力
        f = atoms_copy.get_forces()
        forces.append(f)

        if verbose and (i + 1) % 10 == 0:
            print(f"- Calculated {i + 1}/{n_total} structures")

    if verbose:
        print(f"- Finished calculating. Total structures: {len(forces)}")

    return np.array(forces)


# Backward compatibility alias
read_forces_from_vasp = read_forces_from_files
