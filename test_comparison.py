#!/usr/bin/env python3
"""
测试脚本：对比 thirdorder 和 thirdorderpp 的输出
使用 ASE 读取结构文件并进行数值比较
"""

import os
import sys
import subprocess
import glob
import numpy as np
from pathlib import Path

try:
    from ase.io import read
except ImportError:
    print("错误：需要安装 ASE 库")
    print("运行: uv pip install ase")
    sys.exit(1)


# 定义目录
THIRDORDER_DIR = Path("/home/gwins/codespace/mlfcs/thirdorder")
THIRDORDERPP_DIR = Path("/home/gwins/codespace/mlfcs/thirdorderpp")
TEST_THIRDORDER = Path("/home/gwins/codespace/mlfcs/test/thirdorder")
TEST_THIRDORDERPP = Path("/home/gwins/codespace/mlfcs/test/thirdorderpp")


def clean_test_files(test_dir):
    """清理测试目录中的所有生成文件"""
    patterns = ["3RD.*", "FORCE_CONSTANTS_3RD", "output.txt"]
    for pattern in patterns:
        for file in test_dir.glob(pattern):
            file.unlink(missing_ok=True)
    print(f"   清理 {test_dir} 完成")


def run_command(cmd, cwd):
    """运行命令并返回结果"""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def compare_structures(file1, file2, decimal_places=6):
    """
    使用 ASE 比较两个结构文件
    
    参数:
        file1, file2: 结构文件路径
        decimal_places: 小数点后位数 (默认6位)
    
    返回:
        (is_same, differences)
    """
    tolerance = 10 ** (-decimal_places)  # 小数点后6位 = 1e-6
    
    try:
        atoms1 = read(str(file1))
        atoms2 = read(str(file2))
    except Exception as e:
        return False, f"读取文件失败: {e}"
    
    differences = []
    
    # 比较原子数量
    if len(atoms1) != len(atoms2):
        differences.append(f"原子数量不同: {len(atoms1)} vs {len(atoms2)}")
        return False, "\n".join(differences)
    
    # 比较晶胞 (cell) - 使用绝对容差
    cell1 = atoms1.get_cell()
    cell2 = atoms2.get_cell()
    if not np.allclose(cell1, cell2, rtol=0, atol=tolerance):
        max_diff = np.max(np.abs(cell1 - cell2))
        differences.append(f"晶胞不同，最大差异: {max_diff:.2e}")
    
    # 比较原子位置 (positions) - 使用绝对容差
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    if not np.allclose(pos1, pos2, rtol=0, atol=tolerance):
        max_diff = np.max(np.abs(pos1 - pos2))
        differences.append(f"原子位置不同，最大差异: {max_diff:.2e}")
    
    # 比较化学符号
    symbols1 = atoms1.get_chemical_symbols()
    symbols2 = atoms2.get_chemical_symbols()
    if symbols1 != symbols2:
        differences.append("化学符号不同")
    
    if differences:
        return False, "\n".join(differences)
    else:
        return True, "结构完全一致（在给定容差内）"


def main():
    print("=" * 60)
    print("测试 thirdorder vs thirdorderpp")
    print("=" * 60)
    
    # 1. 清理之前的测试文件
    print("\n1. 清理之前的测试文件...")
    clean_test_files(TEST_THIRDORDER)
    clean_test_files(TEST_THIRDORDERPP)
    
    # 2. 运行 thirdorder (原始版本)
    print("\n2. 运行 thirdorder (C库版本)...")
    cmd = f"uv run python {THIRDORDER_DIR}/thirdorder_vasp.py sow 2 2 2 -2"
    returncode, stdout, stderr = run_command(cmd, TEST_THIRDORDER)
    
    # 保存输出
    (TEST_THIRDORDER / "output.txt").write_text(stdout + stderr)
    
    if returncode != 0:
        print(f"   错误：thirdorder 运行失败")
        print(f"   错误信息: {stderr}")
        return 1
    print("   完成")
    
    # 3. 运行 thirdorderpp (Python版本)
    print("\n3. 运行 thirdorderpp (Python包版本)...")
    cmd = f"uv run python {THIRDORDERPP_DIR}/thirdorder_vasp.py sow 2 2 2 -2"
    returncode, stdout, stderr = run_command(cmd, TEST_THIRDORDERPP)
    
    # 保存输出
    (TEST_THIRDORDERPP / "output.txt").write_text(stdout + stderr)
    
    if returncode != 0:
        print(f"   错误：thirdorderpp 运行失败")
        print(f"   错误信息: {stderr}")
        return 1
    print("   完成")
    
    # 4. 获取生成的文件列表
    print("\n4. 对比生成的文件...")
    print("=" * 60)
    
    # 获取所有 3RD.* 文件
    files_thirdorder = sorted(TEST_THIRDORDER.glob("3RD.*"))
    files_thirdorderpp = sorted(TEST_THIRDORDERPP.glob("3RD.*"))
    
    if not files_thirdorder:
        print("错误：thirdorder 没有生成文件")
        return 1
    
    print(f"\n找到 {len(files_thirdorder)} 个文件（thirdorder）")
    print(f"找到 {len(files_thirdorderpp)} 个文件（thirdorderpp）")
    
    # 5. 比较每个文件
    all_match = True
    decimal_places = 6  # 小数点后6位
    tolerance = 10 ** (-decimal_places)  # 1e-6
    
    for file1 in files_thirdorder:
        file2 = TEST_THIRDORDERPP / file1.name
        
        print(f"\n对比文件: {file1.name}")
        
        if not file2.exists():
            print(f"  ✗ 文件在 thirdorderpp 中不存在")
            all_match = False
            continue
        
        # 使用 ASE 比较
        is_same, message = compare_structures(file1, file2, decimal_places)
        
        if is_same:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
            all_match = False
    
    # 检查是否有额外的文件
    for file2 in files_thirdorderpp:
        file1 = TEST_THIRDORDER / file2.name
        if not file1.exists():
            print(f"\n✗ 额外文件 {file2.name} 在 thirdorderpp 中存在但在 thirdorder 中不存在")
            all_match = False
    
    # 6. 检查 FORCE_CONSTANTS_3RD 文件
    print("\n" + "=" * 60)
    print("5. 对比 FORCE_CONSTANTS_3RD 文件...")
    
    fc_file1 = TEST_THIRDORDER / "FORCE_CONSTANTS_3RD"
    fc_file2 = TEST_THIRDORDERPP / "FORCE_CONSTANTS_3RD"
    
    if fc_file1.exists() and fc_file2.exists():
        try:
            data1 = np.loadtxt(str(fc_file1))
            data2 = np.loadtxt(str(fc_file2))
            
            if np.allclose(data1, data2, rtol=0, atol=tolerance):
                print("  ✓ FORCE_CONSTANTS_3RD 文件数值一致（小数点后6位相同）")
            else:
                max_diff = np.max(np.abs(data1 - data2))
                print(f"  ✗ FORCE_CONSTANTS_3RD 文件数值不同，最大差异: {max_diff:.2e}")
                all_match = False
        except Exception as e:
            print(f"  ✗ 比较 FORCE_CONSTANTS_3RD 失败: {e}")
            all_match = False
    elif fc_file1.exists() or fc_file2.exists():
        print("  ✗ FORCE_CONSTANTS_3RD 文件只在一个目录中存在")
        all_match = False
    
    # 7. 总结
    print("\n" + "=" * 60)
    print("6. 测试总结")
    print("=" * 60)
    
    if all_match:
        print("✓✓✓ 所有文件在数值上完全一致！测试通过！")
        print(f"\n容差设置: 小数点后 {decimal_places} 位 (绝对容差 = {tolerance:.2e})")
        print("\nthirdorderpp 成功迁移到 Python spglib 包")
        print("生成的结构文件与原版本一致")
        exit_code = 0
    else:
        print("✗✗✗ 发现差异，需要进一步检查")
        exit_code = 1
    
    # 8. 清理测试文件
    print("\n" + "=" * 60)
    print("7. 清理测试文件...")
    clean_test_files(TEST_THIRDORDER)
    clean_test_files(TEST_THIRDORDERPP)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
