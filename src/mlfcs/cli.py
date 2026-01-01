#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mlfcs 命令行入口点

用法:
    mlfcs thirdorder sow|reap na nb nc --cutoff VALUE [options]
    mlfcs fourthorder sow|reap na nb nc --cutoff VALUE [options]

示例:
    mlfcs thirdorder sow 2 2 2 --cutoff 0.4
    mlfcs thirdorder sow 2 2 2 --cutoff -3 --structure prim.cif
    mlfcs thirdorder reap 2 2 2 --cutoff -3 --forces output.xyz
    mlfcs fourthorder reap 2 2 2 --cutoff -3 --forces vasprun/*.xml --symprec 1e-4
"""

import argparse
import sys
import glob
from .file_io import StructureData
from .utils import read_forces_from_files


def main():
    """主命令行入口函数"""
    parser = argparse.ArgumentParser(
        prog="mlfcs",
        description="Machine Learning Force Constant Suite - 计算非谐力常数 (IFCs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成位移结构
  mlfcs thirdorder sow 2 2 2 --cutoff 0.4 --structure POSCAR
  mlfcs thirdorder sow 3 3 3 --cutoff -8 --bundle-output displaced.xyz
  
  # 收集力常数 (需要包含 'forces' 属性的文件)
  mlfcs thirdorder reap 2 2 2 --cutoff 0.4 --forces output.xyz
  mlfcs fourthorder reap 2 2 2 --cutoff -3 --forces vasprun*.xml
  
  # 自定义参数
  mlfcs thirdorder sow 3 3 3 --cutoff -8 --h 0.002 --symprec 1e-4

参数说明:
  cutoff     正数为距离截断 (nm); 负整数为近邻数截断 (如 -3 表示第3近邻)
  h          有限差分位移幅度 (nm), 默认 1e-3 即 0.01 Å
  symprec    对称性检测容差, 默认 1e-5
  
注意:
  reap 模式要求力文件必须包含 'forces' 属性 (extxyz 格式)
""",
    )

    subparsers = parser.add_subparsers(dest="order", help="选择计算阶数")

    # thirdorder 子命令
    third_parser = subparsers.add_parser(
        "thirdorder", aliases=["3rd"], help="三阶力常数计算"
    )
    _add_common_args(third_parser)

    # fourthorder 子命令
    fourth_parser = subparsers.add_parser(
        "fourthorder", aliases=["4th"], help="四阶力常数计算"
    )
    _add_common_args(fourth_parser)

    args = parser.parse_args()

    if args.order is None:
        parser.print_help()
        sys.exit(1)

    if args.order in ("thirdorder", "3rd"):
        _run_thirdorder(args)
    elif args.order in ("fourthorder", "4th"):
        _run_fourthorder(args)


def _add_common_args(parser):
    """添加通用参数"""
    parser.add_argument(
        "action",
        choices=["sow", "reap"],
        help="sow: 生成位移结构; reap: 从力文件收集力常数",
    )
    parser.add_argument("na", type=int, help="超胞 a 方向倍数")
    parser.add_argument("nb", type=int, help="超胞 b 方向倍数")
    parser.add_argument("nc", type=int, help="超胞 c 方向倍数")
    parser.add_argument(
        "--cutoff",
        required=True,
        help="截断参数: 正数为距离(nm), 负整数为近邻数 (如 -3)",
    )
    parser.add_argument(
        "--structure",
        default="POSCAR",
        help="结构文件路径 (默认: POSCAR, 支持 VASP/CIF/XYZ 等格式)",
    )
    parser.add_argument(
        "--prim", action="store_true", help="标记输入结构为原胞 (目前未使用)"
    )
    parser.add_argument(
        "--forces", nargs="+", help="力文件列表, 必须包含 'forces' 属性 (extxyz 格式)"
    )
    parser.add_argument(
        "--bundle-output",
        help="sow 模式: 将所有位移结构写入单个文件 (推荐 .xyz 或 .extxyz)",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=None,
        help="有限差分位移幅度 (nm), 默认 1e-3 (0.01 Å)",
    )
    parser.add_argument(
        "--symprec", type=float, default=None, help="对称性检测容差, 默认 1e-5"
    )


def _run_thirdorder(args):
    """运行 thirdorder 计算"""
    # 读取结构
    try:
        struct = StructureData.from_file(args.structure)
    except Exception as e:
        sys.exit(f"Error reading structure file '{args.structure}': {e}")

    # 准备力数据
    forces = None
    if args.action == "reap":
        if not args.forces:
            sys.exit("Error: 'reap' action requires --forces argument")

        # 处理 glob 模式 (如果 shell 没有展开)
        expanded_files = []
        for f in args.forces:
            if "*" in f or "?" in f:
                globbed = glob.glob(f)
                if not globbed:
                    print(f"Warning: No files matched pattern '{f}'")
                expanded_files.extend(globbed)
            else:
                expanded_files.append(f)

        if not expanded_files:
            sys.exit("Error: No force files found")

        forces = read_forces_from_files(expanded_files)

    from .thirdorder import thirdorder_vasp as vasp

    vasp.run(
        args.action,
        args.na,
        args.nb,
        args.nc,
        args.cutoff,
        structure=struct,
        forces=forces,
        bundle_output=args.bundle_output,
        h=args.h,
        symprec=args.symprec,
    )


def _run_fourthorder(args):
    """运行 fourthorder 计算"""
    # 读取结构
    try:
        struct = StructureData.from_file(args.structure)
    except Exception as e:
        sys.exit(f"Error reading structure file '{args.structure}': {e}")

    # 准备力数据
    forces = None
    if args.action == "reap":
        if not args.forces:
            sys.exit("Error: 'reap' action requires --forces argument")

        # 处理 glob 模式
        expanded_files = []
        for f in args.forces:
            if "*" in f or "?" in f:
                globbed = glob.glob(f)
                if not globbed:
                    print(f"Warning: No files matched pattern '{f}'")
                expanded_files.extend(globbed)
            else:
                expanded_files.append(f)

        if not expanded_files:
            sys.exit("Error: No force files found")

        forces = read_forces_from_files(expanded_files)

    from .fourthorder import fourthorder_vasp as vasp

    vasp.run(
        args.action,
        args.na,
        args.nb,
        args.nc,
        args.cutoff,
        structure=struct,
        forces=forces,
        bundle_output=args.bundle_output,
        h=args.h,
        symprec=args.symprec,
    )


if __name__ == "__main__":
    main()
