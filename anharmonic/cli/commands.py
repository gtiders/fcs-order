"""
CLI 命令定义模块

定义所有 Typer 命令
"""

from pathlib import Path
import numpy as np
import typer
from rich.progress import track

from anharmonic.config import Config
from anharmonic.io.readers import StructureReader
from anharmonic.io.writers import ForceConstantsWriter
from anharmonic.core.symmetry import SymmetryAnalyzer
from anharmonic.core.wedge import TripletWedge
from anharmonic.core.reconstruction import IFCReconstructor
from anharmonic.core.utils import (
    calculate_distances,
    calculate_cutoff_range,
    displace_two_atoms,
    generate_supercell,
    parse_cutoff,
    validate_supercell_size,
)
from anharmonic.calculators import CalculatorFactory

app = typer.Typer(
    name="anharmonic",
    help="三阶非谐力常数计算工具",
    add_completion=False,
)


@app.command("calculate")
def calculate(
    na: int = typer.Argument(..., help="超胞 a 方向扩展倍数"),
    nb: int = typer.Argument(..., help="超胞 b 方向扩展倍数"),
    nc: int = typer.Argument(..., help="超胞 c 方向扩展倍数"),
    cutoff: str = typer.Option(
        ...,
        "--cutoff", "-c",
        help="截断距离（负值表示第 n 近邻，正值表示距离/nm）",
    ),
    calculator_type: str = typer.Option(
        ...,
        "--calculator", "-calc",
        help="计算器类型: nep, dp, mtp, polymlp, tace",
    ),
    potential: Path = typer.Option(
        ...,
        "--potential", "-p",
        exists=True,
        help="势函数文件路径",
    ),
    structure: Path = typer.Option(
        Path("POSCAR"),
        "--structure", "-s",
        exists=True,
        help="结构文件路径",
    ),
    output: Path = typer.Option(
        Path("FORCE_CONSTANTS_3RD"),
        "--output", "-o",
        help="输出文件路径",
    ),
    save_intermediates: bool = typer.Option(
        False,
        "--save-intermediates", "-w",
        help="是否保存中间计算文件",
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="使用 GPU 计算（仅部分计算器支持）",
    ),
):
    """
    计算三阶力常数
    
    使用机器学习势函数计算三阶非谐力常数，
    输出 ShengBTE 格式的 FORCE_CONSTANTS_3RD 文件。
    """
    config = Config()
    displacement_magnitude = config.displacement_magnitude
    symmetry_precision = config.symmetry_precision
    
    # 验证参数
    validate_supercell_size(na, nb, nc)
    neighbor_order, cutoff_range = parse_cutoff(cutoff)
    
    # 创建计算器
    typer.echo(f"创建 {calculator_type} 计算器...")
    calculator = CalculatorFactory.create(
        calculator_type=calculator_type,
        potential_path=potential,
        is_gpu=gpu,
    )
    
    # 读取结构
    typer.echo("读取结构文件...")
    primitive = StructureReader.read(structure)
    
    # 分析对称性
    typer.echo("分析晶体对称性...")
    symmetry = SymmetryAnalyzer(
        lattice_vectors=primitive.lattice_vectors,
        atom_types=primitive.atom_types,
        positions=primitive.positions.T,
        symmetry_precision=symmetry_precision,
    )
    typer.echo(f"  空间群: {symmetry.space_group_symbol}")
    typer.echo(f"  对称操作数: {symmetry.num_symmetry_operations}")
    
    # 生成超胞
    typer.echo("生成超胞...")
    supercell = generate_supercell(primitive, na, nb, nc)
    
    # 计算距离
    typer.echo("计算原子间距离...")
    distance_matrix, equivalent_count, shift_vectors = calculate_distances(supercell)
    
    # 确定截断距离
    if neighbor_order is not None:
        cutoff_range = calculate_cutoff_range(
            primitive, supercell, neighbor_order, distance_matrix
        )
        typer.echo(f"  自动截断距离: {cutoff_range:.4f} nm")
    else:
        typer.echo(f"  用户指定截断距离: {cutoff_range:.4f} nm")
    
    # 寻找不可约力常数集合
    typer.echo("寻找不可约三阶力常数集合...")
    wedge = TripletWedge(
        primitive=primitive,
        supercell=supercell,
        symmetry=symmetry,
        distance_matrix=distance_matrix,
        equivalent_count_matrix=equivalent_count,
        shift_vectors=shift_vectors,
        cutoff_range=cutoff_range,
    )
    
    irreducible_displacements = wedge.get_irreducible_displacements()
    num_irreducible = len(irreducible_displacements)
    typer.echo(f"  不可约位移数: {num_irreducible}")
    
    num_supercell_atoms = supercell.num_atoms
    
    # 保存参考结构
    if save_intermediates:
        atoms = supercell.to_atoms(calculator)
        atoms.get_forces()
        atoms.write("3RD.SPOSCAR.xyz", format="extxyz")
    
    # 计算力常数
    typer.echo("计算非谐力常数...")
    width = len(str(4 * (num_irreducible + 1)))
    name_pattern = f"3RD.POSCAR.{{:0{width}d}}.xyz"
    
    partial_force_constants = np.zeros((3, num_irreducible, num_supercell_atoms))
    
    for disp_idx, displacement in enumerate(track(irreducible_displacements, description="计算位移")):
        atom_i, atom_j, coord_i, coord_j = displacement
        
        for n in range(4):
            sign_i = (-1) ** (n // 2)
            sign_j = -((-1) ** (n % 2))
            
            # 位移原子
            displaced_structure = displace_two_atoms(
                supercell,
                atom_j, coord_j, sign_i * displacement_magnitude,
                atom_i, coord_i, sign_j * displacement_magnitude,
            )
            
            # 计算力
            atoms = displaced_structure.to_atoms(calculator)
            forces = atoms.get_forces()
            
            # 累积力常数
            partial_force_constants[:, disp_idx, :] -= sign_i * sign_j * forces.T
            
            # 保存中间文件
            if save_intermediates:
                number = num_irreducible * n + disp_idx + 1
                atoms.write(name_pattern.format(number), format="extxyz")
    
    # 归一化
    partial_force_constants /= (400.0 * displacement_magnitude * displacement_magnitude)
    
    # 重构完整力常数
    typer.echo("重构完整力常数矩阵...")
    full_force_constants = IFCReconstructor.reconstruct(
        partial_force_constants=partial_force_constants,
        wedge=wedge,
        irreducible_displacements=irreducible_displacements,
        primitive=primitive,
        supercell=supercell,
    )
    
    # 写入输出文件
    typer.echo(f"写入力常数文件: {output}")
    ForceConstantsWriter.write_third_order(
        force_constants=full_force_constants,
        primitive=primitive,
        supercell=supercell,
        distance_matrix=distance_matrix,
        equivalent_count=equivalent_count,
        shift_vectors=shift_vectors,
        cutoff_range=cutoff_range,
        output_path=output,
    )
    
    typer.secho("三阶力常数计算完成!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
