# fcs-order

基于 ASE 与 Typer 的声子力常数计算工具。支持计算二、三、四阶相互作用力常数，并提供基于多种机器学习势（NEP、DeepMD、PolyMLP、MTP）或 Hiphive 的 SCPH 自洽声子流程。

仓库地址：https://github.com/gtiders/fcs-order

## 功能特性

- 二阶力常数（兼容 phonopy）
- 三、四阶力常数（ShengBTE/文本格式）
- SCPH（自洽声子）流程
- 多种后端：NEP、DeepMD、PolyMLP、MTP 以及 Hiphive
- 面向 VASP 的 sow/reap 工具链

## 安装

- Python 3.9+
- 建议使用全新虚拟环境

```bash
pip install git+https://github.com/gtiders/fcs-order.git
```

按需安装可选依赖（根据所用后端选择）：

- NEP: `pip install calorine`
- DeepMD: `pip install deepmd-kit`
- PolyMLP: `pip install pypolymlp`
- Hiphive: `pip install hiphive`
- MTP: 外部 `mlp` 二进制（Moment Tensor Potential），确保在 PATH 中

## 命令行概览

主入口为 Typer 应用，包含以下子命令组：

- 顶层工具
  - `sow3` / `sow4`：生成三/四阶位移结构
  - `reap3` / `reap4`：收集 VASP 力并构建力常数
  - `plot_phband`：从多个 FORCE_CONSTANTS 绘制声子能带
  - `phonon_sow`：结构扰动工具
- 机器学习势与 SCPH 子应用
  - `mlp2`：使用 ML 势计算二阶力常数（子命令：`nep`/`dp`/`ploymp`/`mtp`）
  - `mlp3`：使用 ML 势计算三阶力常数（子命令：同上）
  - `mlp4`：使用 ML 势计算四阶力常数（子命令：同上）
  - `scph`：SCPH 工作流（子命令：`nep`/`dp`/`hiphive`/`ploymp`/`mtp`）

查看完整帮助：

```bash
python -m fcsorder --help
```

## 常用命令

### sow3（三阶位移）
```bash
python -m fcsorder sow3 NA NB NC --cutoff <CUTOFF> --poscar POSCAR
```

### sow4（四阶位移）
```bash
python -m fcsorder sow4 NA NB NC --cutoff <CUTOFF> --poscar POSCAR
```

### reap3（收集并生成三阶力常数）
```bash
python -m fcsorder reap3 NA NB NC --cutoff <CUTOFF> [--is-sparse] --poscar POSCAR VASPRUN1.xml VASPRUN2.xml ...
```

### reap4（收集并生成四阶力常数）
```bash
python -m fcsorder reap4 NA NB NC --cutoff <CUTOFF> [--is-sparse] --poscar POSCAR VASPRUN*.xml
```

### mlp2（二阶）
子命令：`nep`、`dp`、`ploymp`、`mtp`

超胞矩阵参数支持 3 个对角元素或 9 个元素（3x3 矩阵）。

- NEP
```bash
python -m fcsorder mlp2 nep 2 2 2 --potential nep.txt --poscar POSCAR --outfile FORCE_CONSTANTS_2ND [--is-gpu]
```
- DeepMD
```bash
python -m fcsorder mlp2 dp 2 2 2 --potential model.pb --poscar POSCAR --outfile FORCE_CONSTANTS_2ND
```
- PolyMLP
```bash
python -m fcsorder mlp2 ploymp 2 2 2 --potential polymlp.pot --poscar POSCAR --outfile FORCE_CONSTANTS_2ND
```
- MTP（需要 `mlp` 可执行文件）
```bash
python -m fcsorder mlp2 mtp 2 2 2 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] --outfile FORCE_CONSTANTS_2ND
```
说明：MTP 后端会自动从 ASE 的 Atoms 中获取元素种类，无需手动传入。

### mlp3（三阶）
```bash
python -m fcsorder mlp3 mtp 2 2 2 --cutoff 3.0 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] [--is-write] [--is-sparse]
```
其他后端与 `mlp2` 类似，但需要提供 `--cutoff`。

### mlp4（四阶）
```bash
python -m fcsorder mlp4 mtp 2 2 2 --cutoff 3.0 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] [--is-write] [--is-sparse]
```

### scph（自洽声子）
子命令：`nep`、`dp`、`hiphive`、`ploymp`、`mtp`

通用参数：

- `primcell`：原胞结构文件（如 POSCAR）
- `supercell_matrix`：3 或 9 个整数
- `temperatures`：如 "100,200,300"
- `cutoff`：簇空间截断（不同后端含义可能不同）

示例：

- NEP
```bash
python -m fcsorder scph nep POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential nep.txt [--is-gpu]
```
- DP
```bash
python -m fcsorder scph dp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential graph.pb
```
- Hiphive
```bash
python -m fcsorder scph hiphive POSCAR 2 2 2 --temperatures 300 --cutoff 3.0 --potential model.fcp
```
- PolyMLP
```bash
python -m fcsorder scph ploymp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential polymlp.pot
```
- MTP
```bash
python -m fcsorder scph mtp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential pot.mtp [--mtp-exe mlp]
```
说明：MTP 会从 `primcell` 的 Atoms 自动识别元素集合。

## MTP 注意事项

- 需要系统中可用的 `mlp` 可执行文件，或通过 `--mtp-exe` 指定路径。
- 临时文件默认写入系统临时目录。
- 计算中元素集合由传入的 ASE Atoms 自动提取。

## 开发

- 欢迎在 GitHub 发起 PR：https://github.com/gtiders/fcs-order

## 许可证

TBD。
