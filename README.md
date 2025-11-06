# FCS-Order: 多阶力常数与机器学习势计算工具

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-GPL%203.0%2B-green.svg)](LICENSE)

FCS-Order是一个全面的Python工具包，专为计算二阶、三阶和四阶力常数而设计，支持基于有限位移方法和机器学习势的高效计算。本工具特别适用于声子散射率计算、热导率预测以及材料热学性质研究。

## 🌟 主要特性

### 📊 多阶力常数计算
- **二阶力常数**：声子色散关系和振动性质分析
- **三阶力常数**：三声子相互作用和声子寿命计算
- **四阶力常数**：四声子相互作用和高阶热输运性质

### 🤖 机器学习势集成
- **NEP势**：高效的神经进化势函数
- **DeepMD势**：深度势能模型
- **HiPhive势**：高阶力常数拟合势
- **Polymlp势**：多项式机器学习势

### 💾 内存优化
- **稀疏张量方法**：大幅降低大系统内存需求
- **批处理计算**：高效处理大规模体系

### 🌡️ 热无序结构生成
- **声子扰动**：基于声子振动的热无序结构生成
- **温度控制**：支持任意温度下的结构生成
- **智能过滤**：自动过滤不合理结构

## 📦 安装

### 基础安装
```bash
pip install fcs-order
```

### 从源码安装
```bash
git clone https://github.com/gtiders/fcs-order.git
cd fcs-order
pip install -e .
```

### 可选依赖
```bash
# 安装所有机器学习势支持
pip install fcs-order[all]

# 或单独安装特定势
pip install fcs-order[deepmd]    # DeepMD势
pip install fcs-order[hiphive]    # HiPhive势
pip install fcs-order[mlp]        # Polymlp势
pip install fcs-order[calorine]   # Calorine库
```

## 🚀 快速开始

### 1. 二阶力常数计算（使用机器学习势）
```bash
# 使用NEP势计算二阶力常数
fcsorder mlp2 nep --potential nep.txt --supercell 4 4 4

# 使用DeepMD势
fcsorder mlp2 dp --potential model.pb --supercell 4 4 4

# 使用Polymlp势
fcsorder mlp2 polymlp --potential polymlp.yaml --supercell 4 4 4
```

### 2. 三阶力常数计算
```bash
# 使用机器学习势计算三阶力常数（稀疏张量优化）
fcsorder mlp3 nep --potential nep.txt --supercell 4 4 4 --cutoff 0.8 --is-sparse

# 传统VASP计算流程
fcsorder sow3 4 4 4 --cutoff -8  # 生成位移结构
# 运行VASP计算...
fcsorder reap3 4 4 4 --cutoff -8 --is-sparse vasprun.*.xml  # 提取力常数
```

### 3. 四阶力常数计算
```bash
# 使用机器学习势计算四阶力常数
fcsorder mlp4 nep --potential nep.txt --supercell 3 3 3 --cutoff 0.8

# 传统VASP计算流程
fcsorder sow4 3 3 3 --cutoff -8  # 生成位移结构
# 运行VASP计算...
fcsorder reap4 3 3 3 --cutoff -8 vasprun.*.xml  # 提取力常数
```

### 4. 热无序结构生成
```bash
# 基于声子振动生成热无序结构
fcsorder phononrattle SPOSCAR FORCE_CONSTANTS_2ND --temperatures 300,600,900 --number 100
```

## 📖 详细命令参考

### 二阶力常数命令 (mlp2)

#### 基本语法
```bash
fcsorder mlp2 <calculator> [options]
```

#### 计算器子命令
- `nep`: NEP势计算器
- `dp`: DeepMD势计算器
- `polymlp`: Polymlp势计算器

#### 共同参数
- `--supercell`: 超胞尺寸（格式：na nb nc）
- `--potential`: 势文件路径
- `--outfile`: 输出文件路径（默认：FORCECONSTANTS_2ND）

#### 示例
```bash
# NEP势计算
fcsorder mlp2 nep --supercell 4 4 4 --potential nep.txt

# 指定输出文件
fcsorder mlp2 dp --supercell 4 4 4 --potential model.pb --outfile my_fc2.dat

# GPU加速（仅NEP支持）
fcsorder mlp2 nep --supercell 4 4 4 --potential nep.txt --gpu
```

### 三阶力常数命令 (mlp3)

#### 基本语法
```bash
fcsorder mlp3 <calculator> [options]
```

#### 计算器子命令
- `nep`: NEP势计算器
- `dp`: DeepMD势计算器
- `polymlp`: Polymlp势计算器

#### 共同参数
- `--supercell`: 超胞尺寸（格式：na nb nc）
- `--cutoff`: 截断距离（负值为最近邻数，正值为距离nm）
- `--potential`: 势文件路径
- `--is-sparse`: 使用稀疏张量方法（推荐大系统）
- `--is-write`: 保存中间文件

#### 示例
```bash
# NEP势计算三阶力常数
fcsorder mlp3 nep --supercell 4 4 4 --cutoff 0.8 --potential nep.txt

# 使用稀疏张量方法（推荐大系统）
fcsorder mlp3 dp --supercell 4 4 4 --cutoff 0.8 --potential model.pb --is-sparse

# 保存中间文件
fcsorder mlp3 hiphive --supercell 4 4 4 --cutoff -8 --potential potential.fcp --is-write
```

### 四阶力常数命令 (mlp4)

#### 基本语法
```bash
fcsorder mlp4 <calculator> [options]
```

#### 计算器子命令
- `nep`: NEP势计算器
- `dp`: DeepMD势计算器
- `hiphive`: HiPhive势计算器
- `polymlp`: Polymlp势计算器

#### 共同参数
- `--supercell`: 超胞尺寸（格式：na nb nc）
- `--cutoff`: 截断距离（负值为最近邻数，正值为距离nm）
- `--potential`: 势文件路径
- `--is-write`: 保存中间文件

#### 示例
```bash
# NEP势计算四阶力常数
fcsorder mlp4 nep --supercell 3 3 3 --cutoff 0.8 --potential nep.txt

# DeepMD势计算
fcsorder mlp4 dp --supercell 3 3 3 --cutoff -8 --potential model.pb
```

### VASP计算命令

#### 三阶力常数 (sow3/reap3)
```bash
# 生成位移结构
fcsorder sow3 <na> <nb> <nc> --cutoff <cutoff>

# 提取力常数
fcsorder reap3 <na> <nb> <nc> --cutoff <cutoff> [--is-sparse] vasprun.*.xml
```

#### 四阶力常数 (sow4/reap4)
```bash
# 生成位移结构
fcsorder sow4 <na> <nb> <nc> --cutoff <cutoff>

# 提取力常数
fcsorder reap4 <na> <nb> <nc> --cutoff <cutoff> vasprun.*.xml
```

### 热无序结构生成 (phononrattle)

#### 基本语法
```bash
fcsorder phononrattle <SPOSCAR> <fc2_file> [options]
```

#### 参数
- `SPOSCAR`: 超胞结构文件
- `fc2_file`: 二阶力常数文件
- `--temperatures`: 温度列表（K），默认"300"
- `--number`: 每个温度生成的结构数，默认100
- `--min-distance`: 最小原子间距（Å），默认1.5
- `--if-qm`: 是否考虑量子效应，默认True
- `--imag-freq-factor`: 虚频因子，默认1.0
- `--output`: 输出文件前缀，默认"structures_phonon_rattle"

#### 示例
```bash
# 单温度生成
fcsorder phononrattle SPOSCAR FORCE_CONSTANTS_2ND --temperatures 300 --number 50

# 多温度生成
fcsorder phononrattle SPOSCAR FORCE_CONSTANTS_2ND --temperatures 300,600,900 --number 100

# 自定义参数
fcsorder phononrattle SPOSCAR FORCE_CONSTANTS_2ND --temperatures 800 --number 200 --min-distance 1.2
```

## 🔧 高级功能

### 稀疏张量优化

对于大系统（4×4×4超胞或更大），建议使用稀疏张量方法大幅降低内存需求：

```bash
# 三阶力常数稀疏计算
fcsorder mlp3 nep --supercell 4 4 4 --cutoff 0.8 --potential nep.txt --is-sparse
fcsorder reap3 4 4 4 --cutoff -8 --is-sparse vasprun.*.xml

# 二阶和四阶力常数目前使用密集存储
fcsorder mlp2 nep --supercell 4 4 4 --potential nep.txt
fcsorder mlp4 nep --supercell 3 3 3 --cutoff 0.8 --potential nep.txt
```

### GPU加速

NEP势支持GPU加速，可显著提高计算速度：

```bash
# 启用GPU加速
fcsorder mlp2 nep --supercell 4 4 4 --potential nep.txt --gpu
fcsorder mlp3 nep --supercell 4 4 4 --cutoff 0.8 --potential nep.txt --gpu
fcsorder mlp4 nep --supercell 3 3 3 --cutoff 0.8 --potential nep.txt --gpu
```

## 📁 文件格式

| 文件类型 | 描述 | 用途 |
|---------|------|------|
| SPOSCAR | VASP超胞结构文件 | 输入结构 |
| FORCECONSTANTS_2ND | 二阶力常数 | mlp2输出，声子计算输入 |
| FORCE_CONSTANTS_3RD | 三阶力常数 | mlp3/reap3输出 |
| FORCE_CONSTANTS_4TH | 四阶力常数 | mlp4/reap4输出 |
| 3RD.POSCAR.* | 三阶位移结构 | VASP计算输入 |
| 4TH.POSCAR.* | 四阶位移结构 | VASP计算输入 |
| *.xyz | 热无序结构 | phononrattle输出 |

## 🛠️ 系统要求

- **Python**: 3.10-3.13
- **操作系统**: Linux, macOS, Windows
- **核心依赖**: NumPy, SciPy, ASE, spglib, Typer
- **VASP**: 用于DFT计算（可选）
- **机器学习势包**: 根据需要安装

## 📚 应用场景

### 1. 声子热导率计算
```bash
# 完整的三阶力常数计算流程
fcsorder mlp2 nep --supercell 4 4 4 --potential nep.txt
fcsorder mlp3 nep --supercell 4 4 4 --cutoff 0.8 --potential nep.txt --is-sparse
# 使用ShengBTE或其他工具计算热导率
```

### 2. 高阶热输运性质研究
```bash
# 四阶力常数计算
fcsorder mlp4 nep --supercell 3 3 3 --cutoff 0.8 --potential nep.txt
# 结合三阶力常数研究四声子散射效应
```

### 3. 有限温度结构生成
```bash
# 生成高温下的热无序结构
fcsorder phononrattle SPOSCAR FORCE_CONSTANTS_2ND --temperatures 300,600,900 --number 100
# 用于分子动力学或结构性质研究
```

## 🤝 贡献指南

我们欢迎社区贡献！请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解详细信息。

## 📄 许可证

本项目采用GNU General Public License v3.0或更高版本许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- ASE项目提供了原子模拟环境
- spglib提供了空间群分析功能
- Typer提供了现代CLI框架
- 各种机器学习势项目的开发者

## 📞 联系我们

- **问题报告**: [GitHub Issues](https://github.com/gtiders/fcs-order/issues)
- **功能请求**: [GitHub Discussions](https://github.com/gtiders/fcs-order/discussions)
- **邮件联系**: [维护者邮箱]

---

**FCS-Order** - 让多阶力常数计算变得简单高效！
