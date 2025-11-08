# FCS-Order: Force Constants Calculation Tool for VASP

FCS-Order是一个用于计算VASP力常数的命令行工具，支持二阶、三阶和四阶力常数计算，以及自洽声子计算。

## 安装

### 使用pip安装

```bash
pip install fcs-order
```

### 从源码安装

```bash
git clone <repository-url>
cd fcs-order
pip install -e .
```

## 依赖

FCS-Order需要以下依赖：
- Python 3.8+
- typer
- numpy
- ase
- phonopy
- seekpath
- matplotlib
- pandas

## 命令行工具

FCS-Order提供以下命令：

### 主要命令

#### `sow3` - 生成三阶力常数计算文件

```bash
fcsorder sow3 na nb nc --cutoff CUTOFF
```

参数：
- `na`, `nb`, `nc`: 超胞在a、b、c方向的维度
- `--cutoff`: 截断距离（负值表示最近邻，正值表示以nm为单位的距离）

#### `sow4` - 生成四阶力常数计算文件

```bash
fcsorder sow4 na nb nc --cutoff CUTOFF
```

参数：
- `na`, `nb`, `nc`: 超胞在a、b、c方向的维度
- `--cutoff`: 截断距离（负值表示最近邻，正值表示以nm为单位的距离）

#### `reap3` - 从VASP计算结果提取三阶力常数

```bash
fcsorder reap3 na nb nc --cutoff CUTOFF [--is-sparse] vasprun.xml...
```

参数：
- `na`, `nb`, `nc`: 超胞在a、b、c方向的维度
- `--cutoff`: 截断距离（负值表示最近邻，正值表示以nm为单位的距离）
- `--is-sparse`: 使用稀疏张量方法提高内存效率（可选）
- `vasprun.xml`: VASP计算结果文件的路径（可多个）

#### `reap4` - 从VASP计算结果提取四阶力常数

```bash
fcsorder reap4 na nb nc --cutoff CUTOFF [--is-sparse] vasprun.xml...
```

参数：
- `na`, `nb`, `nc`: 超胞在a、b、c方向的维度
- `--cutoff`: 截断距离（负值表示最近邻，正值表示以nm为单位的距离）
- `--is-sparse`: 使用稀疏张量方法提高内存效率（可选）
- `vasprun.xml`: VASP计算结果文件的路径（可多个）

#### `plot_phband` - 绘制声子能带图

```bash
fcsorder plot_phband na nb nc primcell fcs_orders...
```

参数：
- `na`, `nb`, `nc`: 超胞在a、b、c方向的维度
- `primcell`: 原胞文件路径（如POSCAR）
- `fcs_orders`: FORCE_CONSTANTS文件的路径（可多个）

此工具使用原胞结构和多个FORCE_CONSTANTS文件的力常数生成声子能带图，每个数据集使用colormap中的不同颜色绘制。

#### `phonon_sow` - 生成声子扰动结构

```bash
fcsorder phonon_sow [OPTIONS]
```

### 子命令组

#### `mlp2` - 使用机器学习势计算二阶力常数

```bash
fcsorder mlp2 [SUBCOMMAND] [OPTIONS]
```

#### `mlp3` - 使用机器学习势计算三阶力常数

```bash
fcsorder mlp3 [SUBCOMMAND] [OPTIONS]
```

#### `mlp4` - 使用机器学习势计算四阶力常数

```bash
fcsorder mlp4 [SUBCOMMAND] [OPTIONS]
```

#### `scph` - 使用机器学习势运行自洽声子计算

```bash
fcsorder scph [SUBCOMMAND] [OPTIONS]
```

## 使用示例

### 1. 生成三阶力常数计算文件

```bash
fcsorder sow3 2 2 2 --cutoff -3
```

### 2. 从VASP计算结果提取三阶力常数

```bash
fcsorder reap3 2 2 2 --cutoff -3 vasprun_1.xml vasprun_2.xml
```

### 3. 绘制声子能带图

```bash
fcsorder plot_phband 2 2 2 POSCAR FORCE_CONSTANTS_1 FORCE_CONSTANTS_2
```

## 输出文件

- `3RD.POSCAR.*`: 三阶力常数计算用的位移结构
- `4TH.POSCAR.*`: 四阶力常数计算用的位移结构
- `FORCE_CONSTANTS`: 提取的力常数文件
- `phband.svg`: 声子能带图

## 许可证

请参阅LICENSE文件了解许可证信息。

## 贡献

欢迎提交问题和拉取请求来改进这个项目。

## 引用

如果您在研究中使用了FCS-Order，请考虑引用相关论文。