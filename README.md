# FCS-Order：二阶、三阶和四阶力常数计算工具包

## 项目背景与改进动机

### 原有脚本的痛点

传统的`thirdorder`和`fourthorder`脚本在力常数计算中存在以下主要问题：

1. **安装配置困难**
   - 依赖Python 2.x，与现代Python环境不兼容
   - 需要手动配置编译环境，安装过程复杂
   - 依赖库版本冲突频繁，环境配置耗时

2. **环境依赖严重**
   - 硬编码路径和系统配置，迁移困难
   - 缺乏现代化的包管理机制
   - 不同操作系统兼容性差

3. **使用门槛高**
   - 缺乏详细的安装和使用文档
   - 错误提示不清晰，调试困难
   - 需要用户手动处理大量中间文件

### 本项目的核心改进

#### 1. 现代化环境配置

- **Python 3.7+支持**：完全兼容现代Python环境
- **标准化安装**：使用`pyproject.toml`和`uv`包管理器
- **一键安装**：简化的安装流程，自动处理依赖关系

#### 3. 命令行界面统一

```bash
# 简化的命令结构，一致的接口设计
thirdorder --help    # 三阶力常数
fourthorder --help   # 四阶力常数
secondorder --help   # 新增：有限温度下的二阶力常数
```

## 核心功能增强

### 1. 机器学习势函数支持（重大改进）

在保持原有VASP兼容性的基础上，新增了对现代机器学习势函数的支持：

```bash
# 使用NEP势函数
thirdorder get-fc 2 2 2 --calc nep --potential nep.txt

# 使用DeepMD势函数
fourthorder get-fc 2 2 2 --calc deepmd --potential frozen_model.pb

# 使用HiPhive势函数
thirdorder get-fc 3 3 3 --calc hiphive --potential potential.fcp
```

### 2. 有限温度二阶力常数计算（全新功能）依赖hiPhive

```bash
# 基于机器学习势函数的有限温度二阶力常数
secondorder 2 2 2 --calc nep --potential model.txt --temperatures "300,600,900"

# 多温度采样，考虑温度效应
secondorder 3 3 3 --calc deepmd --potential model.pb --temperatures "200,400,600,800"
```

## 技术架构升级

### 现代化依赖管理

```python
# 使用Click替代optparse，更好的命令行体验
import click

@click.group()
def thirdorder():
    """Third-order force constants calculation toolkit."""
    pass

## 安装使用（极大简化）

### 快速安装

直接通过pip安装：

```bash
pip install fcs-order
```

```bash
# 克隆仓库
git clone <repository-url>
cd fcs-order
uv pip install -e .

# 验证安装
thirdorder --help
fourthorder --help
secondorder --help
```

### 可选依赖（按需安装）

```bash
# 如果需要使用机器学习势函数
uv pip install calorine     # NEP势函数
uv pip install deepmd-kit  # DeepMD势函数
uv pip install hiphive      # HiPhive势函数
uv pip install pypolymlp   # PYPOLYMlp势函数
```


## 致谢与依赖

### 机器学习势函数

- **[CALORINE](https://gitlab.com/materials-modeling/calorine)**：NEP势函数支持
- **[DeepMD-kit](https://github.com/deepmodeling/deepmd-kit)**：深度机器学习势函数
- **[HiPhive](https://hiphive.materialsmodeling.org/)**：力常数势函数
- **[PYPOLYMlp](https://github.com/segala-project/pypolymlp)**：多项式机器学习势

### 原有项目致谢

我们衷心感谢原始`thirdorder`和`fourthorder`项目的开发者们，他们的工作为本项目奠定了基础。我们在保持原有功能兼容性的同时，进行了现代化改进和功能扩展。
