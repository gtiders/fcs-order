"""
anharmonic - 非谐力常数计算模块

现代化架构的非谐力常数计算库，支持：
- 二阶力常数 (基于 phonopy)
- 三阶力常数 (基于有限位移法)
- 四阶力常数 (基于有限位移法)
"""

from anharmonic.config import Config

__version__ = "0.1.0"

__all__ = [
    "Config",
    "core",
    "io",
    "models",
    "calculators",
    "cli",
]
