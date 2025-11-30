"""
配置管理模块

集中管理常量、环境变量和默认配置
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """全局配置类"""
    
    # 有限位移幅度 (nm)
    displacement_magnitude: float = field(default_factory=lambda: Config._get_displacement())
    
    # 对称性搜索容差
    symmetry_precision: float = field(default_factory=lambda: Config._get_symprec())
    
    @staticmethod
    def _get_displacement() -> float:
        """从环境变量获取有限位移幅度"""
        env_value = os.getenv("FCS_ORDER_H")
        if env_value is not None:
            try:
                return float(env_value)
            except ValueError:
                pass
        return 1e-3  # 默认值
    
    @staticmethod
    def _get_symprec() -> float:
        """从环境变量获取对称性容差"""
        env_value = os.getenv("FCS_ORDER_SYMPREC")
        if env_value is not None:
            try:
                return float(env_value)
            except ValueError:
                pass
        return 1e-5  # 默认值


# 全局默认配置实例
default_config = Config()
