"""
计算器工厂模块

统一创建和管理各类 ASE 计算器
"""

from pathlib import Path
from typing import Union, Optional, Any

from ase.calculators.calculator import Calculator


class CalculatorFactory:
    """
    计算器工厂类
    
    提供统一的接口来创建各种机器学习势函数计算器
    """
    
    @staticmethod
    def create(
        calculator_type: str,
        potential_path: Union[str, Path],
        **kwargs: Any,
    ) -> Calculator:
        """
        创建指定类型的计算器
        
        Args:
            calculator_type: 计算器类型 (nep, dp, mtp, polymlp, tace, hiphive)
            potential_path: 势函数文件路径
            **kwargs: 额外参数
            
        Returns:
            ASE Calculator 对象
        """
        creators = {
            "nep": CalculatorFactory._create_nep,
            "dp": CalculatorFactory._create_dp,
            "mtp": CalculatorFactory._create_mtp,
            "polymlp": CalculatorFactory._create_polymlp,
            "tace": CalculatorFactory._create_tace,
        }
        
        if calculator_type not in creators:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
        
        return creators[calculator_type](potential_path, **kwargs)
    
    @staticmethod
    def _create_nep(
        potential_path: Union[str, Path],
        is_gpu: bool = False,
        **kwargs: Any,
    ) -> Calculator:
        """创建 NEP 计算器"""
        try:
            if is_gpu:
                from pynep.calculate import NEP as NEP_GPU
                return NEP_GPU(str(potential_path))
            else:
                from calorine.calculators import CPUNEP
                return CPUNEP(str(potential_path))
        except ImportError as e:
            raise ImportError(
                "NEP calculator requires 'pynep' (GPU) or 'calorine' (CPU) package"
            ) from e
    
    @staticmethod
    def _create_dp(
        potential_path: Union[str, Path],
        **kwargs: Any,
    ) -> Calculator:
        """创建 DeepMD 计算器"""
        try:
            from deepmd.calculator import DP
            return DP(model=str(potential_path))
        except ImportError as e:
            raise ImportError(
                "DeepMD calculator requires 'deepmd-kit' package"
            ) from e
    
    @staticmethod
    def _create_mtp(
        potential_path: Union[str, Path],
        mtp_exe: str = "mlp",
        unique_elements: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Calculator:
        """创建 MTP 计算器"""
        try:
            from mlip_ase import MTP
            return MTP(
                pot=str(potential_path),
                mlp_command=mtp_exe,
                elements=unique_elements,
            )
        except ImportError as e:
            raise ImportError(
                "MTP calculator requires 'mlip_ase' package"
            ) from e
    
    @staticmethod
    def _create_polymlp(
        potential_path: Union[str, Path],
        **kwargs: Any,
    ) -> Calculator:
        """创建 PolyMLP 计算器"""
        try:
            from pypolymlp.api import Pypolymlp
            polymlp = Pypolymlp()
            polymlp.load_mlp(str(potential_path))
            return polymlp.get_calculator()
        except ImportError as e:
            raise ImportError(
                "PolyMLP calculator requires 'pypolymlp' package"
            ) from e
    
    @staticmethod
    def _create_tace(
        potential_path: Union[str, Path],
        device: str = "cuda",
        dtype: Optional[str] = "float32",
        level: int = 0,
        **kwargs: Any,
    ) -> Calculator:
        """创建 TACE 计算器"""
        try:
            from tace.mce_ase import MCE
            return MCE(
                model_path=str(potential_path),
                device=device,
                default_dtype=dtype,
                level=level,
            )
        except ImportError as e:
            raise ImportError(
                "TACE calculator requires 'tace' package"
            ) from e
