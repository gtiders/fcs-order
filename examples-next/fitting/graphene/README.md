# 石墨烯旋转约束案例

`input/phonopy_snapshot.extxyz` 是一个 phonopy 位移快照，脚本从其中恢复参考超胞并拟合
二阶模型。结果分别保存在 `results/asr/` 和 `results/born-huang-huang/`，用于比较 ASR
以及 Born-Huang/Huang 后处理对二维材料声子的影响。

运行 `python run.py`。`reference/` 中保留原始 hiPhive 对比脚本和图，仅作为第三方参考。
