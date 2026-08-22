# AlN FC2 验证夹具

[English](README.md) | 中文

`reference.npz` 比较 MLFCS 二阶中心差分与 phonopy traditional FC2 solver。它与 AlN
FC3 基准使用同一训练集和 pypolymlp 势、2x2x2 超胞、0.01 Angstrom 位移，并覆盖全部
最小镜像原子对。MLFCS 计算 12 个中心差分构型，phonopy 计算 4 个对称性筛选构型，
因此测试比较最终 FC2，而不要求位移计划相同。精确 SHA-256 和再生成命令见
[英文说明](README.md)。
