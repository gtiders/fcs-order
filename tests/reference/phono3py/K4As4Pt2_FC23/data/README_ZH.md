# K4As4Pt2 FC2/FC3 验证夹具

[English](README.md) | 中文

该多组分独立基准使用十原子正交 K4As4Pt2 原胞、2x2x3 超胞、0.01 Angstrom 中心差分
和随仓库保存的 pypolymlp 势。MLFCS 使用超胞最大单原子最小镜像半径
`12.6461502669 Angstrom`，以便与 phono3py 无截断稠密数组比较，而不是使用势函数的
8 Angstrom 物理截断。

`reference.npz` 保存准确 sow 顺序的力，以及 phono3py traditional 和 symfc 投影后的
FC2/FC3 在 MLFCS 稀疏团簇上的值。MLFCS 分别需要 24 个 FC2 和 6636 个 FC3 构型。
原始结果、ASR 结果、官方 HDF5 读回和 ShengBTE 经 hiphive 读回均有独立测试。具体
误差、原始文件 SHA-256 和周期映射方法见[英文说明](README.md)。
