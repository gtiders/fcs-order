# phonopy KCl SSCHA 参考

[English](README.md) | 中文

本文件只记录夹具溯源。数值结果与解释单独保存在
[`../COMPARISON_ZH.md`](../COMPARISON_ZH.md)。

本夹具来自 phonopy 官方仓库提交
`fb63319c071f264e01e1cd4d85a81526c6c7a40a`（BSD-3-Clause）：

- `test/polymlp_KCL-120.yaml` 保存为 `polymlp.yaml`；
- `test/phonopy_KCl.yaml`；
- `example/KCl-SSCHA/phonopy_sscha_fc_JPCM2022.yaml.xz`。

该势函数由 120 个 KCl 常规胞 2x2x2 随机位移超胞训练。phonopy 官方 SSCHA 测试在
300 K 使用随机种子 42，每轮 50 个快照并执行 3 轮 canonical 迭代；其验收范围为
K 原子自相互作用块 `2.1 +/- 0.1 eV/Angstrom^2`，以及每个原胞
`-0.0986 +/- 0.001 eV` 的自由能。

MLFCS 的串行 CI 参考为控制内存，使用 10 个快照和 1 轮 canonical 迭代，但常规胞、
2x2x2 超胞、温度、种子与势函数均与上游一致。测试检查初始化 FC2、canonical FC2、
phonopy 的 FC2 验收范围，以及将常规胞结果归一化到单个原胞后的自由能。自由能
容差还包含一项明确的方法差别：phonopy 在稠密倒空间网格上计算谐振项，当前 MLFCS
使用与超胞相容的 q 点。

初始化轮的笛卡尔随机位移不是从拟合后的谐振哈密顿量采样，因此不报告 SSCHA
自由能；这与 phonopy 的约定一致。

上游 BSD-3-Clause 声明保存在 `LICENSE.phonopy`。
