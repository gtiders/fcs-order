# Si FC3 ShengBTE 基准

[English](README.md) | 中文

该基准使用两原子金刚石 Si 原胞、3x3x3 超胞、第六近邻壳层（半径
`6.9007549956 Angstrom`）、0.01 Angstrom 中心位移和 grouped VASP 原子顺序，共 168 个
构型。`structures/sow-plan.json` 保存每个 `POSCAR-xxx` 的零基 ID 和文件 SHA-256；
`POSCAR-001` 的力必须对应 `reap()` 的第 0 项。

当前 MLFCS 与旧 thirdorder 都生成 168 个任务，但不可约代表和模板顺序不同，旧
`3RD.POSCAR.NNN` 的力不能直接按位置交给新 API。紧凑的 `data/reference.npz` 用当前
VASP 力与旧 thirdorder 输出比较；周期等价归一化后两者具有相同的 3858 个区块和顺序。
无 ASR/有 ASR 的相对范数差分别约为 `0.693%`/`0.633%`。原始 VASP 目录被 Git 忽略，
完整来源、误差和再生成命令见[英文说明](README.md)。
