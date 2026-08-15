# AlN FC3 验证夹具

[English](README.md) | 中文

`reference.npz` 是 CI 使用的紧凑夹具；`training/` 保存 phono3py 官方 200 构型训练集和
实际使用的 pypolymlp 0.20.4 `polymlp.yaml`，普通 CI 不重新训练势函数。上游文件来自
phono3py 提交 `5d6d3bef5443269295f96dcf8b6c3601364b93ee` 的
`example/AlN-rd/phonopy_params_mp-661.yaml.xz`，许可见 `LICENSE.phono3py`；完整 SHA-256
和拟合误差记录在[英文来源说明](README.md)中。

夹具包含 2x2x2 超胞、0.01 Angstrom 位移下当前准确 sow 顺序的 508 个 MLFCS 构型力，以及 phono3py
4.4.0 traditional solver 在 ASR 前后的完整 FC3。双方覆盖全部最小镜像原子对。
CI 使用 hiphive 只是为了规范化张量表示，MLFCS 运行时不依赖 hiphive、phono3py 或
symfc。再生成命令和精确校验和见英文说明。
