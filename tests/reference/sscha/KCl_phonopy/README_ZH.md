# KCl SSCHA 对比

[English](README.md) | 中文

本目录是原生 MLFCS SSCHA 与 phonopy 官方 KCl SSCHA 参考之间的独立端到端对比。

- [`COMPARISON_ZH.md`](COMPARISON_ZH.md) 记录数值条件、结果、解释和局限；
- [`data/`](data/) 保存锁定的上游势函数、结构、已发表 FC2、许可证和溯源说明；
- `test_kcl_potential.py` 将官方势函数接入 MLFCS，检查物理量级；
- `test_provenance.py` 通过 SHA-256 校验所有导入夹具；
- `case.py` 定义对比使用的精确 KCl 常规胞。

该参考应独立运行，避免 pypolymlp 和 JAX 与其他参考在同一进程累积内存：

```bash
uv run pytest tests/reference/sscha/KCl_phonopy
```
