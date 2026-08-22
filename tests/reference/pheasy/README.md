# Pheasy reference benchmarks

[中文](README_ZH.md)

These benchmarks use raw displacement-force data and force-constant files published in the
[Pheasy repository](https://gitlab.com/cplin/pheasy). Pheasy is distributed under GPL-3.0-only.
The external checkout itself is not vendored into MLFCS.

Two comparisons are kept explicit:

- `tests/Si` provides ten 128-atom silicon snapshots and FC2/FC3 reference files. It supports a
  direct repository-data cross-check, but it is not the paper's complete 64-training plus
  64-test FC2--FC6 ensemble.
- `examples/SrTiO3-QE` provides thirty 40-atom structures and an explicit FC2--FC6 setup. Run
  `reference_tools/benchmark_pheasy_fc6.py` to fit those five orders with MLFCS.

The SrTiO3 settings are FC2/FC3 without a finite cutoff, 6 A for FC4--FC6, and maximum body
orders `2, 3, 3, 2, 2`. Pheasy removes an analytical long-range electrostatic force before
fitting this polar material, whereas MLFCS currently fits the supplied total forces. Tensor
differences therefore remain diagnostic; force reconstruction, invariance residuals, and
downstream phonons are the stronger comparisons.

The paper uses larger independent ensembles than these repository fixtures. Its published
errors and transport results are literature baselines, not pass/fail tolerances for this data.

```bash
git clone --depth 1 https://gitlab.com/cplin/pheasy.git
uv run python reference_tools/benchmark_pheasy_fc6.py --pheasy-root pheasy
```
