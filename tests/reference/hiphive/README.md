# hiPhive public-example benchmark

This optional benchmark uses the public Si and BaGaGe clathrate datasets from
`materials-modeling/hiphive-examples`. The data are not vendored because the
repository contains about 114 MB of Git LFS objects.

## Data acquisition

```bash
git clone --depth 1 https://gitlab.com/materials-modeling/hiphive-examples.git
git -C hiphive-examples lfs pull
git -C hiphive-examples rev-parse HEAD
```

The recorded comparison used commit
`05216055abca04ef9476bb9a5ba5b0f050993b2d` (hiPhive examples generated with
hiPhive 0.5). Runtime comparison uses the installed hiPhive 1.5 and
TrainStation 1.2, while the datasets and model definitions remain those of the
published example.

## Reproduction

Run one memory-intensive process at a time:

```bash
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py si-mlfcs
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py si-hiphive
uv run python reference_tools/benchmark_hiphive_examples.py si-compare
uv run python reference_tools/benchmark_hiphive_examples.py bagage-wick
```

The Si comparison uses all 20 Si250 snapshots, FC2+FC3 cutoffs of 9.65 Å,
least squares, and translational ASR. Both implementations produce 150 orbits,
2598 unconstrained parameters, 77 independent ASR constraints, and therefore
2521 constrained degrees of freedom.

## Recorded results

| Quantity | MLFCS | hiPhive |
|---|---:|---:|
| Force RMSE (meV/Å) | 3.131365 | 3.131190 |
| Relative force error | 0.949040% | 0.948987% |
| Wall time | 427.79 s | 251.12 s |
| Peak RSS | 2,164,816 KiB | 1,926,200 KiB |

Aligned tensor comparisons give relative RMS differences of
`1.31e-5` for FC2 and `7.60e-4` for FC3. Atom indices and tensor axes are
explicitly aligned before comparison; serialized file order is not compared.

For the 200-snapshot BaGaGe dataset, 8192 reproducibly sampled cubic features
give the following absolute linear/cubic feature correlations:

| Statistic | Taylor cubic | Wick cubic |
|---|---:|---:|
| Mean | 0.06409 | 0.06280 |
| RMS | 0.08094 | 0.07867 |
| 95th percentile | 0.15781 | 0.15399 |
| 99th percentile | 0.21127 | 0.20241 |
| Maximum | 0.68239 | 0.40341 |

Thus Wick substantially suppresses the worst FC2/FC4 feature correlations but
does not eliminate finite-sample, non-Gaussian correlations. This is a sampled
raw-feature diagnostic, not a claim that every constrained physical-design
singular value improves.

Compact periodic-coordinate storage and pre-Gram constraint parameterization reduce the MLFCS
peak RSS by 56% relative to its earlier 4,951,592 KiB implementation while preserving the force
and tensor results. The remaining runtime difference is primarily physical-design evaluation, not
the Gram solve.

The published BaGaGe model uses 200 structures, cutoffs
`[5.4, 4.35, 4.35] Å`, two-body support, 10-fold OLS, and reports 6052
parameters, 48.17 meV/Å training RMSE, and 69.67 meV/Å validation RMSE. A full
MLFCS FC2+FC3+FC4 run is intentionally not part of the routine test suite: its
physical parameterization has 25,495 coefficients, but the new block-sparse ASR map reduces it to
6052 fitting coordinates before Gram construction. The resulting Gram requires about 279 MiB
instead of 4.84 GiB; a complete 200-structure run remains an opt-in benchmark rather than a CI job.
