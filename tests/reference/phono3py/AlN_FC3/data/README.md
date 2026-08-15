# AlN FC3 validation fixture

English | [中文](README_ZH.md)

`reference.npz` is the compact fixture used by CI. The complete upstream AlN
training dataset and the exact trained pypolymlp potential used to derive it
are preserved in `training/`, but ordinary CI does not retrain or reevaluate
the potential.

Directory contents:

- `reference.npz`: forces and raw/ASR-projected FC3 used by CI;
- `training/phonopy_params_mp-661.yaml.xz`: upstream 200-structure training dataset;
- `training/polymlp.yaml`: exact trained potential used to generate the fixture;
- `LICENSE.phono3py`: license covering the redistributed upstream dataset.

Upstream provenance:

- project: phono3py
- repository: <https://github.com/phonopy/phono3py>
- source path: `example/AlN-rd/phonopy_params_mp-661.yaml.xz`
- source commit: `5d6d3bef5443269295f96dcf8b6c3601364b93ee`
- source SHA-256: `de153514ace4f0828d4111228b20f67fde02dd8bcac7e6c49ad52f24f958007e`
- upstream license: BSD 3-Clause; see `LICENSE.phono3py`
- potential generator: pypolymlp 0.20.4
- potential SHA-256: `cb81eb864fdc29e6f725d6ac9ec41b043beeadc073416d42fb75e3728ce415ec`
- reference calculator: phono3py 4.4.0 traditional finite-difference solver

The pypolymlp fit used the upstream deterministic 180/20 train/test split and
selected ridge parameter `alpha=0.1`. Reported errors were:

| Dataset | Energy RMSE (meV/atom) | Force RMSE (eV/Angstrom) |
|---|---:|---:|
| Train | 0.00041 | 0.00026 |
| Test | 0.00038 | 0.00029 |

The fixture contains the four-atom wurtzite AlN unit cell, captured forces for
the current exact MLFCS sow order (508 configurations) and the full phono3py FC3 before
and after explicit traditional `symmetrize_fc3(level=3)` projection for a
2x2x2 supercell. Both calculations cover every MIC atom pair in that
supercell. MLFCS uses a direct radius of `5.8760168278` Angstrom (the maximum
MIC distance `5.8760158278` Angstrom plus a numerical margin of `1e-6`
Angstrom), while phono3py uses its default without `cutoff_pair_distance`.
Both use 0.01 Angstrom displacements and the same pypolymlp potential. This is
a full-supercell comparison, not a cutoff-convergence claim for the infinite
crystal. The potential itself is not needed by CI.

Derived fixture SHA-256:
`72eca945495dfd8fa147a82016a30eb50881cb279df2e80a8cbe5b97120e62f2`.

Regeneration is intentionally a maintainer operation because fitting the
upstream 200-structure dataset takes several minutes and multiple gigabytes of
temporary memory:

```bash
uv run python reference_tools/generate_AlN_phono3py_fixture.py \
  tests/reference/phono3py/AlN_FC3/data/training/phonopy_params_mp-661.yaml.xz \
  tests/reference/phono3py/AlN_FC3/data/training/polymlp.yaml \
  tests/reference/phono3py/AlN_FC3/data/reference.npz
```

CI uses hiphive only as an independent representation adapter. MLFCS does not
depend on hiphive, phono3py, or symfc at runtime.
