# AlN FC3 validation fixture

`reference.npz` is a derived, compact CI fixture generated from the AlN data
distributed in phono3py's `example/AlN-rd` directory. The upstream dataset is
not copied into this repository because CI does not retrain the potential.

Upstream provenance:

- project: phono3py
- repository: <https://github.com/phonopy/phono3py>
- source path: `example/AlN-rd/phonopy_params_mp-661.yaml.xz`
- source commit: `5d6d3bef5443269295f96dcf8b6c3601364b93ee`
- source SHA-256: `de153514ace4f0828d4111228b20f67fde02dd8bcac7e6c49ad52f24f958007e`
- upstream license: BSD 3-Clause; see `LICENSE.phono3py`
- potential generator: pypolymlp 0.20.4
- reference calculator: phono3py 4.4.0 traditional finite-difference solver

The fixture contains the four-atom wurtzite AlN unit cell, captured forces for
the exact MLFCS sow order, the sow plan hash, and the full phono3py FC3 before
and after explicit traditional `symmetrize_fc3(level=3)` projection for a
2x2x1 supercell. Both calculations use 0.01 Angstrom displacements and the
same pypolymlp potential. The potential itself is not needed by CI.

Derived fixture SHA-256:
`13476da6a52534d5799f092c0605193270aafbe212b2f00387b53f88cc32fa3f`.

Regeneration is intentionally a maintainer operation because fitting the
upstream 200-structure dataset takes several minutes and multiple gigabytes of
temporary memory:

```bash
uv run python tools/generate_AlN_phono3py_fixture.py \
  phonopy_params_mp-661.yaml.xz polymlp.yaml \
  tests/reference/phono3py/AlN_FC3/data/reference.npz
```

CI uses hiphive only as an independent representation adapter. MLFCS does not
depend on hiphive, phono3py, or symfc at runtime.
