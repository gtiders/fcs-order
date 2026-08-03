# AlN FC2 validation fixture

`reference.npz` compares MLFCS order-2 finite differences with phonopy's
traditional FC2 solver. It uses the same upstream AlN dataset and the same
trained pypolymlp potential as the sibling phono3py FC3 validation:

- dataset: `tests/reference/phono3py/AlN_FC3/data/training/phonopy_params_mp-661.yaml.xz`;
- potential: `tests/reference/phono3py/AlN_FC3/data/training/polymlp.yaml`;
- supercell: 2x2x2;
- displacement: 0.01 Angstrom;
- MLFCS cutoff: `5.8760168278` Angstrom, covering every MIC atom pair;
- phonopy pair cutoff: none (the default).

MLFCS evaluates 12 central-difference force configurations. Phonopy evaluates
four symmetry-selected configurations and reconstructs the complete FC2 with
its traditional solver. Consequently the comparison validates the final force
constants rather than requiring identical displacement plans.

Derived fixture SHA-256:
`677f95b8fa8018fa3b5d43add18b1b11ed2f33643f962c66d0e3dcb36ae8c45c`.

Regenerate with:

```bash
uv run python tools/generate_AlN_phonopy_fc2_fixture.py \
  tests/reference/phono3py/AlN_FC3/data/training/phonopy_params_mp-661.yaml.xz \
  tests/reference/phono3py/AlN_FC3/data/training/polymlp.yaml \
  tests/reference/phonopy/AlN_FC2/data/reference.npz
```
