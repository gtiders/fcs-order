# K4As4Pt2 FC2/FC3 validation fixture

English | [中文](README_ZH.md)

This independent, multi-component reference uses the supplied ten-atom
orthorhombic K4As4Pt2 cell, a 2x2x3 supercell, 0.01 Angstrom central
differences, and the supplied pypolymlp potential. MLFCS deliberately uses
the supercell's maximum single-atom MIC radius, 12.6461502669 Angstrom,
instead of the potential's 8 Angstrom physical cutoff. This gives the closest
common support to phono3py's untruncated dense arrays.

`reference.npz` stores forces in the exact MLFCS sow order and phono3py's raw
`traditional` and symfc-projected FC2/FC3 values only at MLFCS sparse
clusters. Keeping the reference sparse avoids committing the 88 MiB dense
FC3 file. `POSCAR` and the exact executable `polymlp.yaml` are retained so
the force provenance can be checked directly.

Reference procedure:

- pypolymlp 0.20.4 evaluated all MLFCS displaced structures;
- MLFCS required 24 FC2 and 4244 FC3 force evaluations. The label-symmetric
  pivot selection reuses force responses more efficiently than the earlier
  6636-configuration plan while retaining all 77730 independent FC3
  observations and the same 136260-cluster support;
- phono3py 4.4.0 reconstructed full FC2 and FC3 with
  `fc_calculator="traditional"`, `symmetrize_fc=False`;
- phono3py atom order was mapped to MLFCS cell-major order by species,
  fractional position, and the minimum-image convention;
- comparison uses the common maximum-MIC interaction support;
- MLFCS strict ASR and the supplied symfc full-space ASR projection are both
  tested, including an explicit numerical comparison.

The supplied final `fc2.hdf5` and `fc3.hdf5` were additionally projected by
symfc in the redundant full dense interaction space. For FC3 this projector
simultaneously averages permutation, lattice-translation and space-group
relations before enforcing the sum rule. MLFCS reconstructs directly in an
irreducible symmetry basis and then solves strict ASR there. Both satisfy ASR,
but the different bases and least-change metrics do not define the same unique
projected tensor.

Observed raw sparse-space metrics:

| Order | Maximum absolute difference | RMS difference | Relative L2 difference | Correlation |
|---:|---:|---:|---:|---:|
| 2 | 2.04635e-3 | 5.35e-5 | 1.246e-4 | 0.999999993 |
| 3 | 1.06429e-2 | 4.19509e-5 | 3.50714e-4 | 0.999999939 |

The MLFCS FC3 ASR residual falls from about `1.12e-2` to `2.91e-12`.
The supplied symfc FC3 residual is about `2.62e-6`. On their common sparse
support, the two valid but differently projected ASR solutions have relative
L2 difference `8.65e-2` and correlation `0.99709`.

Original supplied artifact SHA-256 values (not committed because compact
references are sufficient):

- `fc2.hdf5`: `a31e5ff534b161f42ea0bfb321022d7d91978dc1b8018037829f7e9b23d8d6bf`
- `fc3.hdf5`: `2c8210d071e847c3cfff06f212be75e5fec6f021f574ea44b7bb30cd7c1aad13`
- `phono3py_mlp_eval_dataset.yaml`: `db23dde2e84607562eaaebf420b6200bf7e6276acbd4e2327a82c5dfbd4f7554`

The complete local source directory is ignored by Git, including `POTCAR`.

I/O interoperability is checked independently from the numerical solver:

- the default symmetry-closed ShengBTE export is read back through hiphive
  and compared on every one of the 136260 sparse FC3 clusters;
- `phonopy_hdf5` and `phono3py_hdf5` are read through the corresponding
  official readers and compared exactly with the compact MLFCS FC2/FC3;
- the separate Si reference invokes `compatibility="thirdorder"` to preserve
  the legacy joint-image block selection and ordering contract.
