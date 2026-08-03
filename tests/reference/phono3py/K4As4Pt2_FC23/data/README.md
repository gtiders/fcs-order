# K4As4Pt2 FC2/FC3 validation fixture

This independent, multi-component reference uses the supplied ten-atom
orthorhombic K4As4Pt2 cell, a 2x2x3 supercell, 0.01 Angstrom central
differences, and the supplied pypolymlp potential with its 8 Angstrom cutoff.

`reference.npz` stores forces in the exact MLFCS sow order and phono3py's raw
`traditional` FC2/FC3 values only at MLFCS sparse clusters. Keeping the sparse
support makes the fixture 7.3 MiB instead of committing the 88 MiB dense FC3
file, and prevents CI from materializing a roughly 356 MiB MLFCS full FC3
array. `POSCAR` and the exact executable `polymlp.yaml` are retained so the
force provenance can be checked directly.

Reference procedure:

- pypolymlp 0.20.4 evaluated all MLFCS displaced structures;
- MLFCS required 24 FC2 and 2328 FC3 force evaluations;
- phono3py 4.4.0 reconstructed full FC2 and FC3 with
  `fc_calculator="traditional"`, `symmetrize_fc=False`;
- phono3py atom order was mapped to MLFCS cell-major order by species,
  fractional position, and the minimum-image convention;
- comparison is restricted to the common 8 Angstrom interaction support.

The supplied final `fc2.hdf5` and `fc3.hdf5` were additionally projected by
symfc in the full dense interaction space. They are intentionally not the
primary numerical oracle: projecting the same raw constants into a full dense
space and an 8 Angstrom sparse space produces different constrained solutions.
The raw comparison isolates the finite-difference implementation; MLFCS ASR is
tested independently elsewhere.

Observed raw sparse-space metrics:

| Order | Maximum absolute difference | RMS difference | Relative L2 difference | Correlation |
|---:|---:|---:|---:|---:|
| 2 | 2.04725e-3 | 7.45151e-5 | 1.31877e-4 | 0.999999992 |
| 3 | 1.06429e-2 | 8.78476e-5 | 3.21657e-4 | 0.999999948 |

Original supplied artifact SHA-256 values (not committed because compact
references are sufficient):

- `fc2.hdf5`: `a31e5ff534b161f42ea0bfb321022d7d91978dc1b8018037829f7e9b23d8d6bf`
- `fc3.hdf5`: `2c8210d071e847c3cfff06f212be75e5fec6f021f574ea44b7bb30cd7c1aad13`
- `phono3py_mlp_eval_dataset.yaml`: `db23dde2e84607562eaaebf420b6200bf7e6276acbd4e2327a82c5dfbd4f7554`

The complete local source directory is ignored by Git, including `POTCAR`.
