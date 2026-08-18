# K4As4Pt2 finite differences with Polymlp

This case compares two finite-difference implementations using the same
orthorhombic 10-atom primitive, 2 x 2 x 3 supercell, 0.01 Angstrom
displacements, no FC2 cutoff, a 12 Bohr FC3 cutoff, and the checked-in
`polymlp.yaml` potential through ASE:

- `harmonic/` and `three-phonon/` are MLFCS calculations. They retain native
  `mlfcs.h5`, phonopy `fc2.h5`/`FORCE_CONSTANTS_2ND`, and ShengBTE
  `FORCE_CONSTANTS_3RD`. The large FC3 is intentionally kept sparse in the
  native file; the compact phono3py FC3 comes from the comparison route.
- `phono3py-reference/` uses phono3py's traditional finite-difference route
  with the same ASE calculator and retains its force archive and compact FC2/
  FC3 HDF5 files.
- `thermal-conductivity/mlfcs-phono3py/` is populated after the MLFCS results
  exist and runs phono3py RTA. Its ShengBTE conductivity can be added later.

Run the two routes together with (a saved `forces.npz` is reused automatically):

```bash
uv run --with pypolymlp --with phono3py python run.py --route both
```

The phono3py route intentionally uses its own displacement enumeration, so
the number and order of force evaluations need not equal MLFCS. The potential
and structures are the common physical inputs; comparison should align the
resulting primitive/supercell mappings before comparing tensors.
