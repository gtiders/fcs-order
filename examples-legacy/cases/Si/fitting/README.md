# ALAMODE Si data as MLFCS fitting input

The displacement-force data originate from
`alamode/example/Si/anharm_IFCs`. They are retained solely as a source of
training configurations for MLFCS. ALAMODE XML and fitted force constants are
deliberately excluded: this case does not claim a numerical comparison against
ALAMODE.

`source/prepare_dataset.py` converts ALAMODE DFSET displacements from Bohr and
forces from Rydberg/Bohr to Angstrom and eV/Angstrom. The checked-in extxyz
files are the inputs consumed by MLFCS and preserve the reference-supercell
atom order.

The harmonic and anharmonic fits are independent. Each directory keeps its
own primitive/reference structures, strict `train.extxyz`, cache, native
`mlfcs.h5`, and format-specific exports:

- `harmonic/`: FC2 in phonopy text and HDF5 formats;
- `anharmonic/`: FC2, FC3, and FC4, including ShengBTE text for the latter
  two orders and phono3py HDF5 for FC2/FC3;
- `frozen-fc2/`: residual FC3-FC4 fitting with an external FC2 baseline,
  including the compatible harmonic-fit run and the documented rejection of
  the structurally incompatible archived finite-difference FC2;
- `thermal-conductivity/`: a phono3py RTA run using the anharmonic HDF5 files.

Run the independent fits with:

```bash
uv run python harmonic/run.py
uv run python anharmonic/run.py
```

The harmonic case contains one displaced 64-atom configuration and fits FC2.
The anharmonic case contains 100 64-atom random configurations and jointly
fits FC2-FC4. Its FC4 cutoff is 11 Bohr and the body-order limits reproduce the
source setup where applicable, but MLFCS uses its own symmetry basis and
regularized solver. The thermal-conductivity wrapper uses the reference
supercell directly; phono3py discovers the primitive from it and runs an
11x11x11 RTA mesh by default.
