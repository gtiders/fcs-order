# Three-phonon finite differences

- The case root contains the 132 ordered MLFCS FC3 displacements, compact VASP
  OUTCAR files, collected forces, native HDF5, phono3py HDF5, and ShengBTE FC3.
  `primitive.vasp` and `supercell.vasp` define the exact structure relation;
  `run.py` validates every archived displacement before reconstruction.
- `thirdorder-reference/` preserves the primitive and reference supercell, all
  132 thirdorder displacement POSCAR files, all 132 matching `vasprun.xml`
  files, the sow log, and the resulting ShengBTE FC3.

`hiphive-reference/FORCE_CONSTANTS_3RD` is the historical conversion
of the MLFCS phono3py HDF5 through hiphive. It is retained because comparison
with the old direct ShengBTE export exposed the periodic-image writer defect;
it is not a general numerical oracle.
