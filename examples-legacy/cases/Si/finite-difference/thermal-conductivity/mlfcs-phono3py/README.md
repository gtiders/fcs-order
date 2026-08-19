# MLFCS FC2/FC3 in phono3py RTA

`fc2.h5` and `fc3.h5` are the phonopy/phono3py-compatible force constants.
`supercell.vasp` is the 128-atom reference supercell. The wrapper
does not supply a primitive or reorder atoms: phono3py discovers the primitive
from the supercell and reads the HDF5 mapping stored with the force constants.

Run from the repository root:

```bash
uv run --with phono3py python examples/cases/Si/finite-difference/thermal-conductivity/mlfcs-phono3py/run_rta.py \
  --mesh 11 11 11 --temperatures 300
```

The `--br` flag selects phono3py's BTE-RTA path. `kappa-*.hdf5`, YAML and
diagnostic files emitted by phono3py remain in this directory. This is an
end-to-end example, not a strict CI oracle; record phono3py version, mesh,
smearing and isotope settings with any reported conductivity.
