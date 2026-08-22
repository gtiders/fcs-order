# Harmonic finite differences

The case root contains six MLFCS displacements, VASP OUTCAR files, collected
  forces, native `mlfcs.h5`, phonopy `fc2.h5`/`FORCE_CONSTANTS_2ND`, and its
  phonon band.
`primitive.vasp` is the explicit primitive and `supercell.vasp` is the
reference-order authority for structures, forces, and exports. `run.py`
validates the archived displacement plan before reconstructing all results.

`phonopy-reference/` contains the original phonopy displacement YAML,
  displaced structure, VASP `vasprun.xml`, FC2, and its phonon band.

Both directories contain `plot_phonon_band.py` and a supercell in the exact
force-constant atom order. Regenerate the MLFCS plot from this directory with:

```bash
uv run --with phonopy --with seekpath --with matplotlib \
  python plot_phonon_band.py --supercell supercell.vasp \
  --force-constants FORCE_CONSTANTS_2ND --output phonon-band.png
```

Run the same command inside `phonopy-reference/` for its plot.
