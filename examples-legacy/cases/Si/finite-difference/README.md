# Silicon finite differences

`harmonic/` compares the complete MLFCS and phonopy FC2 paths, including their
ordered displacement structures, readable VASP result, force constants,
plotting script, and generated band plot. `three-phonon/` does the same for
MLFCS FC3 and thirdorder; all 132 thirdorder `vasprun.xml` files are retained.
`thermal-conductivity/` connects each FC2+FC3 pair to its compact ShengBTE
inputs and final temperature-dependent conductivity tables.

For MLFCS, the primitive has two atoms and the 4 x 4 x 4 reference supercell
has 128 atoms. FC2 uses six force calculations without a cutoff. FC3 uses 132
force calculations with cutoff `-5`. Both use 0.01 Angstrom displacements and
ASR. `harmonic/run.py` and `three-phonon/run.py` reconstruct results from the
collected `forces.npz` files and validate the archived sow order.

Only one ASE-readable force output is kept per VASP job. MLFCS uses the smaller
`OUTCAR`; phonopy uses its smaller `vasprun.xml`; thirdorder deliberately keeps
`vasprun.xml`. The checked-in `VASP_INPUT` directories document the run
settings; all other electronic scratch files are excluded.
