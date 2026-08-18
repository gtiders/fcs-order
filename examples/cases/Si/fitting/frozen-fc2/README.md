# Si fitting with frozen FC2

This case reuses the 100 ALAMODE Si displacement-force snapshots in
`../anharmonic/train.extxyz` and fits residual FC3-FC4 while FC2 remains an
external physical Taylor baseline.

The compatible baseline is the independent harmonic fit:

```bash
uv run python run.py --baseline harmonic-fit
```

Results are written below `harmonic-fit/`. The frozen FC2 is copied exactly
after strict structure alignment; ASR is applied only to the fitted residual.
The checked run converged in 557 iterations with a 3.135438% relative training
force error and a 0.00351696 eV/Angstrom force RMSE. The reloaded output FC2
has exactly identical lattice labels and tensor bytes to the frozen input.
`fit-summary.json` records the numerical diagnostics.

The repository's VASP finite-difference FC2 cannot be used with this training
set. It belongs to a strained 128-atom `4x4x4` primitive supercell, whereas the
ALAMODE fitting data use an unstrained 64-atom conventional reference. This
command records the expected validation failure without writing partial IFCs:

```bash
uv run python run.py --baseline finite-difference
```

MLFCS deliberately does not strain, shrink, periodically fold, or project a
frozen IFC to make incompatible inputs appear compatible. A genuine second
comparison requires recomputing finite-difference FC2 on the exact
`../anharmonic/primitive.vasp` and `../anharmonic/supercell.vasp` structures.
