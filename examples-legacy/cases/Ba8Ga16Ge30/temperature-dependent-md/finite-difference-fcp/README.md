# Finite differences from the public hiPhive FCP

This case evaluates the public
`fcp_2body-5.4_4.35_4.35_least-squares.fcp` through its ASE calculator and
reconstructs force constants with MLFCS finite differences. The primitive is
the 54-atom `input/reference.vasp`; the reference cell is its `2x2x2` repeat.
Both orders use the upstream two-body restriction: a `5.40 A` FC2 cutoff,
`4.35 A` FC3 cutoff, and `max_body_order=2`.

Run both orders with:

```bash
uv run python run.py
```

Force evaluations are cached in `harmonic/forces.npz` and
`three-phonon/forces.npz`. Outputs are the generic MLFCS HDF5 file, phonopy
FC2 text, and ShengBTE FC3 text. Phono3py-specific `fc2.h5`/`fc3.h5` files are
intentionally not generated in this case.
