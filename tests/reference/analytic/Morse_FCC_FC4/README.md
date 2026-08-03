# FCC Morse FC4 analytic reference

English | [中文](README_ZH.md)

This benchmark validates fourth-order force constants without using another force-constant
fitter. ASE relaxes a one-component FCC cell with `MorsePotential`; an independent JAX
implementation differentiates the Morse pair energy four times, and MLFCS samples forces from
the ASE calculator by central finite differences.

The ASE element label is Ar, but the parameters define a reduced-unit numerical benchmark and
are not a fitted argon model:

| parameter | value |
| --- | ---: |
| `epsilon` | 1.0 eV |
| `rho0` | 6.0 |
| `r0` | 1.0 Å |
| `rcut1`, `rcut2` | 1.15 Å, 1.30 Å |
| supercell | 3x3x3 |
| MLFCS cutoff | 1.1 Å |

The switching interval lies above the FCC nearest-neighbour distance and below the
second-neighbour distance. Consequently the cutoff is constant for all contributing bonds near
equilibrium. With nearest-neighbour interactions only, the analytic conventional lattice
constant is `sqrt(2) r0` and the cohesive energy is `-6 epsilon` per atom.

The exact oracle fixes the periodic nearest-neighbour bond list and evaluates

```text
V(r) = epsilon exp[rho0 (1-r/r0)] {exp[rho0 (1-r/r0)] - 2}
```

with four nested `jax.jacfwd` operations. It then assembles the derivative into the same sparse
atomic clusters using only bond incidence and endpoint signs. This path does not use MLFCS
finite-difference stencils, symmetry reconstruction, or ASR. The test also requires a factor of
four error reduction when the displacement is halved, which is the expected second-order
convergence of the central-difference construction.

Run independently with:

```bash
uv run pytest tests/reference/analytic/Morse_FCC_FC4/test_morse_fc4.py
```
