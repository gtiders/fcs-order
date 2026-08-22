# MLFCS and phonopy KCl SSCHA comparison

English | [中文](COMPARISON_ZH.md)

## Purpose

This comparison tests the complete native MLFCS path with a real anharmonic machine-learning
potential maintained in phonopy's own test suite. It is more demanding than the analytic harmonic
sampling test: the potential supplies nonlinear energies and forces, MLFCS performs stochastic
sampling and constrained FC2 fitting, and the resulting tensor and free energy are compared with
phonopy's accepted values.

## Provenance

The artifacts come from the official phonopy repository at commit
`fb63319c071f264e01e1cd4d85a81526c6c7a40a` under BSD-3-Clause:

- `test/polymlp_KCL-120.yaml`;
- `test/phonopy_KCl.yaml`;
- `example/KCl-SSCHA/phonopy_sscha_fc_JPCM2022.yaml.xz`.

The pypolymlp potential was trained from 120 randomly displaced KCl structures. Exact hashes and
the upstream license are stored in [`data/`](data/).

## Common physical setup

| Setting | MLFCS | phonopy reference |
|---|---:|---:|
| Material | KCl | KCl |
| Input cell | Eight-atom conventional cell | Eight-atom conventional cell |
| Supercell | 2x2x2, 64 atoms | 2x2x2, 64 atoms |
| Temperature | 300 K | 300 K |
| Root random seed | 42 | 42 |
| Potential | Same `polymlp_KCL-120.yaml` | `polymlp_KCL-120.yaml` |
| Displacement limit | None | None |
| Statistics | Quantum | Quantum |

## Deliberately different numerical workload

The upstream phonopy test uses 50 snapshots in each iteration and three canonical iterations.
Running that full workload repeatedly in the current WSL environment caused the process to be
terminated under accumulated JAX and pypolymlp memory pressure. The serial CI reference therefore
uses ten snapshots and one canonical iteration. This is sufficient to test the complete API and
physical scale without presenting it as a converged production calculation.

There is also a harmonic-free-energy convention difference. Phonopy evaluates the harmonic term
on a dense reciprocal-space mesh. The current native MLFCS implementation evaluates the q points
commensurate with the sampling supercell. FC2 comparison is therefore the cleaner primary check;
the free-energy comparison uses a tolerance that includes finite sampling and q-mesh effects.

## Results

| Quantity | MLFCS CI reference | phonopy test reference | Interpretation |
|---|---:|---:|---|
| Initialization K self FC2 | `1.9042 eV/Angstrom^2` | Not an upstream acceptance target | Stable initial fit |
| Canonical K self FC2 | `2.1625 eV/Angstrom^2` | `2.1 +/- 0.1 eV/Angstrom^2` | Inside the official range |
| Free energy per primitive cell | `-0.0949 eV` | `-0.0986 +/- 0.001 eV` | Difference about `3.7 meV` |

The MLFCS input object is the conventional cell containing four primitive cells. Its reported
free energy is therefore divided by four before comparison with phonopy's per-primitive-cell
value.

## Semantics confirmed by the comparison

- Cartesian initialization does not define an SSCHA free energy because its configurations are
  not sampled from the fitted harmonic Hamiltonian. MLFCS therefore reports `None` for that round.
- Each canonical iteration derives a distinct child seed from the root seed and iteration index.
  Repeated runs remain reproducible, while different iterations do not reuse the same random
  stream.
- No maximum-displacement clipping is applied by default, so the sampled distribution is not
  silently truncated.

## What this test proves

The test demonstrates that the official nonlinear KCl potential can pass through the native
MLFCS stochastic sampling, ASE calculator, Gram FC2 fitting, ASR, and free-energy path and recover
the FC2 scale accepted by phonopy.

It does not prove fully converged equality of the two SSCHA implementations. Such a claim would
require matching snapshot counts, iteration convergence, reciprocal meshes, fit constraints, and
statistical error bars. The bundled published FC2 remains available for future band-structure and
mode-resolved comparisons.
