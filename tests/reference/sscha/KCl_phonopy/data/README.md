# Phonopy KCl SSCHA reference

English | [中文](README_ZH.md)

This file documents artifact provenance. Numerical results and interpretation are kept separately
in [`../COMPARISON.md`](../COMPARISON.md).

This fixture is copied from the official phonopy repository at commit
`fb63319c071f264e01e1cd4d85a81526c6c7a40a` (BSD-3-Clause):

- `test/polymlp_KCL-120.yaml` -> `polymlp.yaml`;
- `test/phonopy_KCl.yaml`;
- `example/KCl-SSCHA/phonopy_sscha_fc_JPCM2022.yaml.xz`.

The potential was trained from 120 randomly displaced 2x2x2 conventional KCl supercells. The
upstream phonopy SSCHA test uses 50 snapshots per iteration, three canonical iterations, 300 K,
and seed 42. It accepts a K self-interaction block of `2.1 +/- 0.1 eV/Angstrom^2` and a free
energy of `-0.0986 +/- 0.001 eV` per primitive cell.

The serial MLFCS reference intentionally uses 10 snapshots and one canonical iteration to keep
CI memory bounded. It uses the identical eight-atom conventional cell, 2x2x2 supercell,
temperature, seed, and potential. The test checks the initialization tensor, canonical tensor,
phonopy's accepted FC2 scale, and free energy after normalization from the conventional input
cell to one primitive cell. The free-energy tolerance also covers a deliberate method difference:
phonopy evaluates its harmonic term on a dense reciprocal mesh, whereas the current native MLFCS
sampler evaluates the commensurate supercell q points.

The initialization round has no SSCHA free energy because its Cartesian displacements are not
drawn from the fitted harmonic Hamiltonian. This follows the same convention as phonopy.

The original BSD-3-Clause notice is retained in `LICENSE.phonopy`.
