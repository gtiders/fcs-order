# Si FC3 ShengBTE reference

English | [中文](README_ZH.md)

This directory is being prepared for an external VASP-to-ShengBTE validation.
The structures use the user-provided two-atom diamond-Si `POSCAR` with:

- order 3;
- 3x3x3 supercell (54 atoms);
- sixth-neighbor shell (`cutoff=-6`, resolved radius `6.9007549956` Angstrom);
- 0.01 Angstrom central displacements;
- grouped VASP atom order;
- 168 configurations in exact positional `reap()` order.

`structures/sow-plan.json` records the zero-based configuration ID,
filename and SHA-256 of every structure. The corresponding force result for
`POSCAR-001` must be supplied to `reap()` at position zero, and so on.

Regenerate into an empty directory with:

```bash
uv run python reference_tools/generate_Si_shengbte_fc3_sow.py \
  POSCAR tests/reference/shengbte/Si_FC3/structures
```

The current implementation and the old package both request 168 calculations,
but their irreducible representatives and positional stencil order differ.
Therefore old `3RD.POSCAR.NNN` force files must not be fed positionally into the
new API.

## Local raw VASP provenance

The cleaned raw calculations are kept locally under the ignored directory
`data/vasp/`:

- `data/vasp/mlfcs`: the current MLFCS sow plan;
- `data/vasp/thirdorder`: the original thirdorder plan and its exported
  `FORCE_CONSTANTS_3RD`.

Both contain `calculations/000` for the undisplaced supercell and numbered
directories `001` through `168` for displaced structures. Each calculation
retains only `POSCAR` and `vasprun.xml`; common `INCAR` and `KPOINTS` files are
stored once at the method root. POTCAR and nonessential VASP outputs are not
retained. Raw XML is intentionally excluded from Git; compact derived fixtures
belong directly in `data/`.

## Numerical comparison

`data/reference.npz` is the 315 KiB CI fixture derived from the 168 current
MLFCS `vasprun.xml` files and the original thirdorder `FORCE_CONSTANTS_3RD`.
It contains grouped-order forces, periodic-canonical block
translations, primitive atom indices, FC3 values, and source hashes.

Both exports contain 3858 blocks. Their raw text chooses a different but
periodically equivalent supercell image for 173 blocks. After reducing lattice
translations modulo the 3x3x3 supercell, every block key appears in exactly the
same order. The numerical results are:

| MLFCS reconstruction | Maximum difference (eV/Angstrom^3) | RMS difference | Relative norm | Correlation |
|---|---:|---:|---:|---:|
| ASR disabled | 0.08693 | 0.003966 | 0.693% | 0.999976 |
| ASR enabled | 0.08675 | 0.003619 | 0.633% | 0.999980 |

The independent calculations use different symmetry-equivalent displacement
representatives, so they do not share identical raw forces. The test therefore
requires exact physical block ordering and bounded FC3 error, not byte-identical
floating-point output.

Fixture SHA-256:
`a6b6f36e416145dbd59d93dd30e0f8105f96065bf453e24d228bd82c4846b44f`.

Regenerate while the ignored raw VASP directories are available:

```bash
uv run python reference_tools/generate_Si_shengbte_fixture.py \
  tests/reference/shengbte/Si_FC3/data/vasp/mlfcs \
  tests/reference/shengbte/Si_FC3/data/vasp/thirdorder \
  tests/reference/shengbte/Si_FC3/data/reference.npz
```
