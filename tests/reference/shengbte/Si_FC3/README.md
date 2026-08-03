# Si FC3 ShengBTE reference

This directory is being prepared for an external VASP-to-ShengBTE validation.
The structures use the user-provided two-atom diamond-Si `POSCAR` with:

- order 3;
- 3x3x3 supercell (54 atoms);
- sixth-neighbor shell (`cutoff=-6`, resolved radius `6.9007549956` Angstrom);
- 0.01 Angstrom central displacements;
- grouped VASP atom order;
- 168 configurations in exact positional `reap()` order.

`structures/sow-plan.json` records the zero-based configuration ID, plan hash,
filename and SHA-256 of every structure. The corresponding force result for
`POSCAR-001` must be supplied to `reap()` at position zero, and so on.

Regenerate into an empty directory with:

```bash
uv run python tools/generate_Si_shengbte_fc3_sow.py \
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
