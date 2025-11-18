# [中文文档 | Chinese README](README_zh.md)

# fcs-order (English Guide)

Tip: If this project helps you, please star the repo. Please read this README first; if you still have issues or suggestions, open an Issue with reproduction steps and environment info.

Note: CLI help evolves with versions. Use `fcs-order --help` and `fcs-order <subcommand> --help` for the most accurate, up‑to‑date options and examples.

Note: The `sow` and `reap` commands are compatible with traditional `thirdorder`/`fourthorder` tools; argument names and I/O conventions are slightly adjusted for a unified interface.

Force-constants and SCPH toolkit built on ASE and Typer. Compute 2nd/3rd/4th-order interatomic force constants and run SCPH workflows using various ML potentials (NEP, DeepMD, PolyMLP, MTP, TACE) or Hiphive.

Repo: <https://github.com/gtiders/fcs-order>

---

## Features

- 2nd-order force constants (compatible with Phonopy `FORCE_CONSTANTS`)
- 3rd/4th-order force constants (ShengBTE/text formats)
- SCPH (self-consistent phonon) workflows
- Multiple backends: NEP, DeepMD, PolyMLP, MTP, TACE, and Hiphive
- VASP-oriented sow (generate displacements) and reap (reconstruct IFCs)
- Phonon-based and Monte Carlo structure generators: `phonon_rattle` and `monte_rattle`
- In principle, sow/reap can work with any ab initio code supported by ASE by adjusting parameters and I/O (file format, file ordering, paths, etc.)

---

## Installation

- Python: >= 3.9, < 3.14
- Optional backends (install as needed):
  - NEP: `pip install calorine`
  - DeepMD: `pip install deepmd-kit`
  - PolyMLP: `pip install pypolymlp`
  - Hiphive: `pip install hiphive`
  - TACE: `pip install tace`

Recommended (PyPI):

```bash
pip install fcs-order
```

From GitHub:

```bash
pip install git+https://github.com/gtiders/fcs-order.git
```

Notes:

- To compute 2nd-order force constants (Phonopy-compatible `FORCE_CONSTANTS_2ND`), install Phonopy: `pip install phonopy`.
- To run SCPH workflows, install Phonopy and Hiphive: `pip install phonopy hiphive`.

---

## I/O and conventions

- Structures: any ASE-readable format (POSCAR/CONTCAR, CIF, XYZ/extxyz, …). Default: `POSCAR`.
- Supercell specification:
  - mlp2/scph: 3 integers (diagonal) or 9 integers (3×3 matrix) as a positional argument list.
  - sow/reap, mlp3, mlp4: three positional integers `na nb nc`.
- Cutoff `--cutoff/-c`:
  - Negative values: by nearest-neighbor shells.
  - Positive values: by real-space distance in nm.

---

## Quick start (typical workflow)

1) Generate displaced and undisplaced structures with sow:

```bash
# Example for 3rd order (order=3), 2×2×2 supercell; negative cutoff means NN shells; VASP output
fcs-order sow 2 2 2 -c -6 -r 3 -p POSCAR -f poscar -o disps
# Output: disps/3RD.SPOSCAR and disps/3RD.POSCAR.0001 ...
```

2) Compute forces with your chosen backend and produce ASE-readable force files (vasprun.xml/OUTCAR, extxyz, …). Keep the file order identical to sow’s displaced structure order.

3) Reconstruct 3rd/4th-order IFCs with reap:

```bash
# 3rd order (order=3). The file sequence MUST match the displaced structure sequence
fcs-order reap 2 2 2 -c -6 -r 3 -p POSCAR disps/vasprun_*.xml
# Output: FORCE_CONSTANTS_3RD
```

4) For 2nd-order IFCs, use mlp2 to write Phonopy-compatible `FORCE_CONSTANTS_2ND`.

5) For SCPH, supply initial 2nd-order IFCs (optional) or have the backend evaluate within iterations, then run `scph <backend>`.

---

## sow: generate displaced structures

Command:

```
fcs-order sow na nb nc --cutoff <CUTOFF> [--order 3|4] [--poscar POSCAR] \
  [--out-format poscar|vasp|cif|xyz] [--out-dir DIR] [--name-template ...] [--undisplaced-name ...]
```

- Key options:
  - `-r/--order`: 3 (third order) or 4 (fourth order), default 3.
  - `-c/--cutoff`: negative=shells; positive=radius (nm).
  - `-p/--poscar`: input structure; default POSCAR.
  - `-f/--out-format`: output format; default poscar.
  - `-o/--out-dir`: output directory.
  - Templates: placeholders `{order}`, `{phase}`, `{index}`, `{index_padded}`, `{width}`, `{ext}` for `--name-template` / `--undisplaced-name`.
- VASP-style outputs:
  - Undisplaced: `3RD.SPOSCAR` or `4TH.SPOSCAR`
  - Displaced: `3RD.POSCAR.0001`, `4TH.POSCAR.0001`, …

---

## reap: reconstruct IFCs from force files

Command:

```
fcs-order reap na nb nc --cutoff <CUTOFF> [--order 3|4] [--poscar POSCAR] VASP_RUN_FILES...
```

- Force files must be ASE-readable (vasprun.xml/OUTCAR, extxyz, etc.).
- Number of files must match required runs:
  - 3rd order: `4 * N_irred`
  - 4th order: `8 * N_irred`
- Output: `FORCE_CONSTANTS_3RD` or `FORCE_CONSTANTS_4TH`

---

## mlp2: 2nd-order IFCs (Phonopy-compatible)

Command group: `fcs-order mlp2 <backend>`, where `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- Common parameters:
  - `supercell_matrix`: 3 or 9 integers (positional).
  - `--poscar`: primitive cell; default POSCAR.
  - `--outfile/-o`: output filename; default `FORCE_CONSTANTS_2ND`.

Examples:

- NEP (GPU optional):

```bash
fcs-order mlp2 nep 2 2 2 -p nep.txt -g -o FORCE_CONSTANTS_2ND
```

- DP:

```bash
fcs-order mlp2 dp 2 2 2 -p model.pb -o FORCE_CONSTANTS_2ND
```

- PolyMLP:

```bash
fcs-order mlp2 ploymp 2 2 2 -p polymlp.pot -o FORCE_CONSTANTS_2ND
```

- MTP (subcommand name is `mtp2`; requires external `mlp` executable):

```bash
fcs-order mlp2 mtp2 2 2 2 -p pot.mtp --mtp-exe mlp -o FORCE_CONSTANTS_2ND
```

- TACE (supports `--device/--dtype/--level`):

```bash
fcs-order mlp2 tace 2 2 2 -m model.ckpt -o FORCE_CONSTANTS_2ND --device cuda --dtype float64 --level 0
```

- Hiphive (write to Phonopy text):

```bash
fcs-order mlp2 hiphive 2 2 2 -p potential.fcp
```

---

## mlp3: 3rd-order IFCs

Command group: `fcs-order mlp3 <backend>`, where `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- Common parameters: `na nb nc`, `--cutoff/-c`, `--poscar` (default POSCAR), `--is-write`.

Examples:

- NEP:

```bash
fcs-order mlp3 nep 2 2 2 -c -6 -p nep.txt --is-write --is-gpu
```

- DP:

```bash
fcs-order mlp3 dp 2 2 2 -c -6 -p model.pb
```

- PolyMLP:

```bash
fcs-order mlp3 ploymp 2 2 2 -c -6 -p polymlp.pot
```

- MTP (subcommand name `mtp2`; requires `mlp`):

```bash
fcs-order mlp3 mtp2 2 2 2 -c -6 -p pot.mtp --mtp-exe mlp --is-write
```

- TACE:

```bash
fcs-order mlp3 tace 2 2 2 -c -6 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive (export directly to ShengBTE format from fcp):

```bash
fcs-order mlp3 hiphive 2 2 2 -p potential.fcp
```

Output: `FORCE_CONSTANTS_3RD`

---

## mlp4: 4th-order IFCs

Command group: `fcs-order mlp4 <backend>`, where `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- Common parameters: `na nb nc`, `--cutoff/-c`, `--poscar` (default POSCAR), `--is-write`.

Note: Despite algorithmic and implementation optimizations, 4th-order calculations may still be memory intensive; prefer high-memory environments.

Examples:

- NEP:

```bash
fcs-order mlp4 nep 2 2 2 -c -6 -p nep.txt --is-gpu
```

- DP:

```bash
fcs-order mlp4 dp 2 2 2 -c -6 -p model.pb
```

- PolyMLP:

```bash
fcs-order mlp4 ploymp 2 2 2 -c -6 -p polymlp.pot
```

- MTP (subcommand `mtp2`):

```bash
fcs-order mlp4 mtp2 2 2 2 -c -6 -p pot.mtp --mtp-exe mlp --is-write
```

- TACE (supports `--device/--dtype/--level`):

```bash
fcs-order mlp4 tace 2 2 2 -c -6 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive (use fcp and supercell to obtain force constants):

```bash
fcs-order mlp4 hiphive 2 2 2 -c -6 -p potential.fcp
```

Output: `FORCE_CONSTANTS_4TH`

---

## phonon_rattle: phonon-based thermal-displacement structures

Command:

```bash
fcs-order phonon_rattle SPOSCAR --fc2 FORCE_CONSTANTS -T 300 600 900 [options]
```

- Key arguments:
  - Positional `SPOSCAR`: supercell structure (SPOSCAR/POSCAR or any ASE-readable format).
  - `--fc2`: 2nd-order force constants file; defaults to `FORCE_CONSTANTS` (Phonopy text-compatible).
  - `-T/--T`: list of temperatures in K, e.g. `-T 300 600 900`.
  - `-n/--n-structures`: number of structures per temperature; default 10.
  - `--qm-statistics/--no-qm-statistics`: use quantum harmonic oscillator amplitudes instead of classical; default is classical.
  - `--imag-freq-factor`: factor for treating imaginary modes; defaults to `1.0`.
  - `-f/--format`: output format, one of `vasp`, `cif`, `qe`, `xyz`; default `vasp`.
  - `-p/--prefix`: optional filename prefix; if set, all files share a global running index like `prefix000.vasp`.
  - `--eps/--no-eps`: enable random volumetric strain; used together with `--min-volume` and `--max-volume`.

- File naming:
  - Without `--prefix`: `phonon_rattle_T{T}_id{i}.vasp` (or matching extension).
  - With `--prefix`: `<prefix><index>.<ext>` with zero-padded running index across all temperatures.

This command is useful for generating thermalized structures consistent with a given 2nd-order IFC model at one or more temperatures, e.g. for ML-potential training or finite-temperature sampling.

---

## monte_rattle: Monte Carlo rattle structures

Command:

```bash
fcs-order monte_rattle SPOSCAR -n 50 --rattle-std 0.05 --d-min 1.0 [options]
```

- Key arguments:
  - Positional `SPOSCAR`: supercell structure file.
  - `-n/--n-structures`: number of structures to generate; default 10.
  - `--rattle-std`: standard deviation of the per-step Gaussian displacement (Å); default 0.05.
  - `--d-min`: minimum interatomic distance (Å) that enters the MC acceptance probability.
  - `--width`: width of the error function controlling the acceptance window around `d_min`; default 0.1.
  - `--n-iter`: number of Monte Carlo cycles; larger values yield larger overall displacements; default 10.
  - `--max-disp`: hard limit on the displacement magnitude of any atom; moves exceeding this are rejected; default 2.0 Å.
  - `--seed`: random seed; default 42.
  - `-f/--format`: output format, one of `vasp`, `cif`, `qe`, `xyz`; default `vasp`.
  - `-p/--prefix`: optional filename prefix; if set, filenames are `<prefix><index>.<ext>`.
  - `--eps/--no-eps`: toggle random volumetric strain using a volume ratio drawn uniformly from `[min_volume, max_volume]`.

- Default naming: without prefix, files are named `mc_rattle_id{i}.vasp` (or matching extension).
- The MC rattle algorithm rejects moves that make atoms too close or displace them excessively, helping avoid unphysical swaps or overlaps.

---

## scph: self-consistent phonons (SCPH)

Command group: `fcs-order scph <backend>`, where `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- Common parameters:
  - `supercell_matrix`: 3 or 9 integers (positional).
  - `--poscar`: primitive cell; default POSCAR.
  - `--temperatures/-T`: e.g., "100,200,300".
  - `--cutoff/-c`: cluster-space cutoff radius.
  - `--alpha/-a`: mixing parameter; default 0.2.
  - `--n-iterations/-i`: iterations; default 100.
  - `--n-structures/-n`: structures per iteration; default 50.
  - `--fcs-2nd/-F`: initial 2nd-order IFCs (optional).
  - `--is-qm/-q`: use quantum statistics (default True).
  - `--imag-freq-factor/-I`: factor for handling imaginary frequencies (default 1.0).

Note: SCPH in this project integrates and calls capabilities from Hiphive. See the Hiphive docs: <https://hiphive.materialsmodeling.org/>

Examples:

- NEP (GPU optional):

```bash
fcs-order scph nep 2 2 2 -T 100,200,300 -c 4.5 -p nep.txt --poscar POSCAR --is-gpu
```

- DP:

```bash
fcs-order scph dp 2 2 2 -T 100,200,300 -c 4.5 -p graph.pb --poscar POSCAR
```

- TACE (supports `--device/--dtype/--level`):

```bash
fcs-order scph tace 2 2 2 -T 300 -c 4.5 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive:

```bash
fcs-order scph hiphive 2 2 2 -T 100,200,300 -c 4.5 -p model.fcp --poscar POSCAR
```

- PolyMLP (ploymp):

```bash
fcs-order scph ploymp 2 2 2 -T 100,200,300 -c 4.5 -p polymlp.pot --poscar POSCAR
```

- MTP (subcommand `mtp2`; requires `mlp`):

```bash
fcs-order scph mtp2 2 2 2 -T 100,200,300 -c 4.5 -p pot.mtp --poscar POSCAR --mtp-exe mlp
```

During the run, `scph_SPOSCAR` is written; forces/IFCs are evaluated iteratively with convergence analysis and outputs afterward.

---

## FAQ

- What does the cutoff sign mean?
  - Negative: by nearest-neighbor shells; Positive: real-space distance (nm).
- File order for reap?
  - The force files must follow the sow displaced-structure order; preserve numbering when generating outputs.
- Why is the MTP subcommand named `mtp2`?
  - In the current implementation the MTP subcommand is `mtp2` (used across mlp2/3/4 and scph groups); see examples above.
- Hiphive notes?
  - Requires a readable fcp file; supercell size should be no smaller than that used when training the fcp.

---

## License

Apache-2.0. See `LICENSE`.
