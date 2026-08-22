# KCl SSCHA band comparison

This case uses the KCl pypolymlp and published phonopy force constants bundled
with the CI reference. One figure contains four curves:

- phonopy harmonic FC2 (`phonopy_fc222_JPCM2022`);
- phonopy MLPSSCHA at 600 K (final iteration);
- MLFCS Cartesian initialization;
- MLFCS canonical SSCHA iteration.

Both implementations use 100 snapshots per iteration, 50 canonical iterations,
600 K, quantum statistics, and seed 42. The MLFCS curve at the solid line is
the Cartesian initialization; the dashed MLFCS curve is its final canonical
iteration. The phonopy solid line is the published harmonic FC2 and the
phonopy dashed line is the final phonopy MLPSSCHA FC2. Since neither API
exposes a directly comparable history average here, the final iteration is
used for both dashed curves.

```bash
uv run --with pypolymlp --with phonopy --with seekpath --with matplotlib \
  python run.py
```

To redraw the figure from the stored 600 K results without rerunning either
SSCHA calculation:

```bash
uv run --with phonopy --with seekpath --with matplotlib \
  python run.py --plot-existing
```

The plot is a 2x2 line-plot layout: phonopy internal comparison, MLFCS
internal comparison, harmonic cross-comparison, and SSCHA cross-comparison.
Harmonic results are solid lines and final SSCHA results are dashed lines.
Phonopy is drawn first in the cross-comparison panels.

Outputs are written under `output/`, including `kcl_sscha_bands.png`, the
native MLFCS initial/canonical FC2 arrays, phonopy's final FC2, and run
metadata. The upstream phonopy data retain their BSD-3-Clause notice in
`data/`.
