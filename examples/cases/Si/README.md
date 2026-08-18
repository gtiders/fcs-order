# Silicon

The silicon records are organized first by the MLFCS calculation method:

- `finite-difference/` contains the harmonic (FC2), three-phonon (FC3), and
  ShengBTE thermal-conductivity evidence chains. Each MLFCS result sits beside
  the corresponding phonopy or thirdorder reference calculation.
- `fitting/` contains strict extxyz training data and independent fitting
  workflows. The current dataset was converted from the ALAMODE Si examples,
  but ALAMODE fitted force constants are not used as reference results.

These are reproducible scientific examples, not pytest fixtures or strict
third-party numerical oracles.
