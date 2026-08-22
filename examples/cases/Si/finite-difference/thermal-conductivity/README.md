# ShengBTE thermal conductivity

`mlfcs/` consumes the MLFCS FC2 and FC3. `phonopy-thirdorder-reference/`
consumes phonopy FC2 and thirdorder FC3. Each directory retains `CONTROL`, both
force-constant inputs, and the final `BTE.KappaTensorVsT_{CONV,RTA,sg}` tables.
Regenerable per-temperature scattering matrices and solver scratch output are
excluded.

`mlfcs-phono3py/` uses the same MLFCS result in phono3py's Python API. It keeps
`fc2.h5`, `fc3.h5`, the reference supercell, the `11 x 11 x 11` RTA output, and
the compact `kappa-rta.npz`/`.txt` summaries.

The historical direct MLFCS ShengBTE writer produced an anomalously low Si
conductivity, while converting the same phono3py HDF5 through hiphive restored
the expected result. This full case documents the physical symptom; focused
periodic-image and writer tests provide the automated regression.
