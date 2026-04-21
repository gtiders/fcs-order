"""High-level HIFINIT library interface."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from mlfcs.hifinit.finite_difference import compute_fcs_from_orbits
from mlfcs.hifinit.simulation import generate_force_constants, prepare_simulation
from mlfcs.hifinit.writers import save_force_constants


class HifinitRun:
    """Single public runner for full HIFINIT workflow."""

    def __init__(
        self,
        primitive: Union[str, Atoms],
        supercell: Optional[Union[str, Atoms]],
        calculator,
        supercell_matrix: Optional[np.ndarray] = None,
        displacement: float = 0.01,
        cutoffs: Optional[List[Optional[float]]] = None,
    ):
        self._sim_result = prepare_simulation(
            primitive=primitive,
            calculator=calculator,
            supercell=supercell,
            supercell_matrix=supercell_matrix,
            displacement=displacement,
            cutoffs=cutoffs,
        )

        self.prim = self._sim_result["prim"]
        self.supercell = self._sim_result["supercell"]
        self.cs = self._sim_result["cs"]
        self.fcm = self._sim_result["fcm"]
        self.displacement = self._sim_result["displacement"]
        self.max_order = self._sim_result["max_order"]

        self.fc_dict = None
        self.fcp = None
        self.fcs = None

    def run(
        self, out_dir: str | None = None, verbose: bool = True
    ) -> tuple[object, object]:
        """Execute full calculation and optionally save results."""
        self.fc_dict = compute_fcs_from_orbits(
            atoms_with_calc=self._sim_result["atoms_with_calc"],
            pos_ideal=self._sim_result["pos_ideal"],
            fcm=self.fcm,
            displacement=self.displacement,
            max_order=self.max_order,
            verbose=verbose,
        )

        self.fcp, self.fcs = generate_force_constants(
            fc_dict=self.fc_dict,
            supercell=self.supercell,
            cs=self.cs,
            verbose=verbose,
        )
        if out_dir is not None:
            save_force_constants(
                prim=self.prim,
                fcp=self.fcp,
                fcs_final=self.fcs,
                max_order=self.max_order,
                out_dir=out_dir,
                verbose=verbose,
            )

        return self.fcp, self.fcs

    def save(self, out_dir: str = "./hifinit_results", verbose: bool = True) -> None:
        """Persist already-computed results."""
        if self.fcs is None:
            raise RuntimeError("Must call run() before save().")
        save_force_constants(
            prim=self.prim,
            fcp=self.fcp,
            fcs_final=self.fcs,
            max_order=self.max_order,
            out_dir=out_dir,
            verbose=verbose,
        )
