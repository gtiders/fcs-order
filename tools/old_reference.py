"""Run the installed legacy package as a black-box numerical reference."""

from __future__ import annotations

import argparse

import numpy as np
from calorine.calculators import CPUNEP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--order", type=int, choices=(3, 4), required=True)
    parser.add_argument("--cutoff", type=int, default=-5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.order == 3:
        import mlfcs.thirdorder.core as legacy

        runner_type = legacy.ThirdOrderRun
        core = legacy.thirdorder_core
    else:
        import mlfcs.fourthorder.core as legacy

        runner_type = legacy.FourthOrderRun
        core = legacy.fourthorder_core

    captured = {}

    def capture(force_constants, *_args, **_kwargs):
        captured["items"] = force_constants.items()

    legacy.write_ifcs = capture
    original_reconstruct = core.reconstruct_ifcs

    def capture_reconstruct(phipart, wedge, displacement_list, poscar, sposcar):
        captured["phipart"] = phipart.copy()
        captured["displacement_list"] = np.asarray(displacement_list, dtype=np.int32)
        return original_reconstruct(phipart, wedge, displacement_list, poscar, sposcar)

    core.reconstruct_ifcs = capture_reconstruct
    runner = runner_type(
        2,
        2,
        2,
        args.cutoff,
        structure_file=args.structure,
        interface="vasp",
        h=1e-3,  # legacy nanometre unit; equal to 0.01 angstrom
    )
    if args.plan_only:
        np.savez(
            args.output,
            representatives=runner.wedge.llist[:, : runner.wedge.nlist].T,
            dimensions=runner.wedge.nindependentbasis[: runner.wedge.nlist],
            independent_basis=runner.wedge.independentbasis[:, : runner.wedge.nlist].T,
            displacement_list=np.asarray(
                runner.list4 if args.order == 3 else runner.list6,
                dtype=np.int32,
            ),
        )
        return
    runner.run_calculator(CPUNEP(args.model))
    items = captured["items"]
    keys = np.asarray([key for key, _ in items], dtype=np.int32)
    values = np.asarray([value for _, value in items], dtype=float)
    np.savez(
        args.output,
        keys=keys,
        values=values,
        phipart=captured["phipart"],
        displacement_list=captured["displacement_list"],
    )


if __name__ == "__main__":
    main()
