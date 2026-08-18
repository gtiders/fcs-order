"""Plot regenerated KCl SSCHA free-energy histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES / "free_energy_convergence.png")
    args = parser.parse_args()
    data = json.loads((RESULTS / "free_energy_convergence.json").read_text())
    colors = {"phonopy": "#5b8f8a", "MLFCS": "#c27b70"}
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for name, values in data.items():
        x = np.asarray(values["iteration"])
        y = np.asarray(values["free_energy_eV_per_atom"])
        error = np.asarray(values["error_eV_per_atom"])
        axis.plot(x, y, color=colors[name], linewidth=1.4, marker="o", markersize=3.0, label=name)
        finite = np.isfinite(error)
        if np.any(finite):
            axis.fill_between(
                x[finite],
                y[finite] - error[finite],
                y[finite] + error[finite],
                color=colors[name],
                alpha=0.14,
                linewidth=0,
            )
    axis.set_xlabel("Canonical iteration")
    axis.set_ylabel("Free energy (eV/atom)")
    axis.set_title("KCl SSCHA free-energy convergence")
    axis.grid(color="#edf1f1", linewidth=0.5)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
