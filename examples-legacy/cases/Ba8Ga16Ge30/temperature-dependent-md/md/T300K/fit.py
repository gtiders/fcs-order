"""Fit 300 K effective FC2+FC3 IFCs from this directory's NVE trajectory."""

import runpy
import sys
from pathlib import Path

script = Path(__file__).parents[2] / "fit_effective_ifcs.py"
sys.argv = [str(script), str(Path(__file__).parent)]
runpy.run_path(str(script), run_name="__main__")
