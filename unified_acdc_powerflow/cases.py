"""Validation cases: validation/cases/<case>.yaml, one per notebook in validation/ (same name).

    ieee57_14_hvdc   IEEE 57 + IEEE 14 connected by two meshed HVDC grids, 8 converters ([P] section IV-B)
    ieee30_hvdc      IEEE 30 with two embedded HVDC grids, 6 converters
    pegase_hvdc      PEGASE 1354 with two embedded HVDC grids, 5 converters
    microgrid        EPFL 26-bus hybrid AC/DC microgrid, 4 converters ([P] section IV-A)
    microgrid_emtp   the same microgrid with the setpoints of its EMTP-RV simulation (validation/emtp/)

Edit a case by editing its YAML file (format: grid_yaml.py); load_case() turns it into a pandapower net.
Any other YAML case file: grid_yaml.read_case(path).
"""
from pathlib import Path

from .grid_yaml import read_case

CASE_DIR = Path(__file__).resolve().parents[1] / "validation" / "cases"
CASES = ("ieee57_14_hvdc", "ieee30_hvdc", "pegase_hvdc", "microgrid", "microgrid_emtp")
# a, b, c (p.u.) of the quadratic loss model, as in the IEEE 57 + 14 HVDC case of the paper
# (W. Lambrichts and M. Paolone, IEEE TPWRS 2024, eq. (23) and section IV-B)
PAPER_LOSS = (11.033e-3, 3.464e-3, 11e-3)


def load_case(name, lossy=False):
    """Load validation/cases/<name>.yaml as a pandapower net; lossy=True sets the loss coefficients of the paper."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; available: {CASES}")
    net = read_case(CASE_DIR / f"{name}.yaml")
    if lossy:
        set_paper_losses(net)
    return net


def set_paper_losses(net):
    net.vsc["loss_a"], net.vsc["loss_b"], net.vsc["loss_c"] = PAPER_LOSS
    return net
