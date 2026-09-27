"""pandapower-only reproducer: AC/DC grids on which pandapower's power flow fails.

Needs nothing but pandapower (tested with 3.5.5):
    pip install pandapower==3.5.5
    python reproduce.py

Every *.json here is a plain pandapower net (buses, lines, trafos, loads, gens, ext_grid, bus_dc, line_dc,
load_dc and vsc with pandapower's own control modes). Each is run with init "auto", "dc" and "flat" and up to
100 iterations. A result with an AC or DC voltage outside 0.8-1.2 p.u. is reported as spurious.
"""
import logging
import warnings
from pathlib import Path

import pandapower as pp

logging.getLogger("pandapower").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


def run(net):
    last = "not converged"
    for init in ("auto", "dc", "flat"):
        try:
            pp.runpp(net, init=init, max_iteration=100)
        except Exception as err:
            if not str(err).startswith("Power Flow"):
                last = f"error ({type(err).__name__}: {str(err)[:60]})"
            continue
        v = list(net.res_bus.vm_pu) + list(net.res_bus_dc.vm_pu)
        if 0.8 <= min(v) and max(v) <= 1.2:
            return f"converged (init={init})"
        last = f"spurious solution, voltages {min(v):.2f}..{max(v):.2f} p.u. (init={init})"
    return last


if __name__ == "__main__":
    print(f"pandapower {pp.__version__}")
    for f in sorted(Path(__file__).parent.glob("*.json")):
        net = pp.from_json(str(f))
        print(f"{net.name}\n    {len(net.bus)} AC buses, {len(net.bus_dc)} DC buses, {len(net.vsc)} VSCs: {run(net)}")
