"""pandapower-only reproducer for the AC/DC power flow findings of this benchmark.

Needs nothing but pandapower (tested with 3.5.5):
    pip install pandapower==3.5.5
    python reproduce.py

Every *.json here is a plain pandapower net (buses, lines, trafos, loads, gens, ext_grid, bus_dc, line_dc,
load_dc and vsc with pandapower's own control modes). Each is run with init "auto", "dc" and "flat" and up to
100 iterations. A result with an AC or DC voltage outside 0.8-1.2 p.u. is reported as spurious.

- minimal_dc_load_sn_mva_*.json: one converter feeds a 5 MW DC load. With sn_mva = 1 it draws 5 MW from the AC
  grid; with sn_mva = 100 it draws about 518 MW, while res_load_dc still reports 5 MW.
- the other files: the grids of the paper (Lambrichts & Paolone, IEEE TPWRS 2024) with a coupling impedance of
  1e-3 p.u. at every converter; pandapower does not converge on them.
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
        if not (0.8 <= min(v) and max(v) <= 1.2):
            last = f"spurious solution, voltages {min(v):.2f}..{max(v):.2f} p.u. (init={init})"
            continue
        out = f"converged (init={init})"
        if len(net.vsc) == 1 and len(net.load_dc):
            out += (f": the converter draws {net.res_vsc.p_mw.iloc[0]:.1f} MW from the AC grid for a DC load of "
                    f"{net.load_dc.p_dc_mw.sum():.1f} MW (res_load_dc: {net.res_load_dc.p_dc_mw.sum():.1f} MW)")
        return out
    return last


if __name__ == "__main__":
    print(f"pandapower {pp.__version__}")
    for f in sorted(Path(__file__).parent.glob("*.json")):
        net = pp.from_json(str(f))
        print(f"{net.name}\n    {len(net.bus)} AC buses, {len(net.bus_dc)} DC buses, {len(net.vsc)} VSCs: {run(net)}")
