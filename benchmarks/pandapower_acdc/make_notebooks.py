"""Generate the two step-by-step benchmark notebooks (pandapower vs unified-acdc-powerflow).

Run:  .venv/bin/python benchmarks/pandapower_acdc/make_notebooks.py
Then: .venv/bin/jupyter nbconvert --execute --to notebook --inplace benchmarks/pandapower_acdc/0*.ipynb

unified-acdc-powerflow always runs with the converters connected directly to their buses. pandapower gets the
coupling impedance its converter model needs.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/WillemLam/unified-acdc-powerflow"
RAW = "https://raw.githubusercontent.com/WillemLam/unified-acdc-powerflow/main"
COLAB = "https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc"
PAPER = ("W. Lambrichts and M. Paolone, \"General and Unified Model of the Power Flow Problem in Multiterminal AC/DC "
         "Networks\", IEEE Trans. Power Systems, 2024, "
         "[doi:10.1109/TPWRS.2024.3378926](https://doi.org/10.1109/TPWRS.2024.3378926)")

SETUP = r'''# Setup. In Colab: download the repository and install pandapower 3.5.5.
# Locally: use the repository this notebook is in.
import os, sys
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("unified-acdc-powerflow").exists():
        !git clone -q https://github.com/WillemLam/unified-acdc-powerflow.git
    %cd unified-acdc-powerflow
    !pip install -q "pandapower==3.5.5" pyyaml
else:
    root = Path.cwd()
    while not (root / "pyproject.toml").exists():
        root = root.parent
    os.chdir(root)
sys.path.insert(0, os.getcwd())

import logging, time, warnings
logging.getLogger("pandapower").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandapower as pp
from unified_acdc_powerflow import run_pf
from unified_acdc_powerflow.cases import load_case

def run_pandapower(net, init="auto"):
    """pandapower's own AC/DC power flow: a status line with the iterations and time reported by pandapower."""
    t0 = time.perf_counter()
    try:
        pp.runpp(net, init=init, max_iteration=100, tolerance_mva=1e-8)
    except Exception as err:
        return f"{type(err).__name__}: {str(err)[:70]} | time: {(time.perf_counter() - t0) * 1e3:.1f} ms"
    return f"converged | iterations: {net._ppc['iterations']} | time: {net._ppc['et'] * 1e3:.1f} ms"

def report(res):
    """The same status line for unified-acdc-powerflow."""
    return (f"{res.message} | iterations: {res.iterations} | max mismatch: {res.max_mismatch:.2e} | "
            f"time: {res.solve_time_s * 1e3:.1f} ms")

def with_coupling_impedance(net, z):
    """A copy of net for pandapower: every converter behind a coupling impedance of z p.u. (AC: r = x; DC: r)."""
    n = pp.from_json_string(pp.to_json(net))
    for i, v in n.vsc.iterrows():
        z_ac = z * n.bus.vn_kv[v.bus] ** 2 / n.sn_mva
        z_dc = z * n.bus_dc.vn_kv[v.bus_dc] ** 2 / n.sn_mva
        n.vsc.loc[i, ["r_ohm", "x_ohm", "r_dc_ohm"]] = [z_ac, z_ac, z_dc]
    return n

print("pandapower", pp.__version__)
'''

NB_HVDC = [
    ("md", f"""# 1. IEEE 57 + IEEE 14 with two HVDC grids: pandapower does not converge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/01_ieee57_14_two_hvdc_grids.ipynb)

Two non-synchronous AC grids, the IEEE 57-bus and the IEEE 14-bus system, exchange power through two meshed
multi-terminal HVDC grids with 8 converters. This is the multi-terminal HVDC case of {PAPER} (section IV-B, Fig. 6).

[unified-acdc-powerflow]({REPO}) solves it in a few iterations. pandapower 3.5.5's AC/DC power flow does not
converge. Step by step we show why: the case combines two situations that pandapower cannot handle.

![Grid topology]({RAW}/validation/figures/ieee57_14_hvdc.png)"""),
    ("code", SETUP),
    ("md", """## Step 1: the case

The case file `validation/cases/ieee57_14_hvdc.yaml` holds:
- all data of both IEEE systems (lines, transformers with taps, shunts, generators);
- two DC grids with one DC load;
- 8 converters, 3 of them holding a DC voltage and 5 with fixed P and Q.

The converters are connected directly to the grid buses."""),
    ("code", '''net = load_case("ieee57_14_hvdc")
print(f"{len(net.bus)} AC buses, {len(net.bus_dc)} DC buses, {len(net.vsc)} converters, {len(net.load_dc)} DC load")
print("converters on a bus with a voltage-controlling generator:", sorted(set(net.vsc.bus) & set(net.gen.bus)))
net.vsc[["name", "bus", "bus_dc", "mode", "p_ac_mw", "q_ac_mvar", "vm_dc_pu"]]'''),
    ("md", "## Step 2: unified-acdc-powerflow"),
    ("code", '''res = run_pf(net)
print("unified-acdc-powerflow:", report(res))
res.converter_table()'''),
    ("md", """## Step 3: pandapower

pandapower models every converter behind a coupling impedance. A direct connection is approximated with a small
one, which is the same as adding a dummy bus with a tiny impedance; with exactly zero it stops with a division by
zero. We use 1e-3 p.u. of the bus base."""),
    ("code", '''print("pandapower, zero coupling impedance:", run_pandapower(with_coupling_impedance(net, 0.0)))
print("pandapower, coupling impedance 1e-3 p.u.:", run_pandapower(with_coupling_impedance(net, 1e-3)))'''),
    ("md", """## Step 4: first reason, the DC load

pandapower 3.5.5 multiplies DC loads (`load_dc`) by the MVA base `sn_mva`, here 100: the 10 MW DC load counts as
1000 MW. We cancel that in pandapower's copy of the net by dividing the DC load by `sn_mva`."""),
    ("code", '''pp_net = with_coupling_impedance(net, 1e-3)
pp_net.load_dc["p_dc_mw"] = pp_net.load_dc.p_dc_mw / pp_net.sn_mva
print("pandapower, DC load corrected:", run_pandapower(pp_net))'''),
    ("md", """Still no convergence.

## Step 5: second reason, a converter next to a generator

Converter `IC 202-5` sits on bus 202, whose voltage is held by a generator. This is a common situation (HVDC
stations are often built next to power plants), but pandapower cannot solve it. We move that converter to a new
busbar next to bus 202, through a short branch (r = 0.01 p.u.), and give **both** tools this modified grid."""),
    ("code", '''net2 = load_case("ieee57_14_hvdc")
i = net2.vsc.index[net2.vsc.bus == 202][0]
busbar = pp.create_bus(net2, net2.bus.vn_kv[202], name="busbar of IC 202-5")
z_base = net2.bus.vn_kv[202] ** 2 / net2.sn_mva
pp.create_line_from_parameters(net2, 202, busbar, 1.0, 0.01 * z_base, 1e-4 * z_base, 0.0, 99.0)
net2.vsc.at[i, "bus"] = busbar

pp_net = with_coupling_impedance(net2, 1e-3)
pp_net.load_dc["p_dc_mw"] = pp_net.load_dc.p_dc_mw / pp_net.sn_mva      # DC-load correction of step 4
print("pandapower:", run_pandapower(pp_net))
res2 = run_pf(net2)
print("unified-acdc-powerflow:", report(res2))

m = res2.model
dv_ac = max(abs(pp_net.res_bus.vm_pu[b] - abs(res2.E_ac[m.bus_lookup[b]])) for b in net2.bus.index)
dv_dc = max(abs(pp_net.res_bus_dc.vm_pu[b] - res2.E_dc[m.bus_dc_lookup[b]]) for b in net2.bus_dc.index)
print(f"largest voltage difference: AC {dv_ac:.1e} p.u., DC {dv_dc:.1e} p.u.")'''),
    ("md", """Now pandapower converges, and the two tools agree. The remaining difference comes from the coupling
impedance in pandapower's model; notebook 2 shows against EMTP-RV that it is not needed.

**Takeaway.** On this multi-terminal HVDC case, pandapower 3.5.5 does not converge because:
- DC loads are scaled by `sn_mva` (a bug);
- a converter cannot share a bus with a voltage-controlling generator (a limitation of its converter model).

unified-acdc-powerflow solves the original case directly, with the converters connected to their buses."""),
]

NB_EMTP = [
    ("md", f"""# 2. Validation against EMTP-RV: no coupling impedance needed

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/02_emtp_rv_microgrid.ipynb)

pandapower's AC/DC power flow puts every converter behind a coupling impedance; a direct connection is
approximated with a small one (a dummy bus with a tiny impedance). Is an impedance needed at all to model a
converter correctly? We check against an independent reference: an
**EMTP-RV** time-domain simulation of the 26-bus hybrid AC/DC microgrid of the EPFL laboratory, including the
converters and their control loops ([EMTP-RV model](https://github.com/DESL-EPFL/Hybrid_ACDC_EMTP_simulation)).
This is the validation of {PAPER}, section IV-A.

Both power flows get the same grid and the injections measured in EMTP-RV, and their voltages are compared with
the EMTP-RV voltages.

![Grid topology]({RAW}/validation/figures/microgrid.png)"""),
    ("code", SETUP),
    ("md", """## Step 1: the EMTP-RV reference

The balanced case of the paper. For each bus, the voltage and the injected power at the end of the EMTP-RV
simulation, in p.u. (100 kVA; 400 V AC, 800 V DC). The case is balanced, so phase a represents the AC side."""),
    ("code", """emtp = pd.read_csv("validation/emtp/emtp_balanced.csv")
E_emtp = dict(zip(emtp.bus, emtp.E_re_pu + 1j * emtp.E_im_pu))
emtp"""),
    ("md", """## Step 2: the grid, with the EMTP-RV setpoints

The case file `validation/cases/microgrid_emtp.yaml` holds the line data and the EMTP-RV injections as setpoints:
- the slack voltage at B01;
- the loads at B03, B05, B11 and B13;
- the DC injections at B23–B26: two sources and two loads of 5 kW;
- two converters in PQ mode;
- two converters holding the DC voltage (VdcQ).

The converters are connected directly to their buses."""),
    ("code", """net = load_case("microgrid_emtp")
display(net.load_dc)
net.vsc[["name", "bus", "bus_dc", "mode", "p_ac_mw", "q_ac_mvar", "vm_dc_pu"]]"""),
    ("code", """def error_vs_emtp(E_ac, E_dc):
    \"\"\"Largest |E_EMTP-RV - E| per grid, from dictionaries bus number -> complex (AC) / real (DC) voltage.\"\"\"
    rows = [(b, "AC", abs(E_emtp[b] - E_ac[int(b[1:])])) for b in emtp.bus[emtp.side == "AC"]]
    rows += [(b, "DC", abs(E_emtp[b] - E_dc[int(b[1:])])) for b in emtp.bus[emtp.side == "DC"]]
    err = pd.DataFrame(rows, columns=["bus", "side", "|ΔE| (p.u.)"])
    return {side: err[err.side == side].iloc[:, 2].max() for side in ("AC", "DC")}
"""),
    ("md", """## Step 3: unified-acdc-powerflow, converters connected directly

For comparison, the paper (Table III) reports a largest voltage error of 7.4e-6 p.u. (AC) and 5.9e-8 p.u. (DC)."""),
    ("code", """res = run_pf(net, tol=1e-10)
m = res.model
ours = error_vs_emtp({b: res.E_ac[m.bus_lookup[b]] for b in net.bus.index},
                     {b: res.E_dc[m.bus_dc_lookup[b]] for b in net.bus_dc.index})
print("unified-acdc-powerflow:", report(res))
print(f"largest |ΔE| vs EMTP-RV: AC {ours['AC']:.1e} p.u., DC {ours['DC']:.1e} p.u.")"""),
    ("md", """Without any coupling impedance, unified-acdc-powerflow reproduces the EMTP-RV steady state to a few 1e-6 p.u.

## Step 4: pandapower

pandapower needs a non-zero coupling impedance: with zero it stops with a division by zero, and with a very small
one (1e-6 p.u.) it does not converge. With 1e-3 p.u. it works."""),
    ("code", """def pandapower_voltages(n):
    E_ac = {b: n.res_bus.vm_pu[b] * np.exp(1j * np.radians(n.res_bus.va_degree[b])) for b in n.bus.index}
    return E_ac, dict(n.res_bus_dc.vm_pu)

for z in (0.0, 1e-6):
    print(f"pandapower, coupling impedance {z:g} p.u.:", run_pandapower(with_coupling_impedance(net, z)))
pp_net = with_coupling_impedance(net, 1e-3)
print("pandapower, coupling impedance 1e-3 p.u.:", run_pandapower(pp_net))
theirs = error_vs_emtp(*pandapower_voltages(pp_net))
print(f"largest |ΔE| vs EMTP-RV: AC {theirs['AC']:.1e} p.u., DC {theirs['DC']:.1e} p.u.")"""),
    ("md", """With 1e-3 p.u. pandapower converges, but its voltages are about 1000 times further from EMTP-RV than those
of unified-acdc-powerflow. The DC powers show why. Per DC bus, the power injected into the DC network (converter
DC side at B19–B22, DC sources and loads at B23–B26):"""),
    ("code", """dc = emtp[emtp.side == "DC"]
p_pp = {b: 0.0 for b in pp_net.bus_dc.index}
for i, l in pp_net.line_dc.iterrows():            # from pandapower's own line results
    p_pp[l.from_bus_dc] += pp_net.res_line_dc.p_from_mw[i]
    p_pp[l.to_bus_dc] += pp_net.res_line_dc.p_to_mw[i]
pd.DataFrame({"bus": dc.bus.values,
              "EMTP-RV (kW)": dc.P_pu.values * 100.0,
              "unified-acdc-powerflow (kW)": [res.P_dc[m.bus_dc_lookup[int(b[1:])]] * 100.0 for b in dc.bus],
              "pandapower (kW)": [p_pp[int(b[1:])] * 1e3 for b in dc.bus]}).round(2)"""),
    ("md", """In pandapower's solution the DC sources and loads at B23–B26 inject ±0.5 kW instead of ±5 kW: it multiplies DC
loads by the MVA base (here 0.1 MVA).

## Step 5: with the DC loads corrected, the coupling impedance remains

We cancel the DC-load scaling and vary pandapower's coupling impedance."""),
    ("code", """rows = [("unified-acdc-powerflow (direct connection)", ours["AC"], ours["DC"])]
for z in (1e-4, 1e-3, 1e-2):
    n = with_coupling_impedance(net, z)
    n.load_dc["p_dc_mw"] = n.load_dc.p_dc_mw / n.sn_mva
    status = run_pandapower(n)
    e = error_vs_emtp(*pandapower_voltages(n)) if status.startswith("converged") else {"AC": np.nan, "DC": np.nan}
    rows.append((f"pandapower, DC loads corrected, coupling impedance {z:g} p.u.", e["AC"], e["DC"]))
pd.DataFrame(rows, columns=["power flow", "largest |ΔE| AC (p.u.)", "largest |ΔE| DC (p.u.)"])"""),
    ("md", """pandapower's AC error grows with its coupling impedance: the impedance is a modelling artefact, not part of the
physics. The closest agreement with EMTP-RV comes from connecting the converters directly. (pandapower's DC error
stays at about 4e-6 p.u. for another reason: it gets no converter losses here, while unified-acdc-powerflow uses
the small losses measured in EMTP-RV, as in the paper.)

**Takeaway.**
- A converter does not need a coupling impedance in the power flow. unified-acdc-powerflow connects converters
  directly and reproduces an EMTP-RV simulation to a few 1e-6 p.u.
- pandapower 3.5.5 approximates a direct connection with a small coupling impedance, so its result depends on
  that choice. On this grid it converges, but its DC powers are wrong by a factor of 10 because it scales DC
  loads by the MVA base."""),
]


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nbf.write(nb, HERE / name)


if __name__ == "__main__":
    write("01_ieee57_14_two_hvdc_grids.ipynb", NB_HVDC)
    write("02_emtp_rv_microgrid.ipynb", NB_EMTP)
