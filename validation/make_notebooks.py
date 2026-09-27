"""Generate one notebook per case: an introduction, then load -> build Y -> solve -> results.

Run:  .venv/bin/python validation/make_notebooks.py
Then: .venv/bin/jupyter nbconvert --execute --to notebook --inplace validation/*.ipynb

The grid facts in each introduction (buses, DC grids, converters) are read from the case file.
"""
from collections import Counter
from pathlib import Path

import nbformat as nbf
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from unified_acdc_powerflow.cases import load_case

HERE = Path(__file__).resolve().parent
KERNEL = {"name": "unified-acdc-powerflow", "display_name": "Python 3.12 (unified-acdc-powerflow)", "language": "python"}

CASES = {
    "ieee57_14_hvdc": dict(
        case="ieee57_14_hvdc", title="IEEE 57 + IEEE 14 connected by two HVDC grids",
        about="Two non-synchronous AC grids, the IEEE 57-bus and the IEEE 14-bus system, exchange power through "
              "two meshed multi-terminal HVDC grids. This is the multi-terminal HVDC case of the "
              "paper (section IV-B, Fig. 6), with the complete data of both IEEE systems (transformer taps, shunts, "
              "generator dispatch).",
        figure_note="The converters in the case file are named `IC <AC bus>-<DC bus>`."),
    "ieee30_hvdc": dict(
        case="ieee30_hvdc", title="IEEE 30 with two embedded HVDC grids",
        about="The IEEE 30-bus system with two HVDC grids embedded in it: power can flow between AC buses both "
              "over the AC lines and through the DC grids, so the converter setpoints steer the AC flows.",
        figure_note=""),
    "pegase_hvdc": dict(
        case="pegase_hvdc", title="PEGASE 1354 with two embedded HVDC grids",
        about="The PEGASE 1354-bus model of part of the European transmission grid, with two HVDC grids embedded "
              "in it. A realistically sized case for convergence and run time.",
        figure_note=""),
    "microgrid": dict(
        case="microgrid", title="EPFL 26-bus hybrid AC/DC microgrid",
        about="The low-voltage hybrid AC/DC microgrid of the EPFL laboratory (paper section IV-A, Fig. 4): an AC grid "
              "(0.4 kV) and a DC grid (0.8 kV) coupled by four converters, each connected through a cable "
              "(AC side B15-B18, DC side B19-B22).",
        figure_note="Bus numbers in the case file are the B-numbers of the figure. The converters are named "
                    "`IC <DC bus>-<AC bus>`: IC 1 = `IC 22-18`, IC 2 = `IC 19-15`, IC 3 = `IC 21-17`, "
                    "IC 4 = `IC 20-16`."),
}

MODE_TEXT = {"PQ": "PQ (fixed P and Q)", "VdcQ": "VdcQ (holds its DC voltage, fixed Q)",
             "PV": "PV (fixed P, holds its AC voltage)",
             "VdcV": "VdcV (holds its DC and AC voltage)", "GF": "grid-forming (sets the AC voltage and angle)",
             "droop": "droop (DC-voltage droop, fixed Q)"}

INTRO = """# {title}

{about}

| | |
|---|---|
| AC buses | {n_ac} ({n_line} lines, {n_trafo} transformers, {n_gen} generators) |
| DC buses | {n_dc}, in {dc_grids} |
| Converters | {n_conv}: {modes} |
| Case file | `validation/cases/{case}.yaml` |

## What this notebook does

1. **Load** the case file into a pandapower net and classify the nodes. `LOSSY = True` adds the converter
   losses a + b·|I| + c·|I|² (coefficients of the paper, section IV-B).
2. **Build Y**: the AC admittance matrix (lines, transformer taps and shifts, shunts) and the DC conductance
   matrix.
3. **Solve** the AC grids, the DC grids and the converters together in one Newton–Raphson, and print the
   iterations, the largest mismatch and the run time.
4. **Results**: per node the voltage E = E_re + j·E_im (p.u.) and the injected power P, Q (p.u., positive into
   the grid); per converter its AC and DC power, losses and current.

## Try it

- Change a converter in the `converter` table of the case file, for example its `mode`, `p_mw`, `q_mvar` or
  `vdc_pu`, then *Run All*. Modes: PQ, VdcQ, PV, VdcV, GF (grid-forming) and droop; the columns each mode uses are
  listed at the top of the case file.
- Set `LOSSY = True` in the first cell to include converter losses.
{figure}
## Running this notebook

Use the kernel **Python 3.12 (unified-acdc-powerflow)** (VS Code: *Select Kernel* → *Jupyter Kernel…*). To set it
up once, in the repository folder:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m ipykernel install --user --name unified-acdc-powerflow --display-name "Python 3.12 (unified-acdc-powerflow)"
```
"""

LOAD = """import logging, warnings
logging.getLogger("pandapower").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.max_rows", 2000)
from unified_acdc_powerflow import build_model, build_ybus, solve
from unified_acdc_powerflow.cases import load_case

# 1) load the case and classify the nodes
CASE = "{case}"
LOSSY = False          # True: converter losses a + b|I| + c|I|^2 (coefficients of the paper)
net = load_case(CASE, lossy=LOSSY)   # pandapower net from validation/cases/{case}.yaml
model = build_model(net)
print(f"{{model.n_ac}} AC nodes, {{model.n_dc}} DC nodes, {{model.n_conv}} converters")
net.vsc[["name", "bus", "bus_dc", "mode", "p_ac_mw", "q_ac_mvar", "vm_dc_pu", "p_dc_ref_mw", "droop_k"]]
"""

BUILD_Y = """# 2) build Y (AC: pandapower Ybus with taps and shunts; DC: conductance matrix G)
build_ybus(model)
print("Y (AC):", model.Y.shape, "nnz", model.Y.nnz)
print("G (DC):", model.G.shape, "nnz", model.G.nnz)
"""

SOLVE = """# 3) Newton-Raphson
res = solve(model)
print(res.message, "| iterations:", res.iterations, "| max mismatch:", f"{res.max_mismatch:.2e}",
      "| time:", f"{res.solve_time_s * 1e3:.1f} ms")
"""

RESULTS = """# 4) results: per node E (real, imag) and injected P, Q in p.u.; then per converter
display(res.table())
res.converter_table()
"""


EMTP_INTRO = """# EPFL microgrid: validation against EMTP-RV

The EPFL 26-bus hybrid AC/DC microgrid (paper section IV-A) was simulated in **EMTP-RV**, a detailed
time-domain simulation that includes the converters and their control loops
([EMTP-RV model](https://github.com/DESL-EPFL/Hybrid_ACDC_EMTP_simulation)). This notebook takes the steady state
of that simulation as the reference:
- the power flow gets the injections measured in EMTP-RV as setpoints;
- its voltages must match the voltages measured in EMTP-RV.

This repeats the balanced validation of the paper (W. Lambrichts and M. Paolone, IEEE Trans. Power Systems, 2024, [doi:10.1109/TPWRS.2024.3378926](https://doi.org/10.1109/TPWRS.2024.3378926)).
Its Table III reports a largest voltage error of 7.4e-6 p.u. (AC) and 5.9e-8 p.u. (DC).

As in the paper, the errors are Δ = EMTP-RV − power flow, per bus: ΔE for the nodal voltages and ΔP (ΔQ) for the
nodal injections, where the power flow's injections are computed from its own voltages. At buses where P is a
setpoint, ΔP is zero by construction; the informative values are at the slack (B01) and at the converters that
hold the DC voltage (AC side B16, B18; DC side B20, B22).

| | |
|---|---|
| AC buses | {n_ac} ({n_line} lines) |
| DC buses | {n_dc}, in {dc_grids} |
| Converters | {n_conv}: {modes} |
| Case file | `validation/cases/microgrid_emtp.yaml` |
| EMTP-RV reference | `validation/emtp/emtp_balanced.csv` (phase a of the AC buses; the case is balanced) |

The setpoints follow the paper:
- the slack voltage at B01, and the loads at B03, B05, B11 and B13;
- the DC injections at B23–B26: two sources and two loads of 5 kW;
- two converters in PQ mode (measured P and Q, plus the small measured difference between their AC and DC power
  as a constant loss);
- two converters in VdcQ mode (measured DC voltage and Q).

`validation/emtp/make_emtp_case.py` shows how the case and the reference were made from the data of the paper.

## Grid topology

![Grid topology](figures/microgrid.png)

## Running this notebook

Use the kernel **Python 3.12 (unified-acdc-powerflow)** (VS Code: *Select Kernel* → *Jupyter Kernel…*). To set it
up once, in the repository folder:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m ipykernel install --user --name unified-acdc-powerflow --display-name "Python 3.12 (unified-acdc-powerflow)"
```
"""

EMTP_LOAD = """import logging, warnings
logging.getLogger("pandapower").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from unified_acdc_powerflow import build_model, build_ybus, solve
from unified_acdc_powerflow.cases import CASE_DIR, load_case

# 1) load the case (EMTP-RV setpoints) and the EMTP-RV reference
net = load_case("microgrid_emtp")
emtp = pd.read_csv(CASE_DIR.parent / "emtp" / "emtp_balanced.csv")
model = build_model(net)
print(f"{model.n_ac} AC nodes, {model.n_dc} DC nodes, {model.n_conv} converters")
emtp
"""

EMTP_SOLVE = """# 3) Newton-Raphson
res = solve(model, tol=1e-10)
print(res.message, "| iterations:", res.iterations, "| max mismatch:", f"{res.max_mismatch:.2e}",
      "| time:", f"{res.solve_time_s * 1e3:.1f} ms")
"""

EMTP_COMPARE = """# 4) compare with EMTP-RV as in the paper (section IV-A): delta = EMTP-RV - power flow, per bus
#    E: nodal voltage; P, Q: nodal injection computed by the power flow from its voltages (DC buses: P only)
m = res.model
ac = emtp.side == "AC"
E_pf = np.array([res.E_ac[m.bus_lookup[int(b[1:])]] if a else res.E_dc[m.bus_dc_lookup[int(b[1:])]]
                 for b, a in zip(emtp.bus, ac)])
S_pf = np.array([res.S_ac[m.bus_lookup[int(b[1:])]] if a else res.P_dc[m.bus_dc_lookup[int(b[1:])]]
                 for b, a in zip(emtp.bus, ac)])
dE = (emtp.E_re_pu + 1j * emtp.E_im_pu).values - E_pf
dS = (emtp.P_pu + 1j * emtp.Q_pu).values - S_pf
delta = pd.DataFrame({"bus": emtp.bus, "side": emtp.side, "dE_re (p.u.)": dE.real, "dE_im (p.u.)": dE.imag,
                      "dP (p.u.)": dS.real, "dQ (p.u.)": np.where(ac, dS.imag, np.nan)})

# summary as in Table III of the paper: mean and max of |dE| and |dP| per grid
paper = {"AC": (2.76e-6, 7.36e-6), "DC": (1.54e-8, 5.88e-8)}
summary = pd.DataFrame([{"grid": g,
                         "|dE| mean": np.abs(dE[emtp.side == g]).mean(), "|dE| max": np.abs(dE[emtp.side == g]).max(),
                         "paper |dE| mean": paper[g][0], "paper |dE| max": paper[g][1],
                         "|dP| mean": np.abs(dS.real[emtp.side == g]).mean(), "|dP| max": np.abs(dS.real[emtp.side == g]).max()}
                        for g in ("AC", "DC")])
display(summary)
delta
"""

EMTP_HIST = """# 5) histograms of the voltage errors, as in Fig. 5 of the paper
import matplotlib.pyplot as plt

fig, (ax_ac, ax_dc) = plt.subplots(1, 2, figsize=(10, 3.5))
ax_ac.hist([dE[ac].real * 1e6, dE[ac].imag * 1e6], bins=10, label=["real", "imaginary"])
ax_ac.set(title="AC grid", xlabel="ΔE (p.u. × 1e-6)", ylabel="number of buses")
ax_ac.legend()
ax_dc.hist(dE[~ac].real * 1e8, bins=6)
ax_dc.set(title="DC grid", xlabel="ΔE (p.u. × 1e-8)")
plt.tight_layout()
"""


def _facts(case):
    net = load_case(case)
    ids = list(net.bus_dc.index)
    pos = {b: i for i, b in enumerate(ids)}
    A = sp.coo_matrix((np.ones(len(net.line_dc)), ([pos[b] for b in net.line_dc.from_bus_dc],
                                                   [pos[b] for b in net.line_dc.to_bus_dc])), shape=(len(ids),) * 2)
    modes = Counter(net.vsc["mode"])
    return dict(n_ac=len(net.bus), n_line=len(net.line), n_trafo=len(net.trafo), n_gen=len(net.gen),
                n_dc=len(net.bus_dc), dc_grids=_plural(connected_components(A, directed=False)[0], "DC grid"),
                n_conv=len(net.vsc),
                modes=", ".join(f"{n} × {MODE_TEXT[m]}" for m, n in modes.items()))


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _figure(name, note):
    """Topology figure from the paper (validation/figures/<name>.png), if there is one."""
    if not (HERE / "figures" / f"{name}.png").exists():
        return ""
    return f"\n## Grid topology\n\n![Grid topology](figures/{name}.png)\n\n{note}\n".replace("\n\n\n", "\n\n")


def main():
    for name, c in CASES.items():
        intro = INTRO.format(title=c["title"], about=c["about"], case=c["case"],
                             figure=_figure(name, c["figure_note"]), **_facts(c["case"]))
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(intro)]
        nb.cells += [nbf.v4.new_code_cell(src) for src in (LOAD.format(case=c["case"]), BUILD_Y, SOLVE, RESULTS)]
        nb.metadata["kernelspec"] = KERNEL
        nbf.write(nb, HERE / f"{name}.ipynb")

    nb = nbf.v4.new_notebook()   # the EMTP-RV validation
    nb.cells = [nbf.v4.new_markdown_cell(EMTP_INTRO.format(**_facts("microgrid_emtp")))]
    nb.cells += [nbf.v4.new_code_cell(src) for src in (EMTP_LOAD, BUILD_Y, EMTP_SOLVE, EMTP_COMPARE, EMTP_HIST)]
    nb.metadata["kernelspec"] = KERNEL
    nbf.write(nb, HERE / "microgrid_emtp.ipynb")


if __name__ == "__main__":
    main()
