# unified-acdc-powerflow

[![tests](https://github.com/WillemLam/unified-acdc-powerflow/actions/workflows/tests.yml/badge.svg)](https://github.com/WillemLam/unified-acdc-powerflow/actions/workflows/tests.yml)

A unified AC/DC Newton–Raphson power flow for hybrid AC/DC grids and multi-terminal HVDC, running on
[pandapower](https://www.pandapower.org) networks.

The AC grids, the DC grids and the interfacing converters are solved in **one** Newton–Raphson, in rectangular coordinates, with an analytical Jacobian. The model is published in:

> W. Lambrichts and M. Paolone, "General and Unified Model of the Power Flow Problem in Multiterminal AC/DC
> Networks", *IEEE Transactions on Power Systems*, 2024. [doi:10.1109/TPWRS.2024.3378926](https://doi.org/10.1109/TPWRS.2024.3378926)

This is the maintained Python implementation. The MATLAB code of the paper is in
[General_and_Unified_Load_flow_for_ACDC](https://github.com/willemLam/General_and_Unified_Load_flow_for_ACDC).

## Features

- **Converter control modes:**
  - `PQ` (P and Q);
  - `VdcQ` (DC voltage and Q);
  - `PV` (P and AC voltage);
  - `VdcV` (DC voltage and AC voltage);
  - `GF`, grid-forming (AC voltage magnitude and angle);
  - `droop`, DC-voltage droop: P_dc = P_dc,ref − k·(V_dc − V_dc,ref);
  - `ACE`, AC emulation of a point-to-point link: P = P₀ − k·(θ_bus − θ_remote), k in MW/deg.
- **Converter losses:** P_loss = a + b·|I| + c·|I|².
- **Direct converter connection:** converters connect directly to their AC and DC bus. No internal nodes or coupling impedances are needed, and the buses may have any number of neighbours, loads or generators.
- **Station components** (transformer, phase reactor, filter, cable) are ordinary network elements.
- **Grids:** meshed multi-terminal DC grids, several DC grids, and AC islands fed by grid-forming converters.
- **AC side:** pandapower's admittance matrix, including transformer taps, phase shifters, shunts and line charging.

## Install

```bash
git clone https://github.com/WillemLam/unified-acdc-powerflow.git && cd unified-acdc-powerflow
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest          # full test suite
```

## Quick start

```python
from unified_acdc_powerflow import run_pf
from unified_acdc_powerflow.cases import load_case

net = load_case("ieee57_14_hvdc")    # validation/cases/ieee57_14_hvdc.yaml as a pandapower net
res = run_pf(net)
print(res.message, res.iterations, f"{res.solve_time_s * 1e3:.0f} ms")
res.table()              # per node: E_re, E_im, P, Q (p.u.)
res.converter_table()    # per converter: P_ac, Q_ac, P_dc, losses, |I|
```

Your own grid is any pandapower net with `bus_dc` / `line_dc` / `load_dc` and converters added with `unified_acdc_powerflow.add_converter(net, bus, bus_dc, mode, ...)`.

## Validation cases

`validation/` holds one notebook per test grid; the grids' case files are in `validation/cases/`:

| Grid | Notebook | Case file |
|---|---|---|
| IEEE 57 + IEEE 14 connected by two meshed HVDC grids, 8 converters | `ieee57_14_hvdc.ipynb` | `cases/ieee57_14_hvdc.yaml` |
| IEEE 30 with two embedded HVDC grids, 6 converters | `ieee30_hvdc.ipynb` | `cases/ieee30_hvdc.yaml` |
| PEGASE 1354 with two embedded HVDC grids, 5 converters | `pegase_hvdc.ipynb` | `cases/pegase_hvdc.yaml` |
| EPFL 26-bus hybrid AC/DC microgrid, 4 converters | `microgrid.ipynb` | `cases/microgrid.yaml` |
| The same microgrid **validated against EMTP-RV** | `microgrid_emtp.ipynb` | `cases/microgrid_emtp.yaml` |

Each notebook introduces the grid (with its topology figure from the paper), loads the case, builds Y, runs
Newton–Raphson and shows the results per node and per converter.

`microgrid_emtp.ipynb` compares the power flow with an independent reference: the EMTP-RV time-domain simulation
of the EPFL microgrid ([model](https://github.com/DESL-EPFL/Hybrid_ACDC_EMTP_simulation)), with the measured
injections as setpoints. The voltages match EMTP-RV to 3.3e-6 p.u. (AC) and 3.2e-8 p.u. (DC), within the errors reported in the paper
(Table III). A test keeps it that way.

A case file holds MATPOWER-style tables, one row per element: `bus`, `slack`, `gen`, `line`, `trafo` (ratio,
shift), `bus_dc`, `line_dc`, `source_dc`, `converter`. Values are per unit, and the format is documented at the
top of every file. To change a grid, edit its case file and re-run the notebook. Any other case file loads with
`unified_acdc_powerflow.grid_yaml.read_case(path)`.

## Comparison with pandapower

`benchmarks/pandapower_acdc/` compares this model with the AC/DC power flow built into pandapower 3.5.5, on the
same pandapower networks (the four grids of the paper):

- **Direct connection.** This model connects converters directly and reproduces an EMTP-RV simulation of the EPFL
  microgrid to 3.3e-6 p.u. pandapower needs a coupling impedance at every converter, and its result depends on the
  value chosen.
- **Converters next to generators.** This model solves the three HVDC grids of the paper in 4–7 iterations.
  pandapower does not converge on them, for any coupling impedance from 1e-4 to 1e-1 p.u., because a converter
  shares its bus with a voltage-controlling generator.
- **DC loads.** pandapower 3.5.5 multiplies DC loads by `sn_mva` (a 5 MW DC load draws 518 MW at `sn_mva` = 100).
- **Control modes and losses.** This model adds grid-forming with a set angle, DC-voltage droop, AC emulation and
  the a + b·|I| + c·|I|² loss model.

Where pandapower converges, the two agree, up to a difference proportional to pandapower's coupling impedance.
Three notebooks run in Colab: [IEEE 57 + IEEE 14 with two HVDC grids](benchmarks/pandapower_acdc/01_ieee57_14_two_hvdc_grids.ipynb),
[validation against EMTP-RV](benchmarks/pandapower_acdc/02_emtp_rv_microgrid.ipynb) and
[the full benchmark](benchmarks/pandapower_acdc/03_full_benchmark.ipynb). The numbers and a pandapower-only
reproducer are in the [benchmark README](benchmarks/pandapower_acdc/README.md).

## Tests

The test suite (`pytest`) covers:
- the analytical Jacobian vs finite differences for every converter mode;
- AC-only IEEE 14/57/118 vs pandapower (< 1e-8 p.u.);
- a meshed 3-terminal DC grid vs pandapower's AC/DC solver (< 1e-5 p.u.);
- droop limits and power balance;
- the AC-emulation law at the solution, and k = 0 against PQ mode;
- the EMTP-RV comparison (largest errors within the paper's Table III);
- direct connection vs a vanishing busbar impedance;
- robustness cases;
- round trips of the case-file format.

The implementation was also checked against the original MATLAB code of the paper
([General_and_Unified_Load_flow_for_ACDC](https://github.com/willemLam/General_and_Unified_Load_flow_for_ACDC)),
to 1e-11 p.u. or better.

## Contact and citation

Questions, comments and collaboration: **Willem Lambrichts**, willem.lambrichts@gmail.com.
If you use this code, please cite the paper above. Licensed under the BSD 3-Clause license (see `LICENSE`).

```bibtex
@article{lambrichts2024acdcpf,
  author  = {Lambrichts, Willem and Paolone, Mario},
  title   = {General and Unified Model of the Power Flow Problem in Multiterminal {AC/DC} Networks},
  journal = {IEEE Transactions on Power Systems},
  year    = {2024},
  doi     = {10.1109/TPWRS.2024.3378926}
}
```
