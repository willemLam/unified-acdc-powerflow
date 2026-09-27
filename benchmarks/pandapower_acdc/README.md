# pandapower AC/DC power flow vs unified-acdc-powerflow

This folder compares pandapower 3.5.5's built-in AC/DC power flow with `unified-acdc-powerflow` (W. Lambrichts and M. Paolone, IEEE TPWRS 2024, doi:10.1109/TPWRS.2024.3378926) on the grids of the paper.

unified-acdc-powerflow connects every converter directly to its AC and DC bus. pandapower models every converter behind a coupling impedance, and approximates a direct connection with a small one (the same as a dummy bus with a tiny impedance). So pandapower gets a small impedance; unified-acdc-powerflow does not.

## Two step-by-step notebooks

| Notebook | What it shows | |
|---|---|---|
| [`01_ieee57_14_two_hvdc_grids.ipynb`](01_ieee57_14_two_hvdc_grids.ipynb) | IEEE 57 + IEEE 14 with two meshed HVDC grids and 8 converters (paper section IV-B): pandapower does not converge. Two causes, removed one at a time: DC loads are scaled by `sn_mva`, and a converter cannot share a bus with a voltage-controlling generator. With both worked around, pandapower agrees with unified-acdc-powerflow. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc/01_ieee57_14_two_hvdc_grids.ipynb) |
| [`02_emtp_rv_microgrid.ipynb`](02_emtp_rv_microgrid.ipynb) | Against an EMTP-RV simulation of the EPFL 26-bus AC/DC microgrid (paper section IV-A), no coupling impedance is needed. unified-acdc-powerflow, with direct connections, reproduces the voltages to 3.3e-6 p.u. and every DC power. pandapower converges, but its DC sources and loads deliver 0.5 kW instead of 5 kW. With that corrected, its error grows with the coupling impedance it needs. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc/02_emtp_rv_microgrid.ipynb) |

Problems found in pandapower 3.5.5:
1. **DC loads are multiplied by `sn_mva`.** A 5 MW `load_dc` draws 500 MW with a 100 MVA base. The results are silently wrong, or the power flow does not converge. This is a bug.
2. **A converter cannot share a bus with a voltage-controlling generator.** The power flow does not converge. This is common in practice: HVDC stations are often built next to power plants.

What the model of unified-acdc-powerflow adds:
- **No coupling impedance:** converters connect directly to their AC and DC bus. pandapower's small impedance works, but its result depends on the chosen value (notebook 2).
- **Converters on buses with a generator or other injections** (the problem of point 2 above).
- **Converter losses:** a + b·|I| + c·|I|².
- **DC-voltage droop control.**
- **Three-phase unbalance**, including intentional negative-sequence injection (in the paper and the MATLAB implementation; not yet in this Python engine).
- **Grid-forming converters with a set voltage angle** (in this Python engine).

## Full benchmark

```bash
.venv/bin/python benchmarks/pandapower_acdc/run_benchmark.py     # results.csv, results.md, reproducer/
```

pandapower gets the three initialisations `auto`, `dc` and `flat` and 100 iterations. Results with an AC or DC voltage outside 0.8–1.2 p.u. count as spurious. The column `pp_dc_load_bug_cancelled` repeats the run with the DC loads divided by `sn_mva`.

| Grid | Layout | pandapower | with the DC-load bug cancelled | Root cause |
|---|---|---|---|---|
| IEEE 57+14, two HVDC grids | converters on grid buses (paper) | not converged | not converged | DC load + converter on generator bus 202 |
| IEEE 57+14, two HVDC grids | converters behind busbar nodes | not converged | converged, same solution | DC load |
| IEEE 30, two HVDC grids | converters on grid buses (paper) | not converged | not converged | DC loads + converter on generator bus 102 |
| IEEE 30, two HVDC grids | converters behind busbar nodes | spurious (V_dc up to 1.92 p.u.) | converged, same solution | DC loads |
| PEGASE 1354, two HVDC grids | converters on grid buses (paper) | error | – | converters on generator buses 352 and 7282 |
| PEGASE 1354, two HVDC grids | converters behind busbar nodes | converged, same solution (9e-9) | – | – |
| EPFL 26-bus microgrid | EMTP-RV setpoints | converged, 4e-3 p.u. off | converged, same solution | DC loads (silently wrong) |

unified-acdc-powerflow converges in 4–7 iterations in every row. In a control run, each converter is replaced by a fixed injection taken from unified-acdc-powerflow's solution; pandapower then solves every AC grid, so the AC data are fine. All numbers are in `results.csv` / `results.md`.

The files:
- `run_benchmark.py`: the full benchmark;
- `make_notebooks.py`: generates the two notebooks;
- `reproducer/`: the grids where pandapower fails, as plain pandapower JSON, with `reproduce.py`, which needs only pandapower.
