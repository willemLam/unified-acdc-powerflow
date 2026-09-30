# unified-acdc-powerflow vs pandapower's AC/DC power flow

This folder compares `unified-acdc-powerflow` (W. Lambrichts and M. Paolone, IEEE TPWRS 2024,
doi:10.1109/TPWRS.2024.3378926) with the AC/DC power flow built into pandapower 3.5.5. Both run on the same
pandapower networks: the four grids of the paper. Three notebooks, each runnable in Colab, hold all the numbers.

## Three notebooks

| Notebook | What it shows | |
|---|---|---|
| [`01_ieee57_14_two_hvdc_grids.ipynb`](01_ieee57_14_two_hvdc_grids.ipynb) | **IEEE 57 + IEEE 14 with two meshed HVDC grids and 8 converters** (paper section IV-B). unified-acdc-powerflow converges in 5 iterations; pandapower does not. Step by step, the two causes are isolated: pandapower multiplies DC loads by `sn_mva`, and it does not converge with a converter next to a voltage-controlling generator. With both worked around, the two agree. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc/01_ieee57_14_two_hvdc_grids.ipynb) |
| [`02_emtp_rv_microgrid.ipynb`](02_emtp_rv_microgrid.ipynb) | **Validation against EMTP-RV** on the EPFL 26-bus AC/DC microgrid (paper section IV-A). unified-acdc-powerflow, with the converters connected directly, reproduces the EMTP-RV voltages to 3.3e-6 p.u. and every DC power. pandapower needs a coupling impedance; it converges, but its DC sources and loads deliver 0.5 kW instead of 5 kW, and with that corrected its error grows with the impedance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc/02_emtp_rv_microgrid.ipynb) |
| [`03_full_benchmark.ipynb`](03_full_benchmark.ipynb) | **The full benchmark** (about 4 minutes): all four grids, pandapower's coupling impedance from 1e-4 to 1e-1 p.u., and three pandapower runs each (as in the paper, DC loads corrected, converters next to generators moved). It also writes the pandapower-only reproducer. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillemLam/unified-acdc-powerflow/blob/main/benchmarks/pandapower_acdc/03_full_benchmark.ipynb) |

## Summary

| | unified-acdc-powerflow | pandapower 3.5.5 |
|---|---|---|
| Converter connection | directly to its AC and DC bus | behind a coupling impedance; with zero it stops with a division by zero, and the result depends on the value chosen |
| Converter on a bus with a voltage-controlling generator | solved (PQ, VdcQ, droop and AC-emulation converters; the generator holds the voltage) | does not converge on the three HVDC grids of the paper, for every coupling impedance from 1e-4 to 1e-1 p.u. and every initialisation |
| DC loads (`load_dc`) | as given | multiplied by `sn_mva`: with `sn_mva` = 100 a converter feeding a 5 MW DC load draws 518 MW, while `res_load_dc` reports 5 MW |
| Converter control modes | PQ, PV, VdcQ, VdcV, grid-forming (voltage magnitude and angle), DC-voltage droop, AC emulation | AC side Q, voltage magnitude or slack (magnitude only); DC side P or DC voltage |
| Converter losses | a + b·\|I\| + c·\|I\|² | no-load (constant) and resistive (through the coupling impedance) |
| EPFL microgrid vs EMTP-RV, largest voltage error | AC 3.3e-6 p.u., DC 3.2e-8 p.u. | AC 4.0e-3 p.u., DC 1.9e-3 p.u. (DC loads scaled); with the DC loads corrected, AC 7e-6 to 5e-5 p.u., growing with the coupling impedance |
| Iterations on the four grids | 4–7 | – |

The paper and the MATLAB implementation also cover three-phase unbalanced grids, including intentional
negative-sequence injection (not yet in this Python engine).

## Where both converge, they agree

Both models describe the same physics. With the DC loads corrected and only the converters next to generators
moved one short branch away (r = 0.01 p.u.; both tools get this grid), pandapower converges on every grid for every
coupling impedance. The largest AC voltage difference to unified-acdc-powerflow is then proportional to the
coupling impedance, so it comes from the impedance, not from a disagreement between the models (notebook 3):

| Grid | z = 1e-4 p.u. | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|
| IEEE 57 + IEEE 14, two HVDC grids, 8 converters | 1.9e-6 | 1.9e-5 | 1.9e-4 | 2.0e-3 |
| IEEE 30, two HVDC grids, 6 converters | 1.2e-7 | 1.2e-6 | 1.2e-5 | 1.1e-4 |
| PEGASE 1354, two HVDC grids, 5 converters | 2.8e-10 | 2.8e-9 | 2.8e-8 | 2.8e-7 |
| EPFL 26-bus microgrid, 4 converters (DC loads corrected) | 3.5e-6 | 7.6e-6 | 5.0e-5 | 4.8e-4 |

Without that move, pandapower does not converge on any of the three HVDC grids, with or without the DC-load
correction, at any of these impedances. The DC voltages agree to 1e-12 p.u. or better on the three HVDC grids.

## pandapower-only reproducer

```bash
pip install pandapower==3.5.5
python benchmarks/pandapower_acdc/reproducer/reproduce.py
```

`reproducer/` holds plain pandapower JSON files that need nothing else (written by notebook 3):
- `minimal_dc_load_sn_mva_1.json` and `minimal_dc_load_sn_mva_100.json`: one converter feeding a 5 MW DC load. It
  draws 5.0 MW from the AC grid with `sn_mva` = 1, and 518.2 MW with `sn_mva` = 100;
- `ieee57_14_hvdc_z_1e-3.json`, `ieee30_hvdc_z_1e-3.json`, `pegase_hvdc_z_1e-3.json`: the three HVDC grids of the
  paper with a coupling impedance of 1e-3 p.u.; pandapower does not converge on them.

`make_notebooks.py` generates the three notebooks.
