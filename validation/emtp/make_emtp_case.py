"""Build the EMTP-RV validation data of the EPFL 26-bus microgrid (paper section IV-A, balanced case).

Paper: W. Lambrichts and M. Paolone, IEEE Trans. Power Systems, 2024, doi:10.1109/TPWRS.2024.3378926.

The EMTP-RV model of the microgrid: https://github.com/DESL-EPFL/Hybrid_ACDC_EMTP_simulation

Source: the MATLAB repository of the paper (https://github.com/willemLam/General_and_Unified_Load_flow_for_ACDC),
folder Data/Microgrid:
- data_balanced.mat: steady state of the EMTP-RV time-domain simulation (voltages E_star, injections S_star;
  18 AC buses x 3 phases, then 8 DC buses, per unit on 100 kVA, 400 V AC and 800 V DC);
- linedata_AC.txt / linedata_DC.txt: the line data used for that validation (R, X in ohm/km, B in S/km, length).

Writes:
- validation/emtp/emtp_balanced.csv: the EMTP-RV reference per bus (phase a for AC; the case is balanced);
- validation/cases/microgrid_emtp.yaml: the grid with the EMTP-RV injections as setpoints, set up as in the paper's MATLAB code (initialize.m):
  slack B01, PQ buses B02-B14, DC buses B23-B26 with their measured injections, converters
  B17-B21 and B15-B19 in PQ mode (measured AC P, Q; the small measured difference between AC and DC power is
  a constant converter loss), B18-B22 and B16-B20 in VdcQ mode (measured DC voltage and Q).
  The DC lines have two conductors: the MATLAB code halves their admittance, i.e. the loop resistance is 2R.

Run:  python validation/emtp/make_emtp_case.py <path to Data/Microgrid>
"""
import sys
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
from scipy.io import loadmat

from unified_acdc_powerflow import add_converter
from unified_acdc_powerflow.grid_yaml import write_case

HERE = Path(__file__).resolve().parent
BASE_MVA, F_HZ = 0.1, 50.0


def main(src):
    src = Path(src)
    d = loadmat(src / "data_balanced.mat", squeeze_me=True, struct_as_record=False)["data"]
    E = np.r_[d.E_star[:54:3], d.E_star[54:]]          # phase a of the 18 AC buses, then the 8 DC buses
    S = np.r_[d.S_star[:54:3], d.S_star[54:]]
    buses = [f"B{i:02d}" for i in range(1, 27)]
    pd.DataFrame({"bus": buses, "side": ["AC"] * 18 + ["DC"] * 8, "E_re_pu": E.real, "E_im_pu": E.imag,
                  "P_pu": S.real, "Q_pu": S.imag}).to_csv(HERE / "emtp_balanced.csv", index=False)

    def lines(name):
        return [[float(x) for x in row.split()[:6]] for row in open(src / name) if row.strip()]

    net = pp.create_empty_network(name="microgrid_emtp", sn_mva=BASE_MVA, f_hz=F_HZ)
    for i in range(1, 19):
        pp.create_bus(net, 0.4, index=i, name=buses[i - 1])
    for i in range(19, 27):
        pp.create_bus_dc(net, 0.8, index=i, name=buses[i - 1])
    for f, t, r, x, b, length in lines("linedata_AC.txt"):
        pp.create_line_from_parameters(net, int(f), int(t), length, r, x, b / (2 * np.pi * F_HZ) * 1e9, 1.0)
    for f, t, r, _, _, length in lines("linedata_DC.txt"):
        pp.create_line_dc_from_parameters(net, int(f), int(t), length, 2 * r, 1.0)
    pp.create_ext_grid(net, 1, vm_pu=abs(E[0]), va_degree=float(np.degrees(np.angle(E[0]))))
    for i in range(2, 15):
        if abs(S[i - 1]) > 1e-12:
            pp.create_load(net, i, p_mw=-S[i - 1].real * BASE_MVA, q_mvar=-S[i - 1].imag * BASE_MVA)
    for i in range(23, 27):
        pp.create_load_dc(net, i, p_dc_mw=-S[i - 1].real * BASE_MVA)
    for ac, dc in ((17, 21), (15, 19)):
        add_converter(net, ac, dc, "PQ", p_ac_mw=S[ac - 1].real * BASE_MVA, q_ac_mvar=S[ac - 1].imag * BASE_MVA,
                      loss_a=float(-(S[ac - 1].real + S[dc - 1].real)), name=f"IC {dc}-{ac}")
    for ac, dc in ((18, 22), (16, 20)):
        add_converter(net, ac, dc, "VdcQ", vm_dc_pu=float(E[dc - 1].real), q_ac_mvar=S[ac - 1].imag * BASE_MVA,
                      name=f"IC {dc}-{ac}")
    write_case(net, HERE.parent / "cases" / "microgrid_emtp.yaml",
               title="microgrid_emtp: EPFL 26-bus microgrid with the setpoints of the EMTP-RV validation (paper section IV-A)")


if __name__ == "__main__":
    main(sys.argv[1])
