"""Validation against the EMTP-RV simulation of the EPFL microgrid (paper section IV-A, balanced case).

The paper (Table III) reports a largest voltage error of 7.36e-6 p.u. (AC) and 5.88e-8 p.u. (DC); the thresholds
below are tighter.
"""
import numpy as np
import pandas as pd

from unified_acdc_powerflow import run_pf
from unified_acdc_powerflow.cases import CASE_DIR, load_case

EMTP = pd.read_csv(CASE_DIR.parent / "emtp" / "emtp_balanced.csv")


def test_engine_reproduces_the_emtp_simulation():
    res = run_pf(load_case("microgrid_emtp"), tol=1e-10)
    assert res.converged, res.message
    m = res.model
    err = {}
    for side in ("AC", "DC"):
        ref = EMTP[EMTP.side == side]
        E = [res.E_ac[m.bus_lookup[int(b[1:])]] if side == "AC" else res.E_dc[m.bus_dc_lookup[int(b[1:])]]
             for b in ref.bus]
        err[side] = np.max(np.abs(np.array(E) - (ref.E_re_pu + 1j * ref.E_im_pu)))
    assert err["AC"] < 5e-6
    assert err["DC"] < 1e-7


def test_dc_powers_match_the_emtp_simulation():
    res = run_pf(load_case("microgrid_emtp"), tol=1e-10)
    m = res.model
    ref = EMTP[EMTP.side == "DC"]
    P = [res.P_dc[m.bus_dc_lookup[int(b[1:])]] for b in ref.bus]
    assert np.max(np.abs(np.array(P) - ref.P_pu.values)) < 1e-5
