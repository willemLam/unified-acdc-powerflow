"""T5: droop converter mode, P_dc = P_dc,ref - k (E_dc - V_dc,ref) (P_dc injected into the DC grid)."""
import numpy as np
import pytest

from unified_acdc_powerflow import run_pf
from tests.helpers import meshed_dc, power_balance_residual, set_losses, two_area_net


def test_stiff_droop_reproduces_vdcq():
    vdcq = run_pf(two_area_net("VdcQ", vm_dc_pu=1.0, q_ac_mvar=4.0), tol=1e-10)
    droop = run_pf(two_area_net("droop", p_dc_ref_mw=0.0, q_ac_mvar=4.0, droop_k=1e6, vdc_ref_pu=1.0), tol=1e-10)
    assert vdcq.converged and droop.converged
    assert np.max(np.abs(droop.E_dc - vdcq.E_dc)) < 1e-5
    assert np.max(np.abs(droop.E_ac - vdcq.E_ac)) < 1e-5


def test_zero_gain_droop_reproduces_pq():
    pq = run_pf(two_area_net("PQ", p_ac_mw=-15.0, q_ac_mvar=4.0), tol=1e-10)
    droop = run_pf(two_area_net("droop", p_dc_ref_mw=15.0, q_ac_mvar=4.0, droop_k=0.0), tol=1e-10)
    assert np.max(np.abs(droop.E_dc - pq.E_dc)) < 1e-9
    assert np.max(np.abs(droop.E_ac - pq.E_ac)) < 1e-9


@pytest.mark.parametrize("vdcq_on_a", [True, False])
def test_droop_law_and_power_balance_close_with_losses(vdcq_on_a):
    net = set_losses(meshed_dc(two_area_net("droop", p_dc_ref_mw=15.0, q_ac_mvar=4.0, droop_k=5.0,
                                            vdc_ref_pu=1.0, with_load_on_conv_bus=True)))
    if not vdcq_on_a:  # DC voltage held by the droop converter alone
        net.vsc.loc[0, "mode"] = "PQ"
        net.vsc.loc[0, "p_ac_mw"] = 10.0
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    m = res.model
    assert np.max(np.abs(res.conv_p_ac + res.conv_p_dc + res.conv_loss)) < 1e-9
    k = m.conv["dc"][1]
    assert res.conv_p_dc[1] == pytest.approx(0.15 - 5.0 * (res.E_dc[k] - 1.0), abs=1e-10)  # droop law
    assert power_balance_residual(res) < 1e-9
