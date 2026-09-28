"""ACE converter mode (AC emulation): P_conv = P0 - k (θ_l - θ_r), k in MW/deg."""
import numpy as np
import pandapower as pp
import pytest

from unified_acdc_powerflow import build_model, run_pf
from tests.helpers import meshed_dc, power_balance_residual, set_losses, two_area_net

# converter B (AC bus 2) in ACE, paired with AC bus 1 (converter A's AC bus)
ACE_KW = dict(p_ac_mw=-15.0, q_ac_mvar=4.0, ace_bus=1, ace_k_mw_per_deg=20.0)


def _dtheta_deg(res, l_bus, r_bus):
    m = res.model
    l, r = m.bus_lookup[l_bus], m.bus_lookup[r_bus]
    return np.degrees(np.angle(res.E_ac[l] * np.conj(res.E_ac[r])))


def test_zero_gain_ace_reproduces_pq():
    pq = run_pf(two_area_net("PQ", p_ac_mw=-15.0, q_ac_mvar=4.0), tol=1e-10)
    ace = run_pf(two_area_net("ACE", **{**ACE_KW, "ace_k_mw_per_deg": 0.0}), tol=1e-10)
    assert pq.converged and ace.converged
    assert np.max(np.abs(ace.E_ac - pq.E_ac)) < 1e-10
    assert np.max(np.abs(ace.E_dc - pq.E_dc)) < 1e-10


@pytest.mark.parametrize("lossy", [False, True])
def test_ace_law_and_power_balance_hold_at_the_solution(lossy):
    net = meshed_dc(two_area_net("ACE", with_load_on_conv_bus=True, **ACE_KW))
    res = run_pf(set_losses(net) if lossy else net, tol=1e-10)
    assert res.converged, res.message
    p_mw = res.conv_p_ac[1] * res.model.base_mva
    assert p_mw == pytest.approx(-15.0 - 20.0 * _dtheta_deg(res, 2, 1), abs=1e-7)
    assert np.max(np.abs(res.conv_p_ac + res.conv_p_dc + res.conv_loss)) < 1e-9
    assert power_balance_residual(res) < 1e-9


def test_a_larger_gain_acts_like_a_stronger_parallel_line():
    """P moves against the angle difference and the angle difference shrinks, as with an extra AC line."""
    res = {k: run_pf(two_area_net("ACE", with_load_on_conv_bus=True, **{**ACE_KW, "ace_k_mw_per_deg": k}),
                     tol=1e-10) for k in (0.0, 10.0, 40.0)}
    assert all(r.converged for r in res.values())
    d0 = _dtheta_deg(res[0.0], 2, 1)
    assert abs(d0) > 0.1
    assert np.sign(res[40.0].conv_p_ac[1] - res[0.0].conv_p_ac[1]) == -np.sign(d0)
    assert abs(_dtheta_deg(res[40.0], 2, 1)) < abs(_dtheta_deg(res[10.0], 2, 1)) < abs(d0)


@pytest.mark.parametrize("change, match", [
    (dict(ace_bus=99), "not an in-service AC bus"),
    (dict(ace_bus=2), "own AC bus"),
    (dict(ace_k_mw_per_deg=-1.0), ">= 0"),
])
def test_bad_ace_inputs_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        build_model(two_area_net("ACE", **{**ACE_KW, **change}))


def test_ace_bus_in_another_ac_grid_is_rejected():
    net = two_area_net("ACE", **ACE_KW)
    b = pp.create_bus(net, 220.0, name="other grid")
    pp.create_ext_grid(net, b)
    net.vsc.loc[1, "ace_bus"] = b
    with pytest.raises(ValueError, match="same AC grid"):
        build_model(net)
