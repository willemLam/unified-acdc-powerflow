import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

from unified_acdc_powerflow import build_model, build_ybus
from unified_acdc_powerflow.jacobian import jacobian
from unified_acdc_powerflow.mismatch import evaluate, pack
from tests.helpers import gf_island_net, meshed_dc, set_losses, two_area_net

MODES = {
    "PQ": dict(p_ac_mw=-15.0, q_ac_mvar=4.0),
    "PV": dict(p_ac_mw=-15.0, vm_ac_pu=1.01),
    "droop": dict(p_dc_ref_mw=15.0, q_ac_mvar=4.0, droop_k=5.0, vdc_ref_pu=1.0),
    "VdcQ": dict(vm_dc_pu=1.0, q_ac_mvar=-3.0),
    "VdcV": dict(vm_dc_pu=1.0, vm_ac_pu=1.01),
    "ACE": dict(p_ac_mw=-15.0, q_ac_mvar=4.0, ace_bus=1, ace_k_mw_per_deg=20.0),
}


def _check(net, seed=0):
    m = build_model(net)
    build_ybus(m)
    rng = np.random.default_rng(seed)
    x = pack(m, np.ones(m.n_ac, complex), np.ones(m.n_dc)) + 0.01 * rng.standard_normal(
        len(pack(m, np.ones(m.n_ac), np.ones(m.n_dc))))
    ev = evaluate(m, x)
    assert ev.ok
    J = jacobian(m, ev).toarray()
    h = 1e-7
    Jfd = np.empty_like(J)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h
        Jfd[:, j] = (evaluate(m, x + e).F - evaluate(m, x - e).F) / (2 * h)
    assert J.shape == (len(x), len(x))
    assert np.max(np.abs(J - Jfd)) < 1e-6


def test_ieee14_ac_only():
    _check(pn.case14())


@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("mode", list(MODES))
def test_each_converter_mode(mode, lossy):
    net = two_area_net(mode, **MODES[mode])
    _check(set_losses(net) if lossy else net)


@pytest.mark.parametrize("lossy", [False, True])
def test_grid_forming(lossy):
    net = gf_island_net(va_degree=1.0)
    _check(set_losses(net) if lossy else net)


@pytest.mark.parametrize("mode", list(MODES))
def test_multi_neighbour_dc_and_load_on_converter_bus(mode):
    net = set_losses(meshed_dc(two_area_net(mode, with_load_on_conv_bus=True, **MODES[mode])))
    _check(net)


@pytest.mark.parametrize("mode", ["PQ", "droop", "VdcQ", "ACE"])
def test_converter_on_a_pv_generator_bus(mode):
    net = set_losses(meshed_dc(two_area_net(mode, **MODES[mode])))
    pp.create_gen(net, 2, p_mw=20.0, vm_pu=1.01)
    _check(net)


def test_grid_forming_meshed_lossy():
    _check(set_losses(meshed_dc(gf_island_net(va_degree=1.0))))


def test_ace_paired_with_the_slack_bus():
    _check(set_losses(two_area_net("ACE", **{**MODES["ACE"], "ace_bus": 0})))
