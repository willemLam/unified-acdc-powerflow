import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest
from scipy.optimize import fsolve

from unified_acdc_powerflow import build_model, build_ybus
from unified_acdc_powerflow.mismatch import evaluate, pack, unpack
from tests.helpers import gf_island_net, one_converter_net, two_area_net


def _model(net, **kw):
    m = build_model(net, **kw)
    build_ybus(m)
    return m


def test_residual_vanishes_at_pandapower_solution_ieee14():
    net = pn.case14()
    pp.runpp(net, tolerance_mva=1e-11)
    m = _model(net)
    V = net._ppc["internal"]["V"]
    ev = evaluate(m, pack(m, V, np.zeros(0)))
    assert np.max(np.abs(ev.F)) < 1e-8


@pytest.mark.parametrize("loss", [(0.0, 0.0, 0.0), (0.011, 0.0035, 0.011)])
def test_vdcq_root_equals_setpoint_at_exact_solution(loss):
    m = _model(one_converter_net(loss))
    Y, G = m.Y.toarray(), m.G.toarray()
    E0, vdc = m.v_set[0], 1.01

    def balance(z):  # independent power-balance form
        E = np.array([E0, z[0] + 1j * z[1]])
        Edc = np.array([vdc, z[2]])
        S = E * np.conj(Y @ E)
        Pdc = Edc * (G @ Edc)
        Iabs = abs(S[1]) / abs(E[1])
        loss_ = loss[0] + loss[1] * Iabs + loss[2] * Iabs**2
        return [S[1].real + Pdc[0] + loss_, S[1].imag - 0.1, Pdc[1] + 0.2]

    z = fsolve(balance, [1.0, 0.0, 1.0], xtol=1e-13)
    assert np.max(np.abs(balance(z))) < 1e-12
    E_ac = np.array([E0, z[0] + 1j * z[1]])
    E_dc = np.array([vdc, z[2]])
    ev = evaluate(m, pack(m, E_ac, E_dc))
    assert ev.ok
    assert np.max(np.abs(ev.F)) < 1e-10


@pytest.mark.parametrize("mode,kw", [("PQ", dict(p_ac_mw=-15.0)), ("PV", dict(p_ac_mw=-15.0, vm_ac_pu=1.0)),
                                     ("droop", dict(p_dc_ref_mw=15.0, droop_k=5.0)), ("VdcQ", dict(vm_dc_pu=1.0))])
def test_rows_equal_unknowns_for_every_mode(mode, kw):
    m = _model(two_area_net(mode, **kw))
    x = pack(m, np.ones(m.n_ac, complex), np.ones(m.n_dc))
    assert evaluate(m, x).F.shape == x.shape


def test_rows_equal_unknowns_grid_forming_and_unpack_restores_fixed_values():
    m = _model(gf_island_net())
    x = pack(m, np.ones(m.n_ac, complex), np.ones(m.n_dc))
    assert evaluate(m, x).F.shape == x.shape
    E_ac, E_dc = unpack(m, x)
    l = m.conv["ac"][2]
    assert E_ac[l] == pytest.approx(1.03 * np.exp(1j * np.deg2rad(5.0)))
    assert E_dc[m.conv["dc"][0]] == pytest.approx(1.01)
