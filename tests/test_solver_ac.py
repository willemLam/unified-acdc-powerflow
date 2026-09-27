import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

from unified_acdc_powerflow import build_model, build_ybus, run_pf, solve


@pytest.mark.parametrize("case", ["case14", "case57", "case118"])
def test_ac_only_matches_pandapower(case):
    net = getattr(pn, case)()
    pp.runpp(net, tolerance_mva=1e-10, calculate_voltage_angles=True, init="flat")
    model = build_model(net)
    build_ybus(model)
    res = solve(model)
    assert res.converged, res.message
    rb = net.res_bus.loc[model.ac_bus_id]
    E_pp = rb.vm_pu.values * np.exp(1j * np.deg2rad(rb.va_degree.values))
    assert np.max(np.abs(res.E_ac - E_pp)) < 1e-8


def test_table_has_one_row_per_node_with_expected_columns():
    res = run_pf(pn.case14())
    t = res.table()
    assert list(t.columns) == ["side", "bus", "name", "type", "E_re", "E_im", "P_pu", "Q_pu"]
    assert len(t) == 14
    assert set(t.type) == {"slack", "PV", "PQ"}
    assert t.loc[t.type == "slack", "E_re"].iloc[0] == pytest.approx(1.06)


def test_non_convergence_is_reported_not_raised():
    net = pn.case14()
    net.load.p_mw *= 50
    res = run_pf(net, max_iter=5)
    assert not res.converged
    assert res.message


def test_solve_time_is_reported():
    res = run_pf(pn.case14())
    assert 0.0 < res.solve_time_s < 5.0
