"""T6: converters connect directly to buses with several neighbours; a busbar node in between is not needed.

Adding a busbar node + branch of vanishing impedance between every converter and its bus must give the same
solution as the direct connection.
"""
import numpy as np
import pandapower as pp
import pytest

from unified_acdc_powerflow import build_model, build_ybus, run_pf, solve
from unified_acdc_powerflow.mismatch import pack
from unified_acdc_powerflow.cases import load_case
from tests.helpers import power_balance_residual


def _with_busbars(net, scale):
    """Put each converter behind a new AC bus + line and a new DC bus + line_dc of impedance ~ scale."""
    zb_ac = lambda b: net.bus.vn_kv[b] ** 2 / net.sn_mva  # noqa: E731
    for i, v in net.vsc.iterrows():
        b = pp.create_bus(net, net.bus.vn_kv[v.bus])
        pp.create_line_from_parameters(net, v.bus, b, 1.0, 0.01 * scale * zb_ac(v.bus),
                                       1e-4 * scale * zb_ac(v.bus), 0.0, 99.0)
        d = pp.create_bus_dc(net, net.bus_dc.vn_kv[v.bus_dc])
        pp.create_line_dc_from_parameters(net, v.bus_dc, d, 1.0,
                                          0.01 * scale * net.bus_dc.vn_kv[v.bus_dc] ** 2 / net.sn_mva, 99.0)
        net.vsc.at[i, "bus"], net.vsc.at[i, "bus_dc"] = b, d
    return net


def _max_diff(a, b):
    ma, mb = a.model, b.model
    da = max(abs(a.E_ac[ma.bus_lookup[x]] - b.E_ac[mb.bus_lookup[x]]) for x in ma.bus_lookup)
    dd = max(abs(a.E_dc[ma.bus_dc_lookup[x]] - b.E_dc[mb.bus_dc_lookup[x]]) for x in ma.bus_dc_lookup)
    return max(da, dd)


@pytest.mark.parametrize("case", ["ieee57_14_hvdc", "ieee30_hvdc"])
def test_direct_connection_equals_vanishing_busbar(case):
    direct = run_pf(load_case(case, lossy=True), tol=1e-10)
    net = _with_busbars(load_case(case, lossy=True), 1e-4)
    # A VdcQ converter behind a ~1e-6 p.u. reactance is extremely stiff (a milli-radian angle error means
    # thousands of p.u. of converter power), so this limit case is warm-started from the direct solution.
    model = build_model(net)
    build_ybus(model)
    E_ac = np.ones(model.n_ac, complex)
    for b, row in model.bus_lookup.items():
        E_ac[row] = direct.E_ac[direct.model.bus_lookup[b]] if b in direct.model.bus_lookup else 1.0
    for i, v in net.vsc.iterrows():  # busbar node starts at its neighbour's voltage
        E_ac[model.bus_lookup[v.bus]] = direct.E_ac[direct.model.bus_lookup[load_case(case).vsc.bus[i]]]
    E_dc = np.ones(model.n_dc)
    for b, row in model.bus_dc_lookup.items():
        E_dc[row] = direct.E_dc[direct.model.bus_dc_lookup[b]] if b in direct.model.bus_dc_lookup else 1.0
    for i, v in net.vsc.iterrows():
        E_dc[model.bus_dc_lookup[v.bus_dc]] = direct.E_dc[direct.model.bus_dc_lookup[load_case(case).vsc.bus_dc[i]]]
    busbar = solve(model, x0=pack(model, E_ac, E_dc))
    assert direct.converged and busbar.converged, busbar.message
    assert power_balance_residual(direct) < 1e-9
    assert _max_diff(direct, busbar) < 1e-5


def test_a_real_busbar_impedance_changes_the_solution():
    direct = run_pf(load_case("ieee57_14_hvdc"))
    busbar = run_pf(_with_busbars(load_case("ieee57_14_hvdc"), 1.0))  # r = 0.01 p.u. as in the paper's case data
    assert 1e-4 < _max_diff(direct, busbar) < 5e-2
