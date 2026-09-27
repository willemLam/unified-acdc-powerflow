"""Inputs found by the final review: fixed DC nodes under converters, grid-forming root branch, out-of-service
elements, native pandapower vsc modes, initialisation, slack generators."""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

from unified_acdc_powerflow import add_converter, build_model, run_pf
from tests.helpers import gf_island_net, power_balance_residual, two_area_net


# ---- converter on a DC bus that is also held by a source_dc ----------------------------------------------
def test_pq_converter_on_source_dc_bus_keeps_other_equations_intact():
    net = two_area_net("PQ", p_ac_mw=-15.0, q_ac_mvar=2.0)
    d2 = pp.create_bus_dc(net, 100.0)
    pp.create_line_dc_from_parameters(net, 1, d2, 1.0, 10.0, 2.0)
    pp.create_load_dc(net, d2, p_dc_mw=5.0)
    pp.create_source_dc(net, 1, vm_pu=1.0)
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert power_balance_residual(res) < 1e-9
    assert np.max(np.abs(res.conv_p_ac + res.conv_p_dc + res.conv_loss)) < 1e-12


def test_grid_forming_converter_on_source_dc_bus():
    net = gf_island_net()
    pp.create_source_dc(net, 2, vm_pu=1.0)
    net.vsc.loc[0, "mode"] = "PQ"  # the source now holds the DC grid
    net.vsc.loc[1, "mode"] = "PQ"
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert power_balance_residual(res) < 1e-9


# ---- grid-forming closed form: root branch and G_ll = 0 --------------------------------------------------
@pytest.mark.parametrize("angle", [20.0, 45.0, 90.0, -60.0])
def test_grid_forming_island_solution_is_rotation_invariant(angle):
    ref = run_pf(gf_island_net(va_degree=0.0), tol=1e-10)
    rot = run_pf(gf_island_net(va_degree=angle), tol=1e-10)
    assert ref.converged and rot.converged, rot.message
    m = rot.model
    island = [m.bus_lookup[b] for b in (3, 4)]
    main = [m.bus_lookup[b] for b in (0, 1, 2)]
    assert np.allclose(rot.E_ac[island], ref.E_ac[island] * np.exp(1j * np.deg2rad(angle)), atol=1e-9)
    assert np.allclose(rot.E_ac[main], ref.E_ac[main], atol=1e-9)
    assert np.allclose(rot.E_dc, ref.E_dc, atol=1e-9)


def test_grid_forming_behind_a_lossless_line():
    net = gf_island_net()
    net.line.loc[2, "r_ohm_per_km"] = 0.0  # G_ll = 0 at the grid-forming bus
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert power_balance_residual(res) < 1e-9


@pytest.mark.parametrize("angle", [-5.0, 3.0, 5.0])
def test_grid_forming_converter_in_the_ext_grid_island(angle):
    net = two_area_net("PQ", p_ac_mw=-15.0)
    b3 = pp.create_bus(net, 220.0)
    pp.create_line_from_parameters(net, 2, b3, 100.0, 0.05, 0.4, 10.0, 1.0)
    pp.create_load(net, b3, p_mw=20.0)
    d2 = pp.create_bus_dc(net, 100.0)
    pp.create_line_dc_from_parameters(net, 1, d2, 1.0, 10.0, 2.0)
    add_converter(net, b3, d2, "GF", vm_ac_pu=1.0, va_degree=angle)
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert power_balance_residual(res) < 1e-9


def test_infeasible_grid_forming_setpoint_is_reported_not_raised():
    # +10 deg next to the slack needs ~2.2 p.u. through a DC path that can carry at most ~1.25 p.u.
    net = two_area_net("PQ", p_ac_mw=-15.0)
    b3 = pp.create_bus(net, 220.0)
    pp.create_line_from_parameters(net, 2, b3, 100.0, 0.05, 0.4, 10.0, 1.0)
    d2 = pp.create_bus_dc(net, 100.0)
    pp.create_line_dc_from_parameters(net, 1, d2, 1.0, 10.0, 2.0)
    add_converter(net, b3, d2, "GF", vm_ac_pu=1.0, va_degree=10.0)
    res = run_pf(net)
    assert not res.converged
    assert "discriminant" in res.message


def test_vdcq_converter_on_dc_bus_without_lines_is_rejected():
    net = two_area_net("PQ")
    d2 = pp.create_bus_dc(net, 100.0)
    b3 = pp.create_bus(net, 220.0)
    pp.create_line_from_parameters(net, 0, b3, 100.0, 0.05, 0.4, 10.0, 1.0)
    add_converter(net, b3, d2, "VdcQ")
    with pytest.raises(ValueError, match="no DC line"):
        build_model(net)


# ---- out-of-service buses -----------------------------------------------------------------------------------
def test_elements_on_out_of_service_buses_are_ignored():
    net = two_area_net("PQ", p_ac_mw=-15.0)
    d2 = pp.create_bus_dc(net, 100.0, in_service=False)
    pp.create_load_dc(net, d2, p_dc_mw=3.0)
    pp.create_source_dc(net, d2, vm_pu=1.0)
    b3 = pp.create_bus(net, 220.0, in_service=False)
    pp.create_ext_grid(net, b3)
    j = add_converter(net, b3, d2, "PQ", p_ac_mw=5.0)
    net.vsc.loc[j, "in_service"] = False
    ref = run_pf(two_area_net("PQ", p_ac_mw=-15.0), tol=1e-10)
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert np.allclose(res.E_dc, ref.E_dc, atol=1e-12)


# ---- native pandapower control modes -------------------------------------------------------------------
def test_mode_is_derived_from_pandapower_control_columns():
    net = two_area_net("PQ")
    net.vsc = net.vsc.iloc[0:0]
    for c in [c for c in net.vsc.columns if c not in ("name", "bus", "bus_dc", "r_ohm", "x_ohm", "r_dc_ohm",
                                                       "pl_dc_mw", "control_mode_ac", "control_value_ac",
                                                       "control_mode_dc", "control_value_dc", "controllable",
                                                       "in_service", "ref_bus")]:
        net.vsc.drop(columns=c, inplace=True)
    pp.create_vsc(net, 1, 0, 0.0, 0.0, 0.0, control_mode_ac="q_mvar", control_value_ac=10.0,
                  control_mode_dc="vm_pu", control_value_dc=1.01)
    pp.create_vsc(net, 2, 1, 0.0, 0.0, 0.0, control_mode_ac="vm_pu", control_value_ac=1.01,
                  control_mode_dc="p_mw", control_value_dc=40.0)
    m = build_model(net)
    assert list(m.conv["mode"]) == ["VdcQ", "PV"]
    assert m.conv["q"][0] == pytest.approx(-0.1)      # pandapower: q consumed
    assert m.conv["vdc"][0] == pytest.approx(1.01)
    assert m.conv["p"][1] == pytest.approx(0.4)       # pandapower: p drawn from DC = AC injection
    assert m.conv["vac"][1] == pytest.approx(1.01)
    assert run_pf(net).converged


# ---- initialisation and slack generators ---------------------------------------------------------------
@pytest.mark.parametrize("case", ["mv_oberrhein", "example_multivoltage"])
def test_networks_with_phase_shifting_transformers_converge(case):
    net = getattr(pn, case)()
    pp.runpp(net, tolerance_mva=1e-10)
    res = run_pf(net)
    assert res.converged, res.message
    m = res.model
    buses = net.res_bus.index[net.res_bus.vm_pu.notna()]
    E_pp = net.res_bus.vm_pu[buses].values * np.exp(1j * np.deg2rad(net.res_bus.va_degree[buses].values))
    E = np.array([res.E_ac[m.bus_lookup[b]] for b in buses])
    assert np.max(np.abs(E - E_pp)) < 1e-6


def test_slack_generator_is_a_reference():
    net = pn.case14()
    net.ext_grid.in_service = False
    net.gen.loc[0, "slack"] = True
    pp.runpp(net, tolerance_mva=1e-10, calculate_voltage_angles=True)
    res = run_pf(net)
    assert res.converged, res.message
    m = res.model
    E_pp = net.res_bus.vm_pu.values * np.exp(1j * np.deg2rad(net.res_bus.va_degree.values))
    assert np.max(np.abs(res.E_ac[[m.bus_lookup[b] for b in net.bus.index]] - E_pp)) < 1e-8
