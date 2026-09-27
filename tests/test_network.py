import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

from unified_acdc_powerflow import add_converter, build_model, build_ybus
from tests.helpers import two_area_net


def test_ieee14_ybus_and_injections_match_pandapower():
    net = pn.case14()
    pp.runpp(net, tolerance_mva=1e-10)
    model = build_model(net)
    build_ybus(model)
    Y_pp = net._ppc["internal"]["Ybus"].toarray()
    assert np.allclose(model.Y.toarray(), Y_pp, atol=1e-12)
    assert model.n_dc == 0 and model.G.shape == (0, 0)
    assert np.flatnonzero(model.ac_ref).tolist() == [0]
    assert sorted(np.flatnonzero(model.ac_pv).tolist()) == sorted(net.gen.bus.tolist())
    # at a converged pandapower solution the PQ-bus injections are the load flow injections
    V = net._ppc["internal"]["V"]
    S = V * np.conj(Y_pp @ V)
    pq = ~(model.ac_ref | model.ac_pv)
    assert np.allclose(model.s_spec[pq], S[pq], atol=1e-8)
    assert np.allclose(model.s_spec.real[model.ac_pv], S.real[model.ac_pv], atol=1e-8)
    assert np.allclose(model.v_set[model.ac_pv], np.abs(V[model.ac_pv]), atol=1e-10)


def test_dc_conductance_load_and_vdc_fixed():
    net = two_area_net("PQ", p_ac_mw=-15.0, q_ac_mvar=2.0)
    model = build_model(net)
    build_ybus(model)
    g = 1.0 / (10.0 * 100.0 / 100.0**2)
    assert np.allclose(model.G.toarray(), [[g, -g], [-g, g]])
    assert np.allclose(model.p_dc_spec, [0.0, -0.2])
    assert model.dc_fixed.tolist() == [True, False]
    assert model.vdc_set[0] == pytest.approx(1.01)
    assert list(model.conv["mode"]) == ["VdcQ", "PQ"]
    assert model.conv["p"][1] == pytest.approx(-0.15)
    assert model.conv["q"][1] == pytest.approx(0.02)


def test_grid_forming_bus_becomes_reference_with_angle():
    net = two_area_net("VdcQ", vm_dc_pu=1.0)
    # separate AC island fed only by a grid-forming converter on the DC grid
    b3 = pp.create_bus(net, 220.0, name="island")
    b4 = pp.create_bus(net, 220.0)
    pp.create_line_from_parameters(net, b3, b4, 100.0, 0.05, 0.4, 10.0, 1.0)
    pp.create_load(net, b4, p_mw=10.0)
    d2 = pp.create_bus_dc(net, 100.0)
    pp.create_line_dc_from_parameters(net, 1, d2, 1.0, 10.0, 2.0)
    net.vsc.loc[1, "vm_dc_pu"] = 1.01  # both VdcQ at the same voltage
    add_converter(net, b3, d2, "GF", vm_ac_pu=1.03, va_degree=5.0)
    model = build_model(net)
    row = model.bus_lookup[b3]
    assert model.ac_ref[row]
    assert model.v_set[row] == pytest.approx(1.03)
    assert model.va_set[row] == pytest.approx(np.deg2rad(5.0))
    assert not model.ac_ref[model.bus_lookup[b4]]


@pytest.mark.parametrize("mode,kw", [("PV", dict(p_ac_mw=-15.0)), ("VdcV", dict(vm_dc_pu=1.0, vm_ac_pu=1.01))])
def test_gen_on_ac_voltage_controlling_converter_bus_is_rejected(mode, kw):
    net = two_area_net(mode, **kw)
    pp.create_gen(net, 2, p_mw=10.0, vm_pu=1.0)
    with pytest.raises(ValueError, match="gen"):
        build_model(net)


def test_ext_grid_on_converter_bus_is_rejected():
    net = two_area_net("PQ")
    pp.create_ext_grid(net, 2)
    with pytest.raises(ValueError, match="ext_grid"):
        build_model(net)


@pytest.mark.parametrize("mode,kw", [("PQ", dict(p_ac_mw=-15.0, q_ac_mvar=4.0)), ("VdcQ", dict(vm_dc_pu=1.0)),
                                     ("droop", dict(p_dc_ref_mw=15.0, droop_k=5.0, q_ac_mvar=4.0))])
def test_converter_next_to_a_pv_generator(mode, kw):
    from unified_acdc_powerflow import run_pf
    from tests.helpers import power_balance_residual, set_losses

    net = set_losses(two_area_net(mode, **kw))
    pp.create_gen(net, 2, p_mw=20.0, vm_pu=1.01)
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    assert abs(res.E_ac[res.model.bus_lookup[2]]) == pytest.approx(1.01, abs=1e-10)  # the gen holds |E|
    assert res.conv_q_ac[1] == pytest.approx(kw.get("q_ac_mvar", 0.0) / 100)       # converter injects Q_set
    assert power_balance_residual(res) < 1e-9


def test_dc_island_without_voltage_control_is_rejected():
    net = two_area_net("PQ")
    net.vsc.loc[0, "mode"] = "PQ"
    with pytest.raises(ValueError, match="DC island"):
        build_model(net)


def test_droop_converter_alone_can_hold_a_dc_island():
    net = two_area_net("droop", droop_k=5.0)
    net.vsc.loc[0, "mode"] = "PQ"
    build_model(net)  # no error


def test_two_converters_on_one_dc_bus_are_rejected():
    net = two_area_net("PQ")
    b3 = pp.create_bus(net, 220.0)
    pp.create_line_from_parameters(net, 0, b3, 100.0, 0.05, 0.4, 10.0, 1.0)
    add_converter(net, b3, 0, "PQ")
    with pytest.raises(ValueError, match="DC bus"):
        build_model(net)


def test_vsc_impedances_become_ordinary_branches_and_buses():
    net = two_area_net("PQ")
    net.vsc.loc[1, ["r_ohm", "x_ohm", "r_dc_ohm"]] = [1.0, 5.0, 0.5]
    model = build_model(net)
    assert model.n_ac == 4 and model.n_dc == 3
    build_ybus(model)
    assert model.G.shape == (3, 3)


def test_vdcv_converter_holds_its_dc_and_ac_voltage():
    from unified_acdc_powerflow import run_pf
    from tests.helpers import power_balance_residual, set_losses

    net = set_losses(two_area_net("VdcV", vm_dc_pu=0.995, vm_ac_pu=1.012))
    res = run_pf(net, tol=1e-10)
    assert res.converged, res.message
    m = res.model
    assert abs(res.E_ac[m.bus_lookup[2]]) == pytest.approx(1.012, abs=1e-10)
    assert res.E_dc[m.bus_dc_lookup[1]] == pytest.approx(0.995, abs=1e-12)
    assert power_balance_residual(res) < 1e-9


def test_native_pandapower_vdc_and_vm_control_becomes_vdcv():
    net = two_area_net("PQ")
    net.vsc = net.vsc.iloc[0:0]
    net.vsc = net.vsc.drop(columns=[c for c in ("mode",) if c in net.vsc])
    pp.create_vsc(net, 1, 0, 0.0, 0.0, 0.0, control_mode_ac="q_mvar", control_value_ac=0.0,
                  control_mode_dc="vm_pu", control_value_dc=1.01)
    pp.create_vsc(net, 2, 1, 0.0, 0.0, 0.0, control_mode_ac="vm_pu", control_value_ac=1.01,
                  control_mode_dc="vm_pu", control_value_dc=1.0)
    m = build_model(net)
    assert list(m.conv["mode"]) == ["VdcQ", "VdcV"]
    assert m.conv["vac"][1] == pytest.approx(1.01) and m.conv["vdc"][1] == pytest.approx(1.0)
