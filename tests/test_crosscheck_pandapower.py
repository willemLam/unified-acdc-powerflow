"""T4: same physical system solved by pandapower's AC/DC load flow and by the engine.

pandapower holds its setpoints at the grid side of its coupling impedances; the engine holds them at the
converter node. The engine therefore gets the converter-node quantities of pandapower's solution as
setpoints (the vsc impedances become ordinary branches), and every bus voltage must agree.
"""
import numpy as np
import pandapower as pp
import pytest

from unified_acdc_powerflow import add_converter, run_pf

R, X, RDC = 2.0, 20.0, 1.0  # ohm; pandapower 3.5.5 converges to ~0.05 p.u. garbage with 1 + j10 ohm


def base_net():
    net = pp.create_empty_network(sn_mva=100.0)
    b = [pp.create_bus(net, 220.0, name=f"ac{i}") for i in range(6)]
    pp.create_ext_grid(net, b[0], vm_pu=1.02)
    for f, t in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]:
        pp.create_line_from_parameters(net, b[f], b[t], 100.0, 0.05, 0.4, 10.0, 1.0)
    for i, (p, q) in zip([1, 2, 3, 4, 5], [(50, 10), (80, 20), (40, 5), (60, 15), (30, 5)]):
        pp.create_load(net, b[i], p_mw=p, q_mvar=q)
    pp.create_gen(net, b[3], p_mw=60, vm_pu=1.01)
    d = [pp.create_bus_dc(net, 320.0, name=f"dc{i}") for i in range(3)]
    for f, t, r in [(0, 1, 20.0), (1, 2, 15.0), (2, 0, 25.0)]:  # meshed 3-terminal DC grid
        pp.create_line_dc_from_parameters(net, d[f], d[t], 1.0, r, 2.0)
    return net


def add_island(net):
    """AC island (fed by a grid-forming converter) and an extra DC bus for that converter."""
    d3 = pp.create_bus_dc(net, 320.0, name="dc3")
    pp.create_line_dc_from_parameters(net, 0, d3, 1.0, 10.0, 2.0)
    i0 = pp.create_bus(net, 220.0, name="island0")
    i1 = pp.create_bus(net, 220.0, name="island1")
    pp.create_line_from_parameters(net, i0, i1, 50.0, 0.05, 0.4, 10.0, 1.0)
    pp.create_load(net, i1, p_mw=20.0, q_mvar=4.0)
    return i0


NATIVE = [  # (ac bus, dc bus, control_mode_ac, value_ac, control_mode_dc, value_dc, engine mode)
    (1, 0, "q_mvar", 10.0, "vm_pu", 1.01, "VdcQ"),
    (2, 1, "q_mvar", -5.0, "p_mw", 40.0, "PQ"),
    (5, 2, "vm_pu", 1.0, "p_mw", -25.0, "PV"),
]


def _internal_quantities(net, i, zb):
    """Converter-node injection and voltages of pandapower's solution for vsc i."""
    rv = net.res_vsc.loc[i]
    V_int = rv.vm_internal_pu * np.exp(1j * np.deg2rad(rv.va_internal_degree))
    V_b = rv.vm_pu * np.exp(1j * np.deg2rad(rv.va_degree))
    z = complex(R, X) / zb
    S_int = V_int * np.conj((V_int - V_b) / z) * net.sn_mva  # injection into the AC grid at the converter node
    return V_int, S_int, rv.vm_internal_dc_pu


@pytest.mark.parametrize("with_gf,vdcv", [(False, False), (True, False), (False, True)])
def test_engine_matches_pandapower_acdc(with_gf, vdcv):
    native = base_net()
    rows = list(NATIVE)
    if vdcv:  # the DC-voltage converter also holds its AC voltage (Edc-|Eac|)
        rows[0] = (1, 0, "vm_pu", 1.01, "vm_pu", 1.01, "VdcV")
    if with_gf:
        rows.append((add_island(native), 3, "slack", 1.03, "p_mw", 0.0, "GF"))
    for ac, dc, mac, vac, mdc, vdc, _ in rows:
        pp.create_vsc(native, ac, dc, R, X, RDC, control_mode_ac=mac, control_value_ac=vac,
                      control_mode_dc=mdc, control_value_dc=vdc)
    pp.runpp(native, tolerance_mva=1e-10, max_iteration=100)  # pandapower needs ~40 iterations with a slack vsc
    assert native.res_bus.vm_pu.min() > 0.9  # guard against pandapower's spurious low-voltage solutions

    ours = base_net()
    if with_gf:
        add_island(ours)
    zb = 220.0**2 / ours.sn_mva
    for i, (ac, dc, _, _, _, _, mode) in enumerate(rows):
        V_int, S_int, vdc_int = _internal_quantities(native, i, zb)
        j = add_converter(ours, ac, dc, mode, p_ac_mw=S_int.real, q_ac_mvar=S_int.imag, vm_ac_pu=abs(V_int),
                          va_degree=np.rad2deg(np.angle(V_int)), vm_dc_pu=vdc_int)
        ours.vsc.loc[j, ["r_ohm", "x_ohm", "r_dc_ohm"]] = [R, X, RDC]
    res = run_pf(ours, tol=1e-10)
    assert res.converged, res.message
    m = res.model
    for b in native.bus.index:
        E = res.E_ac[m.bus_lookup[b]]
        assert abs(E) == pytest.approx(native.res_bus.vm_pu[b], abs=1e-5)
        assert np.rad2deg(np.angle(E)) == pytest.approx(native.res_bus.va_degree[b], abs=1e-4)
    for d in native.bus_dc.index:
        assert res.E_dc[m.bus_dc_lookup[d]] == pytest.approx(native.res_bus_dc.vm_pu[d], abs=1e-5)


def test_add_converter_fills_pandapower_control_columns_in_load_convention():
    net = base_net()
    add_converter(net, 1, 0, "VdcQ", q_ac_mvar=10.0, vm_dc_pu=1.01)
    add_converter(net, 2, 1, "PQ", p_ac_mw=40.0, q_ac_mvar=-5.0)
    net.vsc[["r_ohm", "x_ohm", "r_dc_ohm"]] = [R, X, RDC]
    pp.runpp(net)
    assert net.res_vsc.q_mvar.tolist() == pytest.approx([-10.0, 5.0], abs=1e-6)  # consumed = -injected
    assert net.res_vsc.p_dc_mw[1] == pytest.approx(40.0, abs=1e-6)
