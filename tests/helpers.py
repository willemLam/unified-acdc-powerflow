import pandapower as pp

from unified_acdc_powerflow import add_converter


def two_area_net(mode="VdcQ", with_load_on_conv_bus=False, **conv_kw):
    """AC: slack bus 0 -- line -- bus 1 (converter A AC side) and a load at bus 1.
    DC: bus_dc 0 (conv A) -- line_dc -- bus_dc 1 (conv B), 100 kV, 10 ohm, plus a DC load.
    Converter B sits on AC bus 2, which is connected to bus 0 through a second line.
    """
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, 220.0, name="slack")
    b1 = pp.create_bus(net, 220.0, name="convA_ac")
    b2 = pp.create_bus(net, 220.0, name="convB_ac")
    pp.create_ext_grid(net, b0, vm_pu=1.02, va_degree=0.0)
    for f, t in [(b0, b1), (b0, b2)]:
        pp.create_line_from_parameters(net, f, t, 100.0, 0.05, 0.4, 10.0, 1.0)
    pp.create_load(net, b1, p_mw=30.0 if with_load_on_conv_bus else 0.0, q_mvar=5.0 if with_load_on_conv_bus else 0.0)
    d0 = pp.create_bus_dc(net, 100.0, name="dcA")
    d1 = pp.create_bus_dc(net, 100.0, name="dcB")
    pp.create_line_dc_from_parameters(net, d0, d1, 1.0, 10.0, 2.0)
    pp.create_load_dc(net, d1, p_dc_mw=20.0)
    add_converter(net, b1, d0, "VdcQ", q_ac_mvar=10.0, vm_dc_pu=1.01, name="A")
    add_converter(net, b2, d1, mode, **conv_kw, name="B")
    return net



def gf_island_net(**gf_kw):
    """two_area_net (both converters VdcQ at 1.01) plus an AC island fed by a grid-forming converter."""
    net = two_area_net("VdcQ", vm_dc_pu=1.01)
    b3 = pp.create_bus(net, 220.0, name="island")
    b4 = pp.create_bus(net, 220.0, name="island load")
    pp.create_line_from_parameters(net, b3, b4, 100.0, 0.05, 0.4, 10.0, 1.0)
    pp.create_load(net, b4, p_mw=10.0, q_mvar=3.0)
    d2 = pp.create_bus_dc(net, 100.0, name="dcC")
    pp.create_line_dc_from_parameters(net, 1, d2, 1.0, 10.0, 2.0)
    kw = dict(vm_ac_pu=1.03, va_degree=5.0)
    kw.update(gf_kw)
    add_converter(net, b3, d2, "GF", **kw, name="C")
    return net


def one_converter_net(loss=(0.0, 0.0, 0.0)):
    """AC slack 0 -- bus 1 (VdcQ converter, Q = 10 Mvar); DC d0 (1.01 pu) -- d1 (20 MW load)."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, 220.0)
    b1 = pp.create_bus(net, 220.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line_from_parameters(net, b0, b1, 100.0, 0.05, 0.4, 10.0, 1.0)
    d0 = pp.create_bus_dc(net, 100.0)
    d1 = pp.create_bus_dc(net, 100.0)
    pp.create_line_dc_from_parameters(net, d0, d1, 1.0, 10.0, 2.0)
    pp.create_load_dc(net, d1, p_dc_mw=20.0)
    add_converter(net, b1, d0, "VdcQ", q_ac_mvar=10.0, vm_dc_pu=1.01,
                  loss_a=loss[0], loss_b=loss[1], loss_c=loss[2])
    return net


def meshed_dc(net):
    """Add a third DC bus tied to both existing DC buses (converter DC nodes get two neighbours)."""
    d = pp.create_bus_dc(net, 100.0, name="dc mesh")
    pp.create_line_dc_from_parameters(net, 0, d, 1.0, 8.0, 2.0)
    pp.create_line_dc_from_parameters(net, 1, d, 1.0, 12.0, 2.0)
    pp.create_load_dc(net, d, p_dc_mw=-5.0)
    return net


def set_losses(net, a=0.011, b=0.0035, c=0.011):
    net.vsc["loss_a"], net.vsc["loss_b"], net.vsc["loss_c"] = a, b, c
    return net


def power_balance_residual(res):
    """Independent (test-only) power-balance form of the load flow, evaluated from the voltages only.

    AC nodes: nodal injection = other injections + converter injection (setpoint or droop law);
    DC nodes: nodal injection = DC loads + converter DC injection, with P_ac + P_dc + loss = 0.
    """
    import numpy as np

    m = res.model
    E, Edc = res.E_ac, res.E_dc
    S = E * np.conj(m.Y @ E)
    Pdc = Edc * (m.G @ Edc)
    s_conv = np.zeros(m.n_ac, complex)
    p_dc_conv = np.zeros(m.n_dc)
    out = []
    c = m.conv
    for i, mode in enumerate(c["mode"]):
        l, k = c["ac"][i], c["dc"][i]
        sc = S[l] - m.s_spec[l]
        iabs = abs(sc) / abs(E[l])
        loss = c["la"][i] + c["lb"][i] * iabs + c["lc"][i] * iabs**2
        if m.ac_pv[l]:  # PV generator on the converter bus: the converter injects its Q setpoint
            sc = sc.real + 1j * c["q"][i]
            iabs = abs(sc) / abs(E[l])
            loss = c["la"][i] + c["lb"][i] * iabs + c["lc"][i] * iabs**2
        if mode in ("PQ", "PV"):
            out.append(sc.real - c["p"][i])
        if mode == "droop":  # P_dc = P_dc,ref - k (E_k - V_ref), P_ac = -(P_dc + loss)
            out.append(sc.real + c["pdc"][i] - c["k"][i] * (Edc[k] - c["vref"][i]) + loss)
        if mode in ("PQ", "VdcQ", "droop") and not m.ac_pv[l]:
            out.append(sc.imag - c["q"][i])
        if mode in ("PV", "VdcV"):
            out.append(abs(E[l]) - c["vac"][i])
        if mode in ("VdcQ", "VdcV"):
            out.append(Edc[k] - c["vdc"][i])
        s_conv[l] = sc
        p_dc_conv[k] = -sc.real - loss
    free_ac = ~m.ac_ref
    s_conv_pq = np.where(m.ac_pv, s_conv.real, s_conv)  # on PV buses only P is specified
    out.extend(np.abs((S - m.s_spec - s_conv_pq)[free_ac & ~m.ac_pv]))
    out.extend(np.abs((S - m.s_spec - s_conv_pq).real[m.ac_pv]))
    out.extend(np.abs(np.abs(E[m.ac_pv]) - m.v_set[m.ac_pv]))
    # DC nodes held by a source_dc absorb any imbalance: no balance equation there
    source = m.dc_fixed.copy()
    source[[k for k, mode in zip(c["dc"], c["mode"]) if mode in ("VdcQ", "VdcV")]] = False
    out.extend(np.abs((Pdc - m.p_dc_spec - p_dc_conv)[~source]))
    return np.max(np.abs(out))
