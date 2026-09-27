"""pandapower net -> numeric model: per-unit injections, node classes, converters, Ybus and DC conductance.

Node classes ([P] Table I / [T] Table 3.1):
    AC slack (ext_grid)             E fixed
    AC PQ                           P, Q given ([P] (1)-(2))
    AC PV (gen)                     P, |E| given ([P] (3)-(4))
    DC P                            P given ([P] (5))
    DC V (source_dc)                E fixed ([P] (6))
    interfacing converter (vsc)     modes PQ ([P] (17)-(19)), VdcQ ([P] (7), (12)-(15)), PV ([P] (8), (4)),
                                    VdcV (the Edc-|Eac| pair of [P] section I: (14)-(15) with (4)),
                                    grid-forming GF ([T] (3.37)-(3.44)), and droop (standard DC-voltage droop)

A converter connects directly to its AC bus `vsc.bus` and DC bus `vsc.bus_dc`; both may have any number of
neighbours and other elements (the closed forms use the sums over all neighbours). Station components
(transformer, phase reactor, filter) are ordinary pandapower elements. Non-zero vsc r/x/r_dc are turned into an
ordinary line / line_dc plus a bus (expand_converter_impedances).
"""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandapower as pp
import scipy.sparse as sp
from pandapower.auxiliary import _add_ppc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import BUS_TYPE, NONE, PD, QD, REF, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG
from pandapower.pypower.makeYbus import makeYbus
from scipy.sparse.csgraph import connected_components

MODES = ("PQ", "VdcQ", "PV", "VdcV", "GF", "droop")
# engine setpoint columns on net.vsc (all injections into the grid, generator convention)
CONV_COLUMNS = {
    "mode": "PQ",
    "p_ac_mw": 0.0, "q_ac_mvar": 0.0,      # PQ / PV: injection into the AC grid; VdcQ / droop: q_ac_mvar only
    "vm_ac_pu": 1.0, "va_degree": 0.0,     # PV / VdcV: |E_l|; GF: |E_l| and angle
    "vm_dc_pu": 1.0,                       # VdcQ / VdcV: E_k
    "p_dc_ref_mw": 0.0, "droop_k": 0.0, "vdc_ref_pu": 1.0,  # droop: P_dc = p_dc_ref - k (E_k - vdc_ref) sn_mva
    "loss_a": 0.0, "loss_b": 0.0, "loss_c": 0.0,           # P_loss = a + b|I| + c|I|^2 (p.u., 3.62)
}
# pandapower's own control modes, used only when pandapower itself solves the net. pandapower uses load
# convention: q_mvar is consumed from the AC bus, p_mw is drawn from the DC bus (= AC injection if lossless).
_PP_CONTROL = {
    "PQ": ("q_mvar", "q_ac_mvar", "p_mw", "p_ac_mw"),
    "VdcQ": ("q_mvar", "q_ac_mvar", "vm_pu", "vm_dc_pu"),
    "PV": ("vm_pu", "vm_ac_pu", "p_mw", "p_ac_mw"),
    "VdcV": ("vm_pu", "vm_ac_pu", "vm_pu", "vm_dc_pu"),
    "GF": ("slack", "vm_ac_pu", "p_mw", None),
    "droop": ("q_mvar", "q_ac_mvar", "p_mw", None),  # value: -p_dc_ref_mw (pandapower has no droop)
}


@dataclass
class Model:
    base_mva: float
    n_ac: int
    n_dc: int
    ac_bus_id: np.ndarray        # pandapower bus index per AC node (ppc row)
    dc_bus_id: np.ndarray        # pandapower bus_dc index per DC node
    bus_lookup: dict             # pandapower bus index -> AC node
    bus_dc_lookup: dict          # pandapower bus_dc index -> DC node
    ac_ref: np.ndarray           # AC node voltage fixed (ext_grid, grid-forming converter, isolated)
    ac_pv: np.ndarray            # AC node with a voltage-controlling generator
    v_set: np.ndarray
    va_set: np.ndarray           # rad
    s_spec: np.ndarray           # complex injection of all non-converter elements (p.u.)
    dc_fixed: np.ndarray
    vdc_set: np.ndarray
    p_dc_spec: np.ndarray        # DC injection of load_dc (p.u., loads negative)
    conv: dict
    dc_source: np.ndarray = None  # DC node held by a source_dc (absorbs any imbalance)
    Y: sp.csr_matrix = None
    G: sp.csr_matrix = None
    ac_names: list = field(default_factory=list)
    dc_names: list = field(default_factory=list)
    _ppc: dict = None
    _dc_lines: tuple = None      # (from, to, g, g_shunt) in DC node indices, p.u.

    @property
    def n_conv(self):
        return len(self.conv["mode"])


def add_converter(net, bus, bus_dc, mode, p_ac_mw=0.0, q_ac_mvar=0.0, vm_ac_pu=1.0, va_degree=0.0,
                  vm_dc_pu=1.0, p_dc_ref_mw=0.0, droop_k=0.0, vdc_ref_pu=1.0, loss_a=0.0, loss_b=0.0,
                  loss_c=0.0, name=None):
    """Add an interfacing converter (zero impedance) as a `vsc` row with the engine's setpoint columns.

    Modes and their setpoints:
        PQ     p_ac_mw, q_ac_mvar         (injection into the AC grid)
        PV     p_ac_mw, vm_ac_pu
        VdcQ   vm_dc_pu, q_ac_mvar
        VdcV   vm_dc_pu, vm_ac_pu         (holds the DC voltage and the AC voltage magnitude)
        GF     vm_ac_pu, va_degree        (grid-forming: the converter fixes the AC voltage phasor)
        droop  p_dc_ref_mw, droop_k, vdc_ref_pu, q_ac_mvar:
               P_dc = p_dc_ref_mw - droop_k * (E_dc - vdc_ref_pu) * sn_mva, P_dc injected into the DC grid
               (rectifier positive), droop_k >= 0 in p.u. power per p.u. voltage (droop_k = 1 / droop).
    Losses: P_loss = loss_a + loss_b |I| + loss_c |I|^2, all in p.u. on sn_mva.
    """
    if mode not in MODES:
        raise ValueError(f"unknown converter mode {mode!r}; use one of {MODES}")
    vals = dict(mode=mode, p_ac_mw=p_ac_mw, q_ac_mvar=q_ac_mvar, vm_ac_pu=vm_ac_pu, va_degree=va_degree,
                vm_dc_pu=vm_dc_pu, p_dc_ref_mw=p_dc_ref_mw, droop_k=droop_k, vdc_ref_pu=vdc_ref_pu,
                loss_a=loss_a, loss_b=loss_b, loss_c=loss_c)
    mac, vac, mdc, vdc = _PP_CONTROL[mode]
    value_dc = vals[vdc] if vdc else (-p_dc_ref_mw if mode == "droop" else 0.0)
    idx = pp.create_vsc(net, bus, bus_dc, r_ohm=0.0, x_ohm=0.0, r_dc_ohm=0.0,
                        control_mode_ac=mac, control_value_ac=-vals[vac] if mac == "q_mvar" else vals[vac],
                        control_mode_dc=mdc, control_value_dc=value_dc, name=name)
    for col, default in CONV_COLUMNS.items():
        if col not in net.vsc.columns:
            net.vsc[col] = default
        net.vsc.at[idx, col] = vals[col]
    return idx


def expand_converter_impedances(net):
    """Replace non-zero vsc r/x (AC) and r_dc (DC) by an ordinary line / line_dc plus a new bus.

    The converter then connects to the new bus. Works in place on `net`.
    """
    for i, row in net.vsc.iterrows():
        r, x, rdc = (float(row.get(c, 0.0) or 0.0) for c in ("r_ohm", "x_ohm", "r_dc_ohm"))
        if r != 0.0 or x != 0.0:
            b = pp.create_bus(net, net.bus.at[row.bus, "vn_kv"], name=f"{row['name'] or i} conv ac")
            pp.create_line_from_parameters(net, row.bus, b, 1.0, r, x, 0.0, 1e6,
                                           name=f"{row['name'] or i} conv impedance")
            net.vsc.at[i, "bus"] = b
            net.vsc.loc[i, ["r_ohm", "x_ohm"]] = 0.0
        if rdc != 0.0:
            d = pp.create_bus_dc(net, net.bus_dc.at[row.bus_dc, "vn_kv"], name=f"{row['name'] or i} conv dc")
            pp.create_line_dc_from_parameters(net, row.bus_dc, d, 1.0, rdc, 1e6,
                                              name=f"{row['name'] or i} conv dc resistance")
            net.vsc.at[i, "bus_dc"] = d
            net.vsc.at[i, "r_dc_ohm"] = 0.0
    return net


def _native_setpoints(row):
    """Engine mode and setpoints from pandapower's control columns (load convention there)."""
    mac, vac = row.control_mode_ac, float(row.control_value_ac)
    mdc, vdc = row.control_mode_dc, float(row.control_value_dc)
    if mac == "slack":
        return {"mode": "GF", "vm_ac_pu": vac, "va_degree": 0.0}
    if mdc == "vm_pu" and mac == "q_mvar":
        return {"mode": "VdcQ", "q_ac_mvar": -vac, "vm_dc_pu": vdc}
    if mdc == "p_mw" and mac == "q_mvar":
        return {"mode": "PQ", "p_ac_mw": vdc, "q_ac_mvar": -vac}
    if mdc == "p_mw" and mac == "vm_pu":
        return {"mode": "PV", "p_ac_mw": vdc, "vm_ac_pu": vac}
    if mdc == "vm_pu" and mac == "vm_pu":
        return {"mode": "VdcV", "vm_dc_pu": vdc, "vm_ac_pu": vac}
    raise ValueError(f"vsc {row.name}: control modes ({mac}, {mdc}) are not supported; set the 'mode' column")


def ac_ppc(net_ac):
    """pandapower's full ppc of an AC network: all buses in lookup order, 26 branch columns (keeps BR_G).

    Uses pandapower internals (pinned version); to_ppc() would drop BR_G. Lookups land in
    net_ac._pd2ppc_lookups.
    """
    net_ac["_options"] = {}
    _add_ppc_options(net_ac, calculate_voltage_angles=True, trafo_model="t", check_connectivity=True,
                     mode="pf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=False, enforce_p_lims=False, recycle=None, voltage_depend_loads=False)
    ppc, _ = _pd2ppc(net_ac)
    return ppc


def _converter_table(net, bus_ok, bus_dc_ok):
    vsc = net.vsc[net.vsc.in_service.astype(bool)] if "vsc" in net and len(net.vsc) else None
    if vsc is None or len(vsc) == 0:
        return None
    vsc = vsc[vsc.bus.isin(bus_ok) & vsc.bus_dc.isin(bus_dc_ok)].copy()  # on out-of-service buses: off
    if len(vsc) == 0:
        return None
    if "mode" not in vsc.columns:
        vsc["mode"] = np.nan
    for i in vsc.index[vsc["mode"].isna()]:  # plain pandapower vsc rows
        for col, val in _native_setpoints(vsc.loc[i]).items():
            if col not in vsc.columns:
                vsc[col] = np.nan
            vsc.loc[i, col] = val
    for col, default in CONV_COLUMNS.items():
        if col not in vsc.columns:
            vsc[col] = default
        vsc[col] = vsc[col].fillna(default)
    bad = set(vsc["mode"]) - set(MODES)
    if bad:
        raise ValueError(f"unknown converter mode(s) {bad}; use one of {MODES}")
    return vsc


def build_model(net):
    """Classify nodes, compute per-unit injections and converter data. Call build_ybus() next."""
    net = expand_converter_impedances(deepcopy(net))
    bus_ok = set(net.bus.index[net.bus.in_service.astype(bool)])
    bus_dc_ok = set(net.bus_dc.index[net.bus_dc.in_service.astype(bool)]) if "bus_dc" in net else set()
    vsc = _converter_table(net, bus_ok, bus_dc_ok)
    base = float(net.sn_mva)

    # ---- AC part: pandapower's per-unit model (ppc) of the AC-only network ---------------------------------
    # s_spec = gen injections - loads (sgens as negative loads), p.u.; shunts, taps, shifts go into Ybus.
    net_ac = deepcopy(net)
    for tab in ("vsc", "bus_dc", "line_dc", "load_dc", "source_dc"):
        if tab in net_ac:
            net_ac[tab] = net_ac[tab].iloc[0:0]
    gf = vsc[vsc["mode"] == "GF"] if vsc is not None else []
    for _, c in (gf.iterrows() if len(gf) else []):
        # temporary slack keeps the island energised in pandapower's connectivity check
        pp.create_ext_grid(net_ac, c.bus, vm_pu=c.vm_ac_pu, va_degree=c.va_degree)
    ppc = ac_ppc(net_ac)
    lookup = net_ac._pd2ppc_lookups["bus"]
    bus_lookup = {int(b): int(lookup[b]) for b in net.bus.index[net.bus.in_service.astype(bool)]}
    bus = ppc["bus"]
    n_ac = bus.shape[0]
    ac_bus_id = np.full(n_ac, -1, dtype=int)
    for b, r in sorted(bus_lookup.items(), reverse=True):
        ac_bus_id[r] = b

    gen = ppc["gen"]
    on = gen[:, GEN_STATUS] > 0
    gbus = gen[on, GEN_BUS].astype(int)
    s_spec = (np.bincount(gbus, gen[on, PG], n_ac) + 1j * np.bincount(gbus, gen[on, QG], n_ac)
              - (bus[:, PD] + 1j * bus[:, QD])) / base

    ac_ref = np.zeros(n_ac, bool)
    v_set = np.ones(n_ac)
    va_set = np.zeros(n_ac)
    iso = bus[:, BUS_TYPE] == NONE
    ac_ref[iso] = True
    v_set[iso] = 0.0
    ac_pv = np.zeros(n_ac, bool)
    ac_pv[gbus] = True
    v_set[gbus] = gen[on, VG]
    ref = bus[:, BUS_TYPE] == REF  # ext_grids and slack gens as pandapower sees them
    ac_ref[ref], va_set[ref] = True, np.deg2rad(bus[ref, VA])
    eg = net.ext_grid[net.ext_grid.in_service.astype(bool) & net.ext_grid.bus.isin(bus_ok)]
    for _, e in eg.iterrows():
        r = bus_lookup[int(e.bus)]
        ac_ref[r], v_set[r], va_set[r] = True, e.vm_pu, np.deg2rad(e.va_degree)

    # ---- DC part: buses, line conductances g = 1/r (p.u. on sn_mva and the bus_dc base), loads, sources ------
    bdc = net.bus_dc[net.bus_dc.in_service.astype(bool)] if "bus_dc" in net else net.bus.iloc[0:0]
    dc_bus_id = np.asarray(bdc.index, dtype=int)
    bus_dc_lookup = {int(b): i for i, b in enumerate(dc_bus_id)}
    n_dc = len(dc_bus_id)
    zb = (bdc.vn_kv.values ** 2 / base) if n_dc else np.zeros(0)
    fr, to, g, gsh = [], [], [], []
    if n_dc and len(net.line_dc):
        for _, ln in net.line_dc[net.line_dc.in_service.astype(bool)].iterrows():
            f, t = bus_dc_lookup.get(int(ln.from_bus_dc)), bus_dc_lookup.get(int(ln.to_bus_dc))
            if f is None or t is None:
                continue
            par = ln.get("parallel", 1) or 1
            r_ohm = ln.r_ohm_per_km * ln.length_km / par
            fr.append(f); to.append(t); g.append(zb[f] / r_ohm)
            gsh.append((ln.get("g_us_per_km", 0.0) or 0.0) * 1e-6 * ln.length_km * par * zb[f])
    dc_lines = (np.array(fr, int), np.array(to, int), np.array(g, float), np.array(gsh, float))

    p_dc_spec = np.zeros(n_dc)
    if n_dc and "load_dc" in net and len(net.load_dc):
        for _, ld in net.load_dc[net.load_dc.in_service.astype(bool) & net.load_dc.bus_dc.isin(bus_dc_ok)].iterrows():
            p_dc_spec[bus_dc_lookup[int(ld.bus_dc)]] -= ld.p_dc_mw * ld.get("scaling", 1.0) / base
    dc_fixed = np.zeros(n_dc, bool)
    vdc_set = np.ones(n_dc)
    if n_dc and "source_dc" in net and len(net.source_dc):
        for _, s in net.source_dc[net.source_dc.in_service.astype(bool)
                                  & net.source_dc.bus_dc.isin(bus_dc_ok)].iterrows():
            k = bus_dc_lookup[int(s.bus_dc)]
            if dc_fixed[k]:
                raise ValueError(f"DC bus {s.bus_dc} has two voltage-fixing elements")
            dc_fixed[k], vdc_set[k] = True, s.vm_pu
    dc_source = dc_fixed.copy()

    # ---- converters -------------------------------------------------------------------------------
    conv = {k: np.zeros(0) for k in ("p", "q", "vac", "va", "vdc", "pdc", "k", "vref", "la", "lb", "lc")}
    conv["ac"], conv["dc"] = np.zeros(0, int), np.zeros(0, int)
    conv["mode"] = np.zeros(0, dtype=object)
    conv["name"] = []
    conv["index"] = np.zeros(0, int)
    if vsc is not None and len(vsc):
        ac_nodes = np.array([bus_lookup[int(b)] for b in vsc.bus])
        dc_nodes = np.array([bus_dc_lookup[int(b)] for b in vsc.bus_dc])
        for nodes, what in ((ac_nodes, "AC bus"), (dc_nodes, "DC bus")):
            u, cnt = np.unique(nodes, return_counts=True)
            if np.any(cnt > 1):
                raise ValueError(f"more than one converter on one {what} (v1 restriction): nodes {u[cnt > 1]}")
        if np.any(iso[ac_nodes]):
            raise ValueError("converter AC bus is not energised: AC island without slack or grid-forming converter")
        # a slack on the converter bus leaves no equation for the converter; a PV generator is fine for
        # PQ / VdcQ / droop converters (the generator holds |E|, the converter injects its Q setpoint)
        slack_nodes = {bus_lookup[int(b)] for b in eg.bus}
        gen_nodes = {bus_lookup[int(b)] for b in net.gen.bus[net.gen.in_service.astype(bool) & net.gen.bus.isin(bus_ok)]}
        for n, (i, c) in zip(ac_nodes, vsc.iterrows()):
            if n in slack_nodes:
                raise ValueError(f"converter {i} shares its AC bus with an ext_grid")
            if n in gen_nodes and c["mode"] in ("PV", "VdcV", "GF"):
                raise ValueError(f"{c['mode']} converter {i} shares its AC bus with a voltage-controlling gen")
        modes = vsc["mode"].to_numpy(dtype=object)
        conv = dict(
            index=np.asarray(vsc.index, int), name=list(vsc["name"]), mode=modes, ac=ac_nodes, dc=dc_nodes,
            p=vsc.p_ac_mw.to_numpy(float) / base, q=vsc.q_ac_mvar.to_numpy(float) / base,
            vac=vsc.vm_ac_pu.to_numpy(float), va=np.deg2rad(vsc.va_degree.to_numpy(float)),
            vdc=vsc.vm_dc_pu.to_numpy(float), pdc=vsc.p_dc_ref_mw.to_numpy(float) / base,
            k=vsc.droop_k.to_numpy(float), vref=vsc.vdc_ref_pu.to_numpy(float),
            la=vsc.loss_a.to_numpy(float), lb=vsc.loss_b.to_numpy(float), lc=vsc.loss_c.to_numpy(float),
        )
        for i, m in enumerate(modes):
            l, k = ac_nodes[i], dc_nodes[i]
            if m in ("VdcQ", "VdcV"):
                if m == "VdcV":  # AC voltage held through the converter's row 2
                    v_set[l] = conv["vac"][i]
                if k not in dc_lines[0] and k not in dc_lines[1]:
                    raise ValueError(f"{m} converter {conv['index'][i]}: DC bus {dc_bus_id[k]} has no DC line")
                if dc_fixed[k]:
                    raise ValueError(f"DC bus {dc_bus_id[k]} has two voltage-fixing elements")
                dc_fixed[k], vdc_set[k] = True, conv["vdc"][i]
            elif m == "GF":
                ac_ref[l], v_set[l], va_set[l] = True, conv["vac"][i], conv["va"][i]
                ac_pv[l] = False
            elif m == "PV":  # voltage held through the converter's row 2, the bus stays a PQ-type node
                v_set[l] = conv["vac"][i]
    _check_dc_islands(n_dc, dc_lines, dc_fixed, conv, dc_bus_id)

    names = net.bus.name.reindex(ac_bus_id).tolist()
    dc_names = bdc.name.tolist() if n_dc else []
    return Model(base_mva=base, n_ac=n_ac, n_dc=n_dc, ac_bus_id=ac_bus_id, dc_bus_id=dc_bus_id,
                 bus_lookup=bus_lookup, bus_dc_lookup=bus_dc_lookup, ac_ref=ac_ref, ac_pv=ac_pv & ~ac_ref,
                 v_set=v_set, va_set=va_set, s_spec=s_spec, dc_fixed=dc_fixed, dc_source=dc_source, vdc_set=vdc_set,
                 p_dc_spec=p_dc_spec, conv=conv, ac_names=names,
                 dc_names=dc_names, _ppc=ppc, _dc_lines=dc_lines)


def _check_dc_islands(n_dc, dc_lines, dc_fixed, conv, dc_bus_id):
    if n_dc == 0:
        return
    fr, to, _, _ = dc_lines
    A = sp.coo_matrix((np.ones(len(fr)), (fr, to)), shape=(n_dc, n_dc))
    n_isl, lab = connected_components(A, directed=False)
    droop = np.zeros(n_dc, bool)
    for m, k, kd in zip(conv["mode"], conv["dc"], conv["k"]):
        if m == "droop" and kd > 0:
            droop[k] = True
    for isl in range(n_isl):
        nodes = lab == isl
        if not (dc_fixed[nodes].any() or droop[nodes].any()):
            raise ValueError(f"DC island {dc_bus_id[nodes].tolist()} has no voltage-controlling element "
                             "(source_dc, VdcQ converter or droop converter with k > 0)")


def build_ybus(model):
    """Fill model.Y (AC nodal admittance matrix, 3.2; pandapower makeYbus incl. taps, phase shifters, shunts,
    line charging) and model.G (DC nodal conductance matrix)."""
    ppc = model._ppc
    branch = ppc["branch"]
    if branch.shape[1] < branch_cols:
        branch = np.hstack([branch, np.zeros((branch.shape[0], branch_cols - branch.shape[1]))])
    Y, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], branch)
    model.Y = sp.csr_matrix(Y)
    fr, to, g, gsh = model._dc_lines
    n = model.n_dc
    rows = np.r_[fr, to, fr, to, fr, to]
    cols = np.r_[fr, to, to, fr, fr, to]
    vals = np.r_[g, g, -g, -g, gsh / 2, gsh / 2]
    model.G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return model
