"""Case files in YAML: MATPOWER-style tables, one row per element, complete with transformers and the DC side.

    bus        id, name, base_kv, pd_mw, qd_mvar, gs_mw, bs_mvar, in_service   (loads/shunts as in MATPOWER)
    slack      bus, vm_pu, va_deg, in_service
    gen        bus, pg_mw, vg_pu, in_service                                   (PV generators)
    line       from, to, r_pu, x_pu, b_pu, in_service                          (b_pu: total charging)
    trafo      from, to, r_pu, x_pu, g_pu, b_pu, ratio, shift_deg, in_service  (MATPOWER branch model: tap at
                                                                                'from', ratio 0 means 1, g/b
                                                                                split half-half, from side /ratio^2)
    bus_dc     id, name, base_kv, pd_mw, in_service                            (pd_mw: DC load)
    line_dc    from, to, r_pu, in_service
    source_dc  bus, vm_pu, in_service                                          (DC voltage source)
    converter  name, ac_bus, dc_bus, mode, p_mw, q_mvar, vac_pu, va_deg, vdc_pu,
               p_dc_ref_mw, droop_k, vdc_ref_pu, loss_a, loss_b, loss_c, in_service

Per unit on base_mva and the bus base_kv (from-bus base for lines). Converters connect directly to ac_bus and
dc_bus (no extra busbar nodes needed). Converter modes and the columns they use:
    PQ     p_mw, q_mvar          (injection into the AC grid)
    PV     p_mw, vac_pu
    VdcQ   vdc_pu, q_mvar
    VdcV   vdc_pu, vac_pu
    GF     vac_pu, va_deg        (grid-forming)
    droop  p_dc_ref_mw, droop_k, vdc_ref_pu, q_mvar:
           P_dc = p_dc_ref_mw - droop_k * (E_dc - vdc_ref_pu) * base_mva, P_dc injected into the DC grid
           (rectifier positive), droop_k = 1 / droop (e.g. 5 % droop -> 20)
Converter losses: P_loss = loss_a + loss_b |I| + loss_c |I|^2 (p.u.).

read_case() builds a pandapower net (lines, 2-winding transformers with a ratio tap, loads, shunts, gens,
ext_grid, bus_dc, line_dc, load_dc, source_dc, vsc via add_converter). write_case() goes the other way.
"""
import json
from copy import deepcopy

import numpy as np
import pandapower as pp
import yaml
from pandapower.pypower.idx_brch import (BR_B, BR_B_ASYM, BR_G_ASYM, BR_R, BR_R_ASYM, BR_STATUS, BR_X,
                                         BR_X_ASYM, F_BUS, SHIFT, T_BUS, TAP)
from pandapower.pypower.idx_bus import BS, GS, PD, QD

from .network import CONV_COLUMNS, _native_setpoints, ac_ppc, add_converter, expand_converter_impedances

BR_G = 23  # pandapower branch column: series/shunt conductance (must be 0 here)

TABLES = {
    "bus": ["id", "name", "base_kv", "pd_mw", "qd_mvar", "gs_mw", "bs_mvar", "in_service"],
    "slack": ["bus", "vm_pu", "va_deg", "in_service"],
    "gen": ["bus", "pg_mw", "vg_pu", "in_service"],
    "line": ["from", "to", "r_pu", "x_pu", "b_pu", "in_service"],
    "trafo": ["from", "to", "r_pu", "x_pu", "g_pu", "b_pu", "ratio", "shift_deg", "in_service"],
    "bus_dc": ["id", "name", "base_kv", "pd_mw", "in_service"],
    "line_dc": ["from", "to", "r_pu", "in_service"],
    "source_dc": ["bus", "vm_pu", "in_service"],
    "converter": ["name", "ac_bus", "dc_bus", "mode", "p_mw", "q_mvar", "vac_pu", "va_deg", "vdc_pu",
                  "p_dc_ref_mw", "droop_k", "vdc_ref_pu", "loss_a", "loss_b", "loss_c", "in_service"],
}
MAX_I_KA = 99.0  # thermal limits are not part of the case files yet


# ---------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------
def read_case(path):
    """YAML case file -> pandapower net."""
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    t = {name: _rows(doc, name) for name in TABLES}
    base = float(doc["base_mva"])
    f_hz = float(doc.get("f_hz", 50.0))
    net = pp.create_empty_network(name=doc.get("name") or "", sn_mva=base, f_hz=f_hz)

    kv = {}
    for r in t["bus"]:
        pp.create_bus(net, float(r["base_kv"]), name=r["name"], index=int(r["id"]), in_service=bool(r["in_service"]))
        kv[int(r["id"])] = float(r["base_kv"])
        if r["pd_mw"] or r["qd_mvar"]:
            pp.create_load(net, int(r["id"]), p_mw=float(r["pd_mw"]), q_mvar=float(r["qd_mvar"]))
        if r["gs_mw"] or r["bs_mvar"]:
            pp.create_shunt(net, int(r["id"]), q_mvar=-float(r["bs_mvar"]), p_mw=float(r["gs_mw"]),
                            vn_kv=float(r["base_kv"]))
    for r in t["slack"]:
        pp.create_ext_grid(net, int(r["bus"]), vm_pu=float(r["vm_pu"]), va_degree=float(r["va_deg"]),
                           in_service=bool(r["in_service"]))
    for r in t["gen"]:
        pp.create_gen(net, int(r["bus"]), p_mw=float(r["pg_mw"]), vm_pu=float(r["vg_pu"]),
                      in_service=bool(r["in_service"]))
    if t["line"]:
        f = np.array([int(r["from"]) for r in t["line"]])
        to = np.array([int(r["to"]) for r in t["line"]])
        bad = [(a, b) for a, b in zip(f, to) if kv[a] != kv[b]]
        if bad:
            raise ValueError(f"lines between buses with different base_kv {bad[:3]}: put them in 'trafo'")
        zb = np.array([kv[a] ** 2 / base for a in f])
        col = lambda c: np.array([float(r[c]) for r in t["line"]])  # noqa: E731
        pp.create_lines_from_parameters(
            net, f, to, 1.0, col("r_pu") * zb, col("x_pu") * zb,
            col("b_pu") / zb / (2 * np.pi * f_hz) * 1e9, MAX_I_KA,
            in_service=np.array([bool(r["in_service"]) for r in t["line"]]))
    for r in t["trafo"]:
        a, b = int(r["from"]), int(r["to"])
        rr, xx = float(r["r_pu"]), float(r["x_pu"])
        if rr < 0 or xx < 0:
            raise ValueError(f"trafo {a}-{b}: negative r or x cannot be a pandapower transformer")
        ratio = float(r["ratio"]) or 1.0
        step = (ratio - 1.0) * 100.0
        pp.create_transformer_from_parameters(
            net, a, b, sn_mva=base, vn_hv_kv=kv[a], vn_lv_kv=kv[b], vkr_percent=rr * 100.0,
            vk_percent=abs(complex(rr, xx)) * 100.0, pfe_kw=0.0, i0_percent=0.0,
            shift_degree=float(r["shift_deg"]), tap_side="hv", tap_neutral=0, tap_min=-1, tap_max=1,
            tap_step_percent=abs(step), tap_pos=1 if step >= 0 else -1, tap_changer_type="Ratio",
            in_service=bool(r["in_service"]))
        g, bb = float(r["g_pu"]), float(r["b_pu"])
        if g or bb:  # branch charging: (g + jb)/2 at each end, from side divided by ratio^2 (as in makeYbus)
            for bus_, scale in ((a, 1.0 / ratio**2), (b, 1.0)):
                pp.create_shunt(net, bus_, q_mvar=-bb / 2 * scale * base, p_mw=g / 2 * scale * base,
                                vn_kv=kv[bus_], name=f"trafo {a}-{b} charging", in_service=bool(r["in_service"]))

    kv_dc = {}
    for r in t["bus_dc"]:
        i = int(r["id"])
        pp.create_bus_dc(net, float(r["base_kv"]), name=r["name"], index=i, in_service=bool(r["in_service"]))
        kv_dc[i] = float(r["base_kv"])
        if r["pd_mw"]:
            pp.create_load_dc(net, i, p_dc_mw=float(r["pd_mw"]))
    for r in t["line_dc"]:
        a = int(r["from"])
        pp.create_line_dc_from_parameters(net, a, int(r["to"]), 1.0, float(r["r_pu"]) * kv_dc[a] ** 2 / base,
                                          MAX_I_KA, in_service=bool(r["in_service"]))
    for r in t["source_dc"]:
        pp.create_source_dc(net, int(r["bus"]), vm_pu=float(r["vm_pu"]), in_service=bool(r["in_service"]))
    for r in t["converter"]:
        j = add_converter(net, int(r["ac_bus"]), int(r["dc_bus"]), r["mode"], p_ac_mw=float(r["p_mw"]),
                          q_ac_mvar=float(r["q_mvar"]), vm_ac_pu=float(r["vac_pu"]), va_degree=float(r["va_deg"]),
                          vm_dc_pu=float(r["vdc_pu"]), p_dc_ref_mw=float(r["p_dc_ref_mw"]),
                          droop_k=float(r["droop_k"]), vdc_ref_pu=float(r["vdc_ref_pu"]),
                          loss_a=float(r["loss_a"]), loss_b=float(r["loss_b"]), loss_c=float(r["loss_c"]),
                          name=r["name"])
        net.vsc.at[j, "in_service"] = bool(r["in_service"])
    return net


def _rows(doc, name):
    tab = doc.get(name) or {}
    cols = tab.get("columns", TABLES[name])
    missing = set(TABLES[name]) - set(cols)
    if missing:
        raise ValueError(f"table '{name}' misses columns {sorted(missing)}")
    out = []
    for i, row in enumerate(tab.get("rows") or []):
        if len(row) != len(cols):
            raise ValueError(f"table '{name}', row {i + 1}: {len(row)} values for {len(cols)} columns")
        out.append(dict(zip(cols, row)))
    return out


# ---------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------
def write_case(net, path, title=None):
    """pandapower net -> YAML case file (branch data taken from pandapower's own per-unit model)."""
    net = expand_converter_impedances(deepcopy(net))
    base = float(net.sn_mva)
    rows = {name: [] for name in TABLES}

    # AC per-unit data from pandapower's ppc (loads, shunts, taps, shifts exactly as pandapower models them)
    net_ac = deepcopy(net)
    for tab in ("vsc", "bus_dc", "line_dc", "load_dc", "source_dc"):
        if tab in net_ac:
            net_ac[tab] = net_ac[tab].iloc[0:0]
    for other in ("trafo3w", "impedance", "xward", "ward", "dcline", "switch"):
        if other in net_ac and len(net_ac[other]):
            raise NotImplementedError(f"'{other}' elements cannot be written to a YAML case file yet")
    voltage_held = set(net.gen.bus) | set(net.ext_grid.bus)
    for b in set(net.vsc.bus if "vsc" in net else []) - voltage_held:  # keep converter-fed islands energised
        pp.create_ext_grid(net_ac, b)
    ppc = ac_ppc(net_ac)
    lookup = net_ac._pd2ppc_lookups["bus"]
    bus, br = ppc["bus"], ppc["branch"]
    row_to_id = {}
    for b, r in net.bus.iterrows():
        on = bool(r.in_service)
        p = bus[lookup[b]] if on else None
        if on:
            row_to_id[int(lookup[b])] = int(b)
        rows["bus"].append([int(b), r["name"], r.vn_kv,
                            *(p[[PD, QD, GS, BS]] if on else (0.0, 0.0, 0.0, 0.0)), on])
    for _, e in net.ext_grid.iterrows():
        rows["slack"].append([int(e.bus), e.vm_pu, e.va_degree, bool(e.in_service)])
    for _, g in net.gen.iterrows():
        rows["gen"].append([int(g.bus), g.p_mw * g.get("scaling", 1.0), g.vm_pu, bool(g.in_service)])
    ranges = net_ac._pd2ppc_lookups.get("branch", {})
    for kind in ("line", "trafo"):
        if kind not in ranges:
            continue
        for k in range(*ranges[kind]):
            if np.any(np.abs(br[k, [BR_R_ASYM, BR_X_ASYM, BR_G_ASYM, BR_B_ASYM]]) > 1e-12):
                raise NotImplementedError(f"{kind} with asymmetric branch data: not in the format")
            if kind == "line" and abs(br[k, BR_G]) > 1e-12:
                raise NotImplementedError("line with shunt conductance: not in the format")
            f, t = row_to_id[int(br[k, F_BUS].real)], row_to_id[int(br[k, T_BUS].real)]
            on = bool(br[k, BR_STATUS].real)
            if kind == "line":
                rows["line"].append([f, t, br[k, BR_R].real, br[k, BR_X].real, br[k, BR_B].real, on])
            else:
                rows["trafo"].append([f, t, br[k, BR_R].real, br[k, BR_X].real, br[k, BR_G].real,
                                      br[k, BR_B].real, br[k, TAP].real, br[k, SHIFT].real, on])

    if "bus_dc" in net:
        base_dc = net.bus_dc.vn_kv
        pd_dc = {}
        for _, ld in net.load_dc[net.load_dc.in_service.astype(bool)].iterrows():
            pd_dc[ld.bus_dc] = pd_dc.get(ld.bus_dc, 0.0) + ld.p_dc_mw * ld.get("scaling", 1.0)
        for b, r in net.bus_dc.iterrows():
            rows["bus_dc"].append([int(b), r["name"], r.vn_kv, pd_dc.get(b, 0.0), bool(r.in_service)])
        for _, ln in net.line_dc.iterrows():
            if (ln.get("g_us_per_km", 0.0) or 0.0) != 0.0:
                raise NotImplementedError("line_dc with shunt conductance: not in the format")
            r_ohm = ln.r_ohm_per_km * ln.length_km / (ln.get("parallel", 1) or 1)
            zb = base_dc[ln.from_bus_dc] ** 2 / base
            rows["line_dc"].append([int(ln.from_bus_dc), int(ln.to_bus_dc), r_ohm / zb, bool(ln.in_service)])
        for _, s in net.source_dc.iterrows():
            rows["source_dc"].append([int(s.bus_dc), s.vm_pu, bool(s.in_service)])
    for i, v in (net.vsc.iterrows() if "vsc" in net else []):
        vals = {c: v.get(c, d) for c, d in CONV_COLUMNS.items()}
        if not isinstance(vals["mode"], str):
            vals.update(_native_setpoints(v))
        vals = {c: (CONV_COLUMNS[c] if (not isinstance(x, str) and x is not None and np.isnan(x)) else x)
                for c, x in vals.items()}
        rows["converter"].append([v["name"] if isinstance(v["name"], str) else f"IC{i}", int(v.bus),
                                  int(v.bus_dc), vals["mode"], vals["p_ac_mw"], vals["q_ac_mvar"],
                                  vals["vm_ac_pu"], vals["va_degree"], vals["vm_dc_pu"], vals["p_dc_ref_mw"],
                                  vals["droop_k"], vals["vdc_ref_pu"], vals["loss_a"], vals["loss_b"],
                                  vals["loss_c"], bool(v.in_service)])

    with open(path, "w") as fh:
        fh.write(_header(title or net.name or "", base))
        fh.write(f"name: {_cell(net.name or '')}\nbase_mva: {_cell(base)}\nf_hz: {_cell(float(net.f_hz))}\n")
        for name, cols in TABLES.items():
            fh.write(_table(name, cols, rows[name]))


def _header(title, base):
    doc = __doc__.split("read_case()")[0].strip().splitlines()
    return "".join(f"# {line}\n" if line else "#\n" for line in [title, ""] + doc) + "\n"


def _cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "null"
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        s = f"{v:.12g}"
        if "e" in s and "." not in s.split("e")[0]:  # YAML 1.1 needs a dot for exponent floats
            m, e = s.split("e")
            s = f"{m}.0e{e}"
        return s
    s = str(v)
    try:
        plain = yaml.safe_load(s) == s and not any(ch in s for ch in ",[]{}#:&*!|>'\"%@`")
    except yaml.YAMLError:
        plain = False
    return s if plain and s.strip() == s and s else json.dumps(s)


def _table(name, cols, rows):
    cells = [[_cell(x) for x in r] for r in rows]
    w = [max([len(c)] + [len(r[j]) for r in cells]) for j, c in enumerate(cols)]
    out = [f"\n{name}:", f"  columns: [{', '.join(cols)}]"]
    if not cells:
        out.append("  rows: []")
        return "\n".join(out) + "\n"
    out.append("  rows:")
    out.append("  #  " + "  ".join(c.rjust(w[j]) for j, c in enumerate(cols)))
    out += ["  - [" + ", ".join(x.rjust(w[j]) for j, x in enumerate(r)) + "]" for r in cells]
    return "\n".join(out) + "\n"
