"""Benchmark: unified-acdc-powerflow vs pandapower 3.5.5's AC/DC power flow on the grids of the paper.

unified-acdc-powerflow connects every converter directly to its AC and DC bus. pandapower models a converter behind
a coupling impedance (with zero it stops with a division by zero), so it gets one of z = 1e-4 ... 1e-1 p.u. of the
bus base, and every chance: init "auto", "dc" and "flat", 100 iterations. Per grid and z:

    pandapower              pandapower on the grid as in the paper
    pp_dc_load_corrected    the same, with load_dc divided by sn_mva (pandapower 3.5.5 multiplies DC loads by sn_mva)
    pp_gen_bus_moved        the same, and each converter that shares its AC bus with a voltage-controlling generator
                            moved one short branch away (unified-acdc-powerflow gets this grid too, for dv_*)
    dv_*_as_is_pu           largest AC / DC voltage difference to unified-acdc-powerflow, pandapower as is
    dv_ac_pu, dv_dc_pu      the same for the last pandapower variant above that converges (on the same grid)
A result with an AC or DC voltage outside 0.8-1.2 p.u. counts as spurious. A control run replaces each converter by
a fixed P/Q injection from unified-acdc-powerflow's solution: pandapower then solves every AC grid, so the AC data
are fine.

Run:  .venv/bin/python benchmarks/pandapower_acdc/run_benchmark.py
Writes results.csv, and to reproducer/ the paper grids at z = 1e-3 p.u. plus a minimal DC-load case, as plain
pandapower JSON (only pandapower's own vsc columns) that reproduce.py runs with pandapower alone.
"""
import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd

from unified_acdc_powerflow import run_pf
from unified_acdc_powerflow.cases import load_case

logging.getLogger("pandapower").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ENGINE_COLUMNS = ("mode", "p_ac_mw", "q_ac_mvar", "vm_ac_pu", "va_degree", "vm_dc_pu", "p_dc_ref_mw", "droop_k",
                  "vdc_ref_pu", "loss_a", "loss_b", "loss_c", "ace_bus", "ace_k_mw_per_deg")
Z_PU = (1e-4, 1e-3, 1e-2, 1e-1)    # pandapower coupling impedance (r = x = r_dc), p.u. of the bus base
Z_REPRODUCER = 1e-3
INITS = ("auto", "dc", "flat")
PLAUSIBLE = (0.8, 1.2)             # AC and DC voltage band (p.u.) of a physically meaningful solution
GRIDS = [                          # (name in the table, case file)
    ("IEEE 57+14, 2 HVDC grids", "ieee57_14_hvdc"),
    ("IEEE 30, 2 HVDC grids", "ieee30_hvdc"),
    ("PEGASE 1354, 2 HVDC grids", "pegase_hvdc"),
    ("EPFL 26-bus AC/DC microgrid (EMTP-RV setpoints)", "microgrid_emtp"),
]


def gen_bus_converters_moved(net):
    """Each converter on the bus of a voltage-controlling generator moves to a new busbar one short branch away
    (r = 0.01, x = 1e-4 p.u.)."""
    gen_buses = set(net.gen.bus[net.gen.in_service.astype(bool)])
    for i, v in net.vsc.iterrows():
        if v.bus in gen_buses:
            zb = net.bus.vn_kv[v.bus] ** 2 / net.sn_mva
            b = pp.create_bus(net, net.bus.vn_kv[v.bus], name=f"busbar of {v['name']}")
            pp.create_line_from_parameters(net, v.bus, b, 1.0, 0.01 * zb, 1e-4 * zb, 0.0, 99.0)
            net.vsc.at[i, "bus"] = b
    return net


def pandapower_native(net, z_pu, dc_load_corrected=False):
    """Engine converter columns -> pandapower's own vsc control modes (load convention there)."""
    n = pp.from_json_string(pp.to_json(net))
    for i, v in n.vsc.iterrows():
        m = v["mode"]
        if m == "PQ":
            ctrl = ("q_mvar", -v.q_ac_mvar, "p_mw", v.p_ac_mw)
        elif m == "VdcQ":
            ctrl = ("q_mvar", -v.q_ac_mvar, "vm_pu", v.vm_dc_pu)
        elif m == "PV":
            ctrl = ("vm_pu", v.vm_ac_pu, "p_mw", v.p_ac_mw)
        elif m == "VdcV":
            ctrl = ("vm_pu", v.vm_ac_pu, "vm_pu", v.vm_dc_pu)
        elif m == "GF":
            ctrl = ("slack", v.vm_ac_pu, "p_mw", 0.0)
        else:
            raise ValueError(f"pandapower has no {m} mode")
        z_ac = z_pu * n.bus.vn_kv[v.bus] ** 2 / n.sn_mva
        z_dc = z_pu * n.bus_dc.vn_kv[v.bus_dc] ** 2 / n.sn_mva
        n.vsc.loc[i, ["control_mode_ac", "control_value_ac", "control_mode_dc", "control_value_dc"]] = ctrl
        n.vsc.loc[i, ["r_ohm", "x_ohm", "r_dc_ohm"]] = [z_ac, z_ac, z_dc]
    if dc_load_corrected and len(n.load_dc):
        n.load_dc["p_dc_mw"] = n.load_dc.p_dc_mw / n.sn_mva
    return n


def ac_only_with_injections(net, res):
    """Control run: AC grid only, every converter replaced by an sgen with the engine's converter P/Q."""
    n = pp.from_json_string(pp.to_json(net))
    for (i, v), p, q in zip(n.vsc.iterrows(), res.conv_p_ac, res.conv_q_ac):
        pp.create_sgen(n, v.bus, p_mw=p * n.sn_mva, q_mvar=q * n.sn_mva)
    for tab in ("vsc", "line_dc", "load_dc", "source_dc", "bus_dc"):
        n[tab] = n[tab].iloc[0:0]
    try:
        pp.runpp(n, max_iteration=100)
        return "converges"
    except Exception as err:
        return type(err).__name__


def run_pandapower(n):
    """Best pandapower outcome over the initialisations: (status, AC |V| per bus, DC V per bus)."""
    outcome = ("not converged", None, None)
    for init in INITS:
        try:
            pp.runpp(n, init=init, max_iteration=100, tolerance_mva=1e-8)
        except Exception as err:
            if outcome[0] == "not converged" and not str(err).startswith("Power Flow"):
                outcome = (f"error: {type(err).__name__}", None, None)
            continue
        vm, vdc = n.res_bus.vm_pu.copy(), n.res_bus_dc.vm_pu.copy()
        lo, hi = min(vm.min(), vdc.min()), max(vm.max(), vdc.max())
        if PLAUSIBLE[0] <= lo and hi <= PLAUSIBLE[1]:
            return ("converged", vm, vdc)
        outcome = (f"spurious (V {lo:.2f}..{hi:.2f} p.u.)", vm, vdc)
    return outcome


def voltage_difference(net, res, vm, vdc):
    """Largest |V| difference (AC, DC) between pandapower's solution and unified-acdc-powerflow's."""
    m = res.model
    dv_ac = max(abs(vm[b] - abs(res.E_ac[m.bus_lookup[b]])) for b in net.bus.index)
    dv_dc = max(abs(vdc[b] - res.E_dc[m.bus_dc_lookup[b]]) for b in net.bus_dc.index)
    return dv_ac, dv_dc


def main():
    for f in (HERE / "reproducer").glob("*.json"):
        f.unlink()
    export_minimal_dc_load_cases()
    rows = []
    for name, case in GRIDS:
        net = load_case(case)
        gen_buses = sorted(int(b) for b in set(net.vsc.bus) & set(net.gen.bus[net.gen.in_service.astype(bool)]))
        ours = run_pf(net)
        moved = gen_bus_converters_moved(load_case(case)) if gen_buses else None
        ours_moved = run_pf(moved) if gen_buses else None
        control = ac_only_with_injections(net, ours)
        for z in Z_PU:
            status, vm, vdc = run_pandapower(pandapower_native(net, z))
            if z == Z_REPRODUCER and status != "converged":
                export_reproducer(pandapower_native(net, z), f"{name} - as in the paper - z {z:g} pu")
            dv = voltage_difference(net, ours, vm, vdc) if status == "converged" else (np.nan, np.nan)
            dv_as_is = dv
            status_corr = "(no DC load)"
            if len(net.load_dc):
                status_corr, vm, vdc = run_pandapower(pandapower_native(net, z, dc_load_corrected=True))
                if status_corr == "converged":
                    dv = voltage_difference(net, ours, vm, vdc)
            status_moved = "(no converter on a generator bus)"
            if gen_buses:
                status_moved, vm, vdc = run_pandapower(pandapower_native(moved, z, dc_load_corrected=True))
                if status_moved == "converged":
                    dv = voltage_difference(moved, ours_moved, vm, vdc)
            rows.append(dict(grid=name, ac_buses=len(net.bus), dc_buses=len(net.bus_dc), converters=len(net.vsc),
                             dc_loads=len(net.load_dc), converters_on_gen_buses=" ".join(map(str, gen_buses)) or "-",
                             coupling_z_pu=z, pandapower=status, pp_dc_load_corrected=status_corr,
                             pp_gen_bus_moved=status_moved, dv_ac_as_is_pu=dv_as_is[0],
                             dv_dc_as_is_pu=dv_as_is[1], dv_ac_pu=dv[0], dv_dc_pu=dv[1],
                             unified=("converged" if ours.converged else ours.message),
                             unified_iterations=ours.iterations, unified_time_ms=round(ours.solve_time_s * 1e3, 1),
                             pp_ac_only_control=control))
            print({k: rows[-1][k] for k in ("grid", "coupling_z_pu", "pandapower", "pp_dc_load_corrected",
                                            "pp_gen_bus_moved", "dv_ac_pu")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False, float_format="%.3g")
    return df


def export_minimal_dc_load_cases():
    """Two AC buses, one converter holding the DC voltage, a 5 MW DC load; once with sn_mva = 1 and once with 100."""
    for sn in (1.0, 100.0):
        net = pp.create_empty_network(sn_mva=sn, name=f"minimal DC-load case, sn_mva = {sn:g}")
        b0, b1 = pp.create_bus(net, 110.0), pp.create_bus(net, 110.0)
        pp.create_ext_grid(net, b0)
        pp.create_line_from_parameters(net, b0, b1, 10.0, 0.05, 0.4, 10.0, 1.0)
        d0, d1 = pp.create_bus_dc(net, 100.0), pp.create_bus_dc(net, 100.0)
        pp.create_line_dc_from_parameters(net, d0, d1, 10.0, 0.05, 1.0)
        pp.create_load_dc(net, d1, p_dc_mw=5.0)
        pp.create_vsc(net, b1, d0, r_ohm=0.1, x_ohm=1.0, r_dc_ohm=0.1, control_mode_ac="q_mvar",
                      control_value_ac=0.0, control_mode_dc="vm_pu", control_value_dc=1.0)
        pp.to_json(net, str(HERE / "reproducer" / f"minimal_dc_load_sn_mva_{sn:g}.json"))


def export_reproducer(native, label):
    """Save a grid as plain pandapower JSON (engine columns removed)."""
    net = pp.from_json_string(pp.to_json(native))
    net.vsc = net.vsc.drop(columns=[c for c in ENGINE_COLUMNS if c in net.vsc])
    for tab in [t for t in net.keys() if t.startswith("res_")]:
        net[tab] = net[tab].iloc[0:0]
    net.name = label
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    pp.to_json(net, str(HERE / "reproducer" / f"{slug}.json"))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
