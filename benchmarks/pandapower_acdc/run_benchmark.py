"""Benchmark: pandapower's AC/DC power flow vs unified-acdc-powerflow on the same physical systems.

The engine connects the converters directly. pandapower cannot (it needs a coupling impedance at each converter),
so it gets a small one, z in p.u. of the AC / DC bus base; its model then equals the direct connection up to O(z). pandapower gets
every chance: init "auto", "dc" and "flat", 100 iterations. A control run replaces each converter by a fixed
P/Q injection (taken from the engine's solution) to show that pandapower solves the AC part itself.

Run:  .venv/bin/python benchmarks/pandapower_acdc/run_benchmark.py
Writes results.csv / results.md, and every grid where pandapower fails as a plain pandapower JSON file (only
pandapower's own vsc columns) to reproducer/, where reproduce.py runs them with pandapower alone.
"""
import re
import logging
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
                  "vdc_ref_pu", "loss_a", "loss_b", "loss_c")
Z_PU = (1e-4, 1e-3, 1e-2)          # pandapower coupling impedance (r = x = r_dc), p.u. of the bus base
INITS = ("auto", "dc", "flat")
PLAUSIBLE = (0.8, 1.2)             # AC and DC voltage band (p.u.) of a physically meaningful solution


def with_busbars(net):
    """Busbar layout: every converter behind a busbar node (AC: r = 0.01, x = 1e-4 p.u.; DC: r = 0.01 p.u.)."""
    for i, v in net.vsc.iterrows():
        zb = net.bus.vn_kv[v.bus] ** 2 / net.sn_mva
        b = pp.create_bus(net, net.bus.vn_kv[v.bus])
        pp.create_line_from_parameters(net, v.bus, b, 1.0, 0.01 * zb, 1e-4 * zb, 0.0, 99.0)
        d = pp.create_bus_dc(net, net.bus_dc.vn_kv[v.bus_dc])
        pp.create_line_dc_from_parameters(net, v.bus_dc, d, 1.0, 0.01 * net.bus_dc.vn_kv[v.bus_dc] ** 2 / net.sn_mva,
                                          99.0)
        net.vsc.at[i, "bus"], net.vsc.at[i, "bus_dc"] = b, d
    return net


# (grid, scenario, net factory, coupling impedances given to pandapower). The engine always runs with the
# converters connected directly; only pandapower gets the coupling impedance.
SCENARIOS = [
    ("IEEE 57+14, 2 HVDC grids", "paper case, converters on grid buses", lambda: load_case("ieee57_14_hvdc"), (1e-3,)),
    ("IEEE 57+14, 2 HVDC grids", "busbar layout", lambda: with_busbars(load_case("ieee57_14_hvdc")), (1e-3,)),
    ("IEEE 30, 2 HVDC grids", "paper case, converters on grid buses", lambda: load_case("ieee30_hvdc"), (1e-3,)),
    ("IEEE 30, 2 HVDC grids", "busbar layout", lambda: with_busbars(load_case("ieee30_hvdc")), (1e-3,)),
    ("PEGASE 1354, 2 HVDC grids", "paper case, converters on grid buses", lambda: load_case("pegase_hvdc"), (1e-3,)),
    ("PEGASE 1354, 2 HVDC grids", "busbar layout", lambda: with_busbars(load_case("pegase_hvdc")), (1e-3,)),
    ("EPFL 26-bus AC/DC microgrid", "EMTP-RV setpoints", lambda: load_case("microgrid_emtp"), Z_PU),
]


def pandapower_native(net, z_pu):
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
    """Best pandapower outcome over the initialisations: (status, init, seconds, vm_pu per bus)."""
    outcome = ("not converged", "-", np.nan, None)
    for init in INITS:
        t = time.perf_counter()
        try:
            pp.runpp(n, init=init, max_iteration=100, tolerance_mva=1e-8)
        except Exception as err:
            if outcome[0] == "not converged" and not str(err).startswith("Power Flow"):
                outcome = (f"error: {type(err).__name__}", init, time.perf_counter() - t, None)
            continue
        dt = time.perf_counter() - t
        vm, vdc = n.res_bus.vm_pu, n.res_bus_dc.vm_pu
        lo, hi = min(vm.min(), vdc.min()), max(vm.max(), vdc.max())
        if PLAUSIBLE[0] <= lo and hi <= PLAUSIBLE[1]:
            return ("converged", init, dt, vm.copy())
        outcome = (f"spurious (V {lo:.2f}..{hi:.2f} p.u.)", init, dt, vm.copy())
    return outcome


def main():
    rows = []
    for name, feature, make, z_values in SCENARIOS:
        net = make()
        ours = run_pf(net)
        E = {b: abs(ours.E_ac[ours.model.bus_lookup[b]]) for b in net.bus.index}
        control = ac_only_with_injections(net, ours)
        for z in z_values:
            native = pandapower_native(net, z)
            status, init, dt, vm = run_pandapower(native)
            if status != "converged":
                export_reproducer(native, f"{name} - {feature} - z {z:g}")
            dv = np.nan if vm is None else max(abs(vm[b] - E[b]) for b in net.bus.index)
            # root causes: pandapower 3.5.5 scales load_dc by sn_mva (a bug; cancelled here), and cannot solve a
            # converter on the bus of a voltage-controlling generator
            corrected = pandapower_native(net, z)
            corrected.load_dc["p_dc_mw"] = corrected.load_dc.p_dc_mw / corrected.sn_mva
            status_corr = run_pandapower(corrected)[0] if len(net.load_dc) else "(no DC load)"
            rows.append(dict(case=name, scenario=feature, n_ac=len(net.bus), n_dc=len(net.bus_dc),
                             converters=len(net.vsc), dc_loads=len(net.load_dc),
                             converters_on_gen_bus=len(set(net.vsc.bus) & set(net.gen.bus)),
                             coupling_z_pu=z, pandapower=status, pp_dc_load_bug_cancelled=status_corr,
                             pp_init=init, pp_time_ms=dt * 1e3,
                             max_dV_vs_engine=dv, engine="converged" if ours.converged else ours.message,
                             engine_iterations=ours.iterations, engine_time_ms=ours.solve_time_s * 1e3,
                             pp_ac_only_control=control))
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)
    (HERE / "results.md").write_text(f"pandapower {pp.__version__}, generated by run_benchmark.py\n\n" + _markdown(df))
    return df


def export_reproducer(native, label):
    """Save a failing grid as plain pandapower JSON (engine columns removed)."""
    net = pp.from_json_string(pp.to_json(native))
    net.vsc = net.vsc.drop(columns=[c for c in ENGINE_COLUMNS if c in net.vsc])
    for tab in [t for t in net.keys() if t.startswith("res_")]:
        net[tab] = net[tab].iloc[0:0]
    net.name = label
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    pp.to_json(net, str(HERE / "reproducer" / f"{slug}.json"))


def _markdown(df):
    fmt = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
