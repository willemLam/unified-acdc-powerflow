"""YAML case files: MATPOWER-style tables (bus, slack, gen, line, trafo, bus_dc, line_dc, source_dc, converter)."""
import numpy as np
import pandapower.networks as pn
import pytest
import yaml

from unified_acdc_powerflow import build_model, build_ybus, run_pf
from unified_acdc_powerflow.cases import CASES, load_case
from unified_acdc_powerflow.grid_yaml import read_case, write_case


def _Y(net):
    m = build_model(net)
    build_ybus(m)
    return m


@pytest.mark.parametrize("case", ["case14", "case57", "case118", "case300"])
def test_pandapower_ieee_cases_survive_the_round_trip(case, tmp_path):
    net = getattr(pn, case)()
    write_case(net, tmp_path / "c.yaml")
    back = read_case(tmp_path / "c.yaml")
    a, b = _Y(net), _Y(back)
    assert np.max(np.abs(a.Y - b.Y)) < 1e-9
    ra, rb = run_pf(net), run_pf(back)
    assert rb.converged and np.max(np.abs(ra.E_ac - rb.E_ac)) < 1e-9


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("lossy", [False, True])
def test_every_case_file_loads_and_solves(name, lossy):
    from tests.helpers import power_balance_residual

    res = run_pf(load_case(name, lossy=lossy), tol=1e-10)
    assert res.converged, res.message
    assert power_balance_residual(res) < 1e-8


@pytest.mark.parametrize("name,ac_buses", [
    ("ieee57_14_hvdc", {212, 150, 151, 158, 202, 205, 115, 117}),
    ("ieee30_hvdc", {113, 115, 130, 102, 105, 106}),
    ("pegase_hvdc", {2072, 8195, 6246, 7282, 352}),
])
def test_hvdc_cases_have_no_dummy_busbar_nodes(name, ac_buses):
    """The converters sit on the grid buses themselves (the dummy busbar nodes of the paper's case data are gone)."""
    net = load_case(name)
    assert set(net.vsc.bus) == ac_buses


HAND_WRITTEN = """
base_mva: 100
f_hz: 50
bus:
  columns: [id, name, base_kv, pd_mw, qd_mvar, gs_mw, bs_mvar, in_service]
  rows:
  - [1, grid,   380, 0,  0, 0, 0,  true]
  - [2, hv,     220, 50, 10, 0, 20, true]
  - [3, conv,   220, 0,  0, 0, 0,  true]
slack:
  columns: [bus, vm_pu, va_deg, in_service]
  rows:
  - [1, 1.02, 0, true]
gen:
  columns: [bus, pg_mw, vg_pu, in_service]
  rows: []
line:
  columns: [from, to, r_pu, x_pu, b_pu, in_service]
  rows:
  - [2, 3, 0.01, 0.05, 0.02, true]
trafo:
  columns: [from, to, r_pu, x_pu, g_pu, b_pu, ratio, shift_deg, in_service]
  rows:
  - [1, 2, 0.002, 0.08, 0, 0, 1.05, 0, true]
bus_dc:
  columns: [id, name, base_kv, pd_mw, in_service]
  rows:
  - [10, dcA, 320, 0,  true]
  - [11, dcB, 320, 30, true]
line_dc:
  columns: [from, to, r_pu, in_service]
  rows:
  - [10, 11, 0.01, true]
source_dc:
  columns: [bus, vm_pu, in_service]
  rows: []
converter:
  columns: [name, ac_bus, dc_bus, mode, p_mw, q_mvar, vac_pu, va_deg, vdc_pu, p_dc_ref_mw, droop_k, vdc_ref_pu, loss_a, loss_b, loss_c, in_service]
  rows:
  - [IC1, 3, 10, VdcQ, 0, 5, 1, 0, 1.0, 0, 0, 1, 0, 0, 0, true]
"""


def test_hand_written_case_with_transformer_tap(tmp_path):
    p = tmp_path / "hand.yaml"
    p.write_text(HAND_WRITTEN)
    net = read_case(p)
    m = _Y(net)
    y = 1 / complex(0.002, 0.08)
    i1, i2 = m.bus_lookup[1], m.bus_lookup[2]
    assert m.Y[i1, i2] == pytest.approx(-y / 1.05, abs=1e-12)   # MATPOWER tap convention (from side)
    assert m.Y[i1, i1] == pytest.approx(y / 1.05**2, abs=1e-12)
    res = run_pf(net)
    assert res.converged, res.message
    assert res.converter_table().Q_ac_pu[0] == pytest.approx(0.05, abs=1e-9)


def test_written_file_is_one_row_per_element(tmp_path):
    net = pn.case14()
    write_case(net, tmp_path / "c.yaml")
    doc = yaml.safe_load((tmp_path / "c.yaml").read_text())
    assert len(doc["bus"]["rows"]) == 14
    assert len(doc["line"]["rows"]) + len(doc["trafo"]["rows"]) == len(net.line) + len(net.trafo)
    assert all(len(r) == len(doc["line"]["columns"]) for r in doc["line"]["rows"])
