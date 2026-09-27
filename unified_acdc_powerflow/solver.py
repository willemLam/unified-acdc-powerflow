"""Newton-Raphson solver for the unified AC/DC power flow ([P] III-D / [T] 3.3.6).

    x(v+1) = x(v) - J(x(v))^-1 F(x(v))                          [P] (29)
    stop when max|F| < tol ([P] (30)) and max|dx| < tol.

Safeguard for the closed-form roots ([P] (15), [T] (3.42)): if a step makes a discriminant negative (no real root) or
does not reduce ||F||, the step is halved (up to MAX_HALVINGS times). Non-convergence is reported, not raised.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .jacobian import jacobian
from .mismatch import evaluate, pack, unpack
from .network import build_model, build_ybus
from .results import Result

MAX_HALVINGS = 10


def initial_state(model):
    """Flat start per island: AC nodes 1 p.u. at the angle of the island's reference node (PV nodes at
    their setpoint magnitude), DC nodes at the island's fixed voltage (else 1 p.u.)."""
    E = np.ones(model.n_ac, complex)
    n_isl, lab = connected_components(model.Y, directed=False)
    for isl in range(n_isl):
        refs = np.flatnonzero((lab == isl) & model.ac_ref & (model.v_set > 0))
        if len(refs):
            E[lab == isl] = np.exp(1j * model.va_set[refs[0]])
    E[model.ac_pv] *= model.v_set[model.ac_pv]
    Edc = np.ones(model.n_dc)
    if model.n_dc:
        n_isl, lab = connected_components(model.G, directed=False)
        for isl in range(n_isl):
            fixed = (lab == isl) & model.dc_fixed
            if fixed.any():
                Edc[lab == isl] = model.vdc_set[fixed].mean()
    return pack(model, E, Edc)


def dc_initial_state(model):
    """DC load-flow angles (incl. transformer phase shifts) for the AC nodes, magnitudes as in initial_state."""
    from pandapower.pypower.idx_brch import branch_cols
    from pandapower.pypower.makeBdc import makeBdc

    ppc = model._ppc
    branch = ppc["branch"]
    if branch.shape[1] < branch_cols:
        branch = np.hstack([branch, np.zeros((branch.shape[0], branch_cols - branch.shape[1]))])
    Bbus, _, Pbusinj, _ = makeBdc(ppc["bus"], branch)[:4]
    P = model.s_spec.real.copy()
    for m, l, p, pdc in zip(model.conv["mode"], model.conv["ac"], model.conv["p"], model.conv["pdc"]):
        if m in ("PQ", "PV"):
            P[l] += p
        elif m == "droop":
            P[l] -= pdc  # lossless guess P_conv = -P_dc,ref
    ref, u = model.ac_ref, ~model.ac_ref
    theta = model.va_set.copy()
    B = sp.csr_matrix(Bbus)
    rhs = P[u] - Pbusinj[u] - B[u][:, ref] @ theta[ref]
    theta[u] = spsolve(B[u][:, u].tocsc(), rhs)
    E_mag = np.where(model.ac_pv, model.v_set, 1.0)
    _, Edc = unpack(model, initial_state(model))
    if model.n_dc:
        # linearised DC load flow (P ~ G E around 1 p.u.) with the converter powers implied by the AC side
        P_ac = Bbus @ theta + Pbusinj  # DC load-flow AC injections
        p_dc = model.p_dc_spec.copy()
        for m, l, k, p, pdc in zip(model.conv["mode"], model.conv["ac"], model.conv["dc"], model.conv["p"],
                                   model.conv["pdc"]):
            if m in ("PQ", "PV"):
                p_dc[k] -= p
            elif m == "droop":
                p_dc[k] += pdc
            elif m == "GF":
                p_dc[k] -= P_ac[l] - model.s_spec.real[l]
        fx, u = model.dc_fixed, ~model.dc_fixed
        G = model.G.tocsr()
        try:
            Edc[u] = spsolve(G[u][:, u].tocsc(), p_dc[u] - G[u][:, fx] @ Edc[fx])
        except Exception:
            pass
        if not np.all(np.isfinite(Edc)) or np.any(Edc <= 0):
            _, Edc = unpack(model, initial_state(model))
    return pack(model, E_mag * np.exp(1j * theta), Edc)


def solve(model, tol=1e-8, max_iter=30, x0=None, init="auto"):
    """Newton-Raphson. init: "flat" (per-island flat start, as in [P]), "dc" (DC load-flow angles),
    or "auto" (flat, then dc if the flat start does not converge). Ignored when x0 is given.
    The wall time is reported in Result.solve_time_s."""
    t0 = time.perf_counter()
    res = _solve(model, tol, max_iter, x0, init)
    res.solve_time_s = time.perf_counter() - t0
    return res


def _solve(model, tol, max_iter, x0, init):
    if model.Y is None:
        build_ybus(model)
    if x0 is not None:
        return _newton(model, np.asarray(x0, float).copy(), tol, max_iter)
    if init not in ("auto", "flat", "dc"):
        raise ValueError("init must be 'auto', 'flat' or 'dc'")
    if init in ("auto", "flat"):
        res = _newton(model, initial_state(model), tol, max_iter)
        if res.converged or init == "flat":
            return res
    try:
        x_dc = dc_initial_state(model)
    except Exception as err:  # singular DC system: keep the flat-start result
        if init == "dc":
            raise
        res.message += f"; dc initialisation failed: {err}"
        return res
    res_dc = _newton(model, x_dc, tol, max_iter)
    if init == "auto":
        res_dc.message += " (dc initialisation; flat start failed: " + res.message + ")"
    return res_dc


def _newton(model, x, tol, max_iter):
    ev = evaluate(model, x)
    if not ev.ok:
        return _result(model, ev, None, False, 0, "closed-form discriminant negative at the initial state")
    dx_max, J, message, converged, it = np.inf, None, "", False, 0
    for it in range(max_iter + 1):
        f_max = np.max(np.abs(ev.F)) if len(ev.F) else 0.0
        if f_max < tol and dx_max < tol:
            converged, message = True, "converged"
            break
        if it == max_iter:
            message = f"no convergence in {max_iter} iterations (max mismatch {f_max:.3e})"
            break
        J = jacobian(model, ev)
        try:
            dx = spsolve(J, -ev.F)
        except Exception as err:  # singular Jacobian
            message = f"linear solve failed: {err}"
            break
        if not np.all(np.isfinite(dx)):
            message = "singular Jacobian"
            break
        step, first_ok, chosen = 1.0, None, None
        f2 = np.linalg.norm(ev.F)
        for _ in range(MAX_HALVINGS + 1):
            ev_new = evaluate(model, x + step * dx)
            if ev_new.ok:
                if first_ok is None:
                    first_ok = (step, ev_new)
                if np.linalg.norm(ev_new.F) < f2:
                    chosen = (step, ev_new)
                    break
            step /= 2
        chosen = chosen or first_ok
        if chosen is None:
            message = "closed-form discriminant stays negative after step halving"
            break
        step, ev = chosen
        x = x + step * dx
        dx_max = np.max(np.abs(step * dx))
    J = jacobian(model, ev) if converged else J
    return _result(model, ev, J, converged, it, message)


def _result(model, ev, J, converged, it, message):
    k = model.conv["dc"]
    s_conv = ev.Sc  # converter injection into the AC grid
    # converter injection into the DC grid: nodal DC injection minus DC loads; at a source_dc node the nodal
    # injection also contains the source, so the converter's own power balance is used there
    src = model.dc_source[k] if model.dc_source is not None else np.zeros(len(k), bool)
    conv_p_dc = np.where(src, -(s_conv.real + ev.loss), ev.Pdc[k] - model.p_dc_spec[k])
    return Result(converged=converged, iterations=it,
                  max_mismatch=float(np.max(np.abs(ev.F))) if len(ev.F) else 0.0,
                  E_ac=ev.E_ac, E_dc=ev.E_dc, S_ac=ev.S, P_dc=ev.Pdc,
                  conv_p_ac=s_conv.real, conv_q_ac=s_conv.imag, conv_p_dc=conv_p_dc,
                  conv_loss=ev.loss, conv_i=ev.Iabs, J=J, message=message, model=model)


def run_pf(net, **solve_kw):
    """build_model -> build_ybus -> solve."""
    model = build_model(net)
    build_ybus(model)
    return solve(model, **solve_kw)
