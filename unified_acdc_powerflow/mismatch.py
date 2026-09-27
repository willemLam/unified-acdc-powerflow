"""Residual vector F(x) of the unified AC/DC power flow.

References:
  [P] W. Lambrichts and M. Paolone, "General and Unified Model of the Power Flow Problem in Multiterminal AC/DC
      Networks", IEEE Transactions on Power Systems, 2024, doi:10.1109/TPWRS.2024.3378926.
  [T] W. Lambrichts, "Computational Methods for Hybrid AC/DC Networks", EPFL thesis 11001, 2025, chapter 3
      (adds the grid-forming converter).
Equation numbers below: [P] (n) / [T] (3.n); single-phase / positive-sequence form. Droop is not part of [P] or
[T]; it is the standard DC-voltage droop (see "droop" below).

Notation
--------
AC node voltage (rectangular):   E_i = E'_i + j E''_i
AC nodal current and power:      I = Y E,   S_i = E_i conj(I_i) = P_i + j Q_i          (nodal analysis, 3.2)
DC nodal current and power:      I_dc = G E_dc,   P_dc,k = E_k (G E_dc)_k
Injections are counted positive INTO the grid (generator convention).
s_spec,i   : injection of all ordinary elements at AC node i (gens, loads, shunt-free), p.u.
p_dc_spec,k: injection of the DC loads at DC node k (negative for a load), p.u.

Interfacing converter (IC) between AC node l and DC node k (notation of [P], (l, k)):
    S_conv = P_conv + j Q_conv : power the converter injects into the AC grid at l
    P_dc,conv                  : power the converter injects into the DC grid at k
    power balance over the IC [P] (7) / [T] (3.17):   P_conv + P_dc,conv + P_loss = 0
    losses, quadratic model [P] (23) / [T] (3.62): P_loss = a + b|I| + c|I|^2,  |I| = |S_conv| / |E_l|

State and residuals
-------------------
Fixed a priori (not in x): slack AC voltages ([P] Table I), grid-forming AC voltages ([T] (3.43)-(3.44)), DC
voltage-source and VdcQ DC voltages ([P] (6)). Everything else is unknown:

    x = [ E' (unknown AC nodes) | E'' (unknown AC nodes) | E (unknown DC nodes) ]
    F = [ row 1 per unknown AC node | row 2 per unknown AC node | one row per unknown DC node ]

All residuals are "calculated - specified":

    node / mode           row 1 (AC node l)                  row 2 (AC node l)         DC row (node k)
    AC PQ bus  (1)-(2)    P_l - Re s_spec                    Q_l - Im s_spec           -
    AC PV bus  (3)-(4)    P_l - Re s_spec                    |E_l|^2 - V_set^2         -
    DC P bus   (5)        -                                  -                         P_dc,k - p_dc_spec,k
    IC PQ  (17)-(19)      P_l - Re s_spec - P_set            Q_l - Im s_spec - Q_set   P_dc,k - p_dc_spec,k + P_set + P_loss
    IC PV  (8), (4)       P_l - Re s_spec - P_set            |E_l|^2 - V_set^2         same as IC PQ
    IC VdcQ (7), (12)-(15)E_k*(x) - E_k,set  (closed form)   Q_l - Im s_spec - Q_set   - (E_k fixed)
    IC VdcV (14)-(15), (4)E_k*(x) - E_k,set  (closed form)   |E_l|^2 - V_set^2         - (E_k fixed)
    IC grid-forming [T]   - (E_l fixed)                      -                         E'_l*(x) - E'_l,set (closed form)
      (3.37)-(3.44)
    IC droop              P_l - Re s_spec + P_dc,conv + P_loss   Q_l - Im s_spec - Q_set   P_dc,k - p_dc_spec,k - P_dc,conv
      with P_dc,conv = P_dc,ref - k_droop (E_k - V_dc,ref)

If the IC's AC bus also carries a PV generator, row 2 stays the generator's voltage equation and the converter
injects its Q_set (the generator supplies the rest). If the IC's DC bus is held by a DC voltage source, the DC
row does not exist (the source absorbs the converter power).
"""
from dataclasses import dataclass

import numpy as np

PQ_LIKE = ("PQ", "PV", "droop")  # converters whose DC node is an unknown with a power-balance row


@dataclass
class Layout:
    u_ac: np.ndarray    # unknown AC nodes
    u_dc: np.ndarray    # unknown DC nodes
    pos_ac: np.ndarray  # node -> position in u_ac (-1 if fixed)
    pos_dc: np.ndarray  # node -> position in u_dc (-1 if fixed)
    n_x: int


@dataclass
class Eval:
    F: np.ndarray        # residual vector
    E_ac: np.ndarray     # AC voltages (unknown + fixed)
    E_dc: np.ndarray     # DC voltages (unknown + fixed)
    S: np.ndarray        # AC nodal injections S = E conj(Y E)
    I: np.ndarray        # AC nodal currents Y E
    Pdc: np.ndarray      # DC nodal injections E (G E)
    Idc: np.ndarray      # DC nodal currents G E
    Sc: np.ndarray       # converter AC injection S_conv (per converter)
    Iabs: np.ndarray     # converter current |I| = |S_conv| / |E_l|
    loss: np.ndarray     # converter losses a + b|I| + c|I|^2
    p_dc_conv: np.ndarray  # droop: P_dc,conv (NaN for other modes)
    root: np.ndarray     # closed-form root (VdcQ: E_k*, GF: E'_l*), NaN otherwise
    lin: np.ndarray      # linear coefficient of the quadratic (VdcQ: s, GF: a)
    alpha: dict          # converter -> discriminant of the quadratic
    ok: bool             # all discriminants >= 0
    sigma: np.ndarray    # GF: root branch (+1 / -1)
    f: np.ndarray        # GF: constant term of the quadratic


def layout(model):
    """Index sets of the unknowns (cached on the model; rebuilt when ac_ref / dc_fixed change)."""
    key = (model.ac_ref.tobytes(), model.dc_fixed.tobytes())
    lay = getattr(model, "_layout", None)
    if lay is None or getattr(model, "_layout_key", None) != key:
        u_ac = np.flatnonzero(~model.ac_ref)
        u_dc = np.flatnonzero(~model.dc_fixed)
        pos_ac = np.full(model.n_ac, -1)
        pos_ac[u_ac] = np.arange(len(u_ac))
        pos_dc = np.full(model.n_dc, -1)
        pos_dc[u_dc] = np.arange(len(u_dc))
        lay = Layout(u_ac, u_dc, pos_ac, pos_dc, 2 * len(u_ac) + len(u_dc))
        model._layout, model._layout_key = lay, key
    return lay


def pack(model, E_ac, E_dc):
    """Voltages -> state vector x."""
    lay = layout(model)
    E_ac = np.asarray(E_ac, complex)
    return np.r_[E_ac.real[lay.u_ac], E_ac.imag[lay.u_ac], np.asarray(E_dc, float)[lay.u_dc]]


def unpack(model, x):
    """State vector x -> voltages, with the fixed voltages at their setpoints."""
    lay = layout(model)
    n = len(lay.u_ac)
    E_ac = (model.v_set * np.exp(1j * model.va_set)).astype(complex)
    E_ac[lay.u_ac] = x[:n] + 1j * x[n:2 * n]
    E_dc = model.vdc_set.astype(float).copy()
    E_dc[lay.u_dc] = x[2 * n:]
    return E_ac, E_dc


def converter_current(model, S, E_ac):
    """Converter AC injection S_conv and current |I| = |S_conv| / |E_l| (used by the loss model 3.62).

    S_conv is the nodal injection minus the other elements at l. On a PV-generator bus the reactive part is
    the converter's Q setpoint (the generator's Q is not known separately).
    """
    c = model.conv
    l = c["ac"]
    Sc = S[l] - model.s_spec[l]
    on_pv = model.ac_pv[l]
    Sc = np.where(on_pv, Sc.real + 1j * c["q"], Sc)
    Iabs = np.abs(Sc) / np.abs(E_ac[l]) if len(l) else np.zeros(0)
    return Sc, Iabs


def evaluate(model, x):
    """Residuals F(x) and the intermediate quantities the Jacobian needs."""
    lay = layout(model)
    c = model.conv
    E_ac, E_dc = unpack(model, x)
    I = model.Y @ E_ac
    S = E_ac * np.conj(I)
    Idc = model.G @ E_dc
    Pdc = E_dc * Idc

    # ordinary nodes: PQ [P] (1)-(2), PV [P] (3)-(4), DC P [P] (5)
    u = lay.u_ac
    r1 = S.real[u] - model.s_spec.real[u]
    r2 = S.imag[u] - model.s_spec.imag[u]
    pv = model.ac_pv[u]
    r2[pv] = np.abs(E_ac[u][pv]) ** 2 - model.v_set[u][pv] ** 2
    rdc = Pdc[lay.u_dc] - model.p_dc_spec[lay.u_dc]

    nc = model.n_conv
    Sc, Iabs = converter_current(model, S, E_ac)
    loss = c["la"] + c["lb"] * Iabs + c["lc"] * Iabs**2 if nc else np.zeros(0)  # [P] (23)
    p_dc_conv, root, lin, sigma, fc = (np.full(nc, np.nan) for _ in range(5))
    alpha, ok = {}, True
    Gd = model.G.diagonal() if model.n_dc else np.zeros(0)
    Yd = model.Y.diagonal()

    for i in range(nc):
        m, l, k = c["mode"][i], c["ac"][i], c["dc"][i]
        pl, pk = lay.pos_ac[l], lay.pos_dc[k]  # row positions (-1: node fixed, no row)
        l_is_pv = model.ac_pv[l]               # PV generator on the converter's AC bus

        if m in ("PQ", "PV"):
            # AC side [P] (17)-(18) / (8): converter injects P_set; DC side [P] (19): P_dc,conv = -(P_set + P_loss)
            r1[pl] -= c["p"][i]
            if m == "PV":
                r2[pl] = abs(E_ac[l]) ** 2 - c["vac"][i] ** 2       # [P] (4)
            elif not l_is_pv:
                r2[pl] -= c["q"][i]                                  # [P] (18)
            if pk >= 0:
                rdc[pk] += c["p"][i] + loss[i]

        elif m == "droop":
            # Standard DC-voltage droop (not in [P] or [T]):
            #   P_dc,conv = P_dc,ref - k (E_k - V_dc,ref),  k >= 0, P_dc,conv injected into the DC grid.
            # A rising DC voltage lowers the converter's DC injection (rectifier) / raises its export to AC.
            p_dc_conv[i] = c["pdc"][i] - c["k"][i] * (E_dc[k] - c["vref"][i])
            r1[pl] += p_dc_conv[i] + loss[i]      # IC power balance: P_conv = -(P_dc,conv + P_loss)
            if not l_is_pv:
                r2[pl] -= c["q"][i]
            if pk >= 0:
                rdc[pk] -= p_dc_conv[i]

        elif m in ("VdcQ", "VdcV"):
            # DC power balance at the fixed node k, written as a quadratic in E_k ([P] (14) / [T] (3.26)):
            #   G_kk E_k^2 + s E_k + c = 0,   s = sum_{m != k} G_km E_m,
            #   c = P_conv + P_loss - p_dc_spec,k   (P_conv = P_l - Re s_spec,l, the converter's AC injection)
            # positive root ([P] (15) / [T] (3.27), Theorem 1 of [T]):  E_k* = (-s + sqrt(s^2 - 4 G_kk c)) / (2 G_kk)
            # residual: E_k*(x) - E_k,set, i.e. the converter delivers exactly the DC power that holds E_k,set.
            Gkk = Gd[k]
            s = Idc[k] - Gkk * E_dc[k]
            cc = (S.real[l] - model.s_spec.real[l]) + loss[i] - model.p_dc_spec[k]
            a_ = s * s - 4 * Gkk * cc
            alpha[i], ok = a_, ok and a_ >= 0
            root[i] = (-s + np.sqrt(max(a_, 0.0))) / (2 * Gkk)
            lin[i] = s
            r1[pl] = root[i] - c["vdc"][i]
            if m == "VdcV":                                          # [P] (4): the converter holds |E_l|
                r2[pl] = abs(E_ac[l]) ** 2 - c["vac"][i] ** 2
            elif not l_is_pv:
                r2[pl] -= c["q"][i]

        elif m == "GF":
            # AC power balance at the fixed node l as a quadratic in E'_l ([T] (3.39)-(3.41)), E''_l fixed:
            #   G_ll E'_l^2 + a E'_l + f = 0
            #   a = sum_{n != l} (G_ln E'_n - B_ln E''_n),   b = sum_{n != l} (B_ln E'_n + G_ln E''_n)   [T] (3.40)
            #   f = G_ll E''_l^2 + E''_l b - Re s_spec,l + P_dc,conv + P_loss,   P_dc,conv = P_dc,k - p_dc_spec,k
            # root [T] (3.42): E'_l* = (-a + sqrt(a^2 - 4 G_ll f)) / (2 G_ll); residual E'_l*(x) - E'_l,set.
            if pk < 0:  # DC node held by a source: E_l and E_k both fixed, nothing left to solve
                continue
            Gll, Bll = Yd[l].real, Yd[l].imag
            er, ei = E_ac[l].real, E_ac[l].imag
            a = I[l].real - (Gll * er - Bll * ei)
            b = I[l].imag - (Bll * er + Gll * ei)
            f = Gll * ei * ei + ei * b - model.s_spec.real[l] + (Pdc[k] - model.p_dc_spec[k]) + loss[i]
            a_ = a * a - 4 * Gll * f
            alpha[i], ok = a_, ok and a_ >= 0
            # The "+" root of [T] is the branch that contains the setpoint while 2 G_ll E'_l + a > 0, which
            # holds for the small angles met in practice; sigma picks the other branch otherwise. Of the two
            # algebraically equal forms (-a + sigma sqrt(alpha)) / (2 G_ll) and 2f / (-a - sigma sqrt(alpha))
            # the one with the larger denominator is used (the first breaks down for G_ll -> 0, e.g. behind a
            # lossless reactor; the second for f -> 0).
            sigma[i] = 1.0 if 2 * Gll * er + a >= 0 else -1.0
            sq = np.sqrt(max(a_, 0.0))
            D = -a - sigma[i] * sq
            root[i] = 2 * f / D if abs(D) > abs(2 * Gll) else (-a + sigma[i] * sq) / (2 * Gll)
            lin[i], fc[i] = a, f
            rdc[pk] = root[i] - er

    F = np.r_[r1, r2, rdc]
    return Eval(F, E_ac, E_dc, S, I, Pdc, Idc, Sc, Iabs, loss, p_dc_conv, root, lin, alpha, bool(ok), sigma, fc)
