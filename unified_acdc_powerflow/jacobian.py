"""Analytical sparse Jacobian J = dF/dx of the residuals in mismatch.py (structure of [P] (28) / [T] (3.66)).

The Jacobian is assembled in "full coordinates" and then restricted to the unknowns:
    columns: [E' of every AC node | E'' of every AC node | E of every DC node]
    rows:    [row 1 of every AC node | row 2 of every AC node | row of every DC node]
Row/column selection afterwards drops fixed nodes (slack, grid-forming, VdcQ and source DC nodes).

Building blocks (rectangular coordinates, S = E conj(Y E)):
    dS/dE'  = diag(conj I) + diag(E) conj(Y)
    dS/dE'' = j diag(conj I) - j diag(E) conj(Y)
    d|E_l|^2 = 2 E'_l dE'_l + 2 E''_l dE''_l
    dθ_n = (E'_n dE''_n - E''_n dE'_n) / |E_n|^2          (ACE: dP_ace = -k (dθ_l - dθ_r))
    dP_dc/dE_dc = diag(G E_dc) + diag(E_dc) G
    d Re(I) = [G, -B] . [dE', dE''],   d Im(I) = [B, G] . [dE', dE'']
Loss derivative ([P] (23)): dP_loss = (b + 2c|I|) d|I|, with
    |I|^2 = (P_conv^2 + Q_conv^2) / |E_l|^2  ->  d|I|^2 = (2 P_conv dP + 2 Q_conv dQ) / |E_l|^2 - |I|^2 d|E_l|^2 / |E_l|^2
Closed-form roots (VdcQ [P] (15), GF [T] (3.42)), root = (-p + sqrt(alpha)) / (2g) with alpha = p^2 - 4 g q:
    d root = (-dp + (p dp - 2 g dq) / sqrt(alpha)) / (2g)
"""
import numpy as np
import scipy.sparse as sp

from .mismatch import layout


def jacobian(model, ev):
    lay = layout(model)
    c = model.conv
    na, nd = model.n_ac, model.n_dc
    nf = 2 * na + nd
    E, I, Edc, Idc = ev.E_ac, ev.I, ev.E_dc, ev.Idc
    Y, G = model.Y.tocsr(), model.G.tocsr()

    # --- building blocks, each as a full-coordinate row block ---------------------------------------------
    dE, dIc = sp.diags(E), sp.diags(np.conj(I))
    dS_re = (dIc + dE @ np.conj(Y)).tocsr()             # dS/dE'
    dS_im = (1j * dIc - 1j * dE @ np.conj(Y)).tocsr()    # dS/dE''
    Zad, Zda = sp.csr_matrix((na, nd)), sp.csr_matrix((nd, 2 * na))
    A_P = sp.hstack([dS_re.real, dS_im.real, Zad]).tocsr()        # dP
    A_Q = sp.hstack([dS_re.imag, dS_im.imag, Zad]).tocsr()        # dQ
    A_dc = sp.hstack([Zda, sp.diags(Idc) + sp.diags(Edc) @ G]).tocsr()  # dP_dc
    A_Idc = sp.hstack([Zda, G]).tocsr()                           # d(G E_dc)
    A_Ire = sp.hstack([Y.real, -Y.imag, Zad]).tocsr()             # d Re(I)
    A_Iim = sp.hstack([Y.imag, Y.real, Zad]).tocsr()              # d Im(I)

    def unit(j, v=1.0):
        return sp.csr_matrix(([v], ([0], [j])), shape=(1, nf))

    def gV2(l):  # d|E_l|^2
        return sp.csr_matrix(([2 * E[l].real, 2 * E[l].imag], ([0, 0], [l, na + l])), shape=(1, nf))

    def gTheta(n):  # dθ_n = (E'_n dE''_n - E''_n dE'_n) / |E_n|^2
        v2 = abs(E[n]) ** 2
        return sp.csr_matrix(([-E[n].imag / v2, E[n].real / v2], ([0, 0], [n, na + n])), shape=(1, nf))

    # --- ordinary nodes: rows 1 = dP, rows 2 = dQ (PQ) or d|E|^2 (PV), DC rows = dP_dc ---------------------
    A_Q2 = A_Q.tolil()
    for l in np.flatnonzero(model.ac_pv):
        A_Q2[l, :] = gV2(l)
    Jf = sp.vstack([A_P, A_Q2.tocsr(), A_dc]).tolil()

    # --- converter rows: add to / replace the ordinary rows, mirroring mismatch.evaluate --------------------
    for i in range(model.n_conv):
        m, l, k = c["mode"][i], c["ac"][i], c["dc"][i]
        r1, r2, rk = l, na + l, 2 * na + k
        gloss = _loss_gradient(model, ev, i, A_P[l], A_Q[l], gV2(l))
        if m in ("PQ", "PV"):
            # row 1: P_l - Re s_spec - P_set   -> no extra term
            if m == "PV":
                Jf[r2, :] = gV2(l)                               # |E_l|^2 - V_set^2
            Jf[rk, :] = Jf[rk, :] + gloss                        # DC row: ... + P_set + P_loss
        elif m == "droop":
            dp_dc = -c["k"][i] * unit(2 * na + k)                # dP_dc,conv = -k dE_k
            Jf[r1, :] = Jf[r1, :] + dp_dc + gloss                # row 1: ... + P_dc,conv + P_loss
            Jf[rk, :] = Jf[rk, :] - dp_dc                        # DC row: ... - P_dc,conv
        elif m == "ACE":
            dp = -c["kace"][i] * (gTheta(l) - gTheta(c["r"][i]))   # dP_ace
            Jf[r1, :] = Jf[r1, :] - dp                              # row 1: ... - P_ace
            Jf[rk, :] = Jf[rk, :] + dp + gloss                      # DC row: ... + P_ace + P_loss
        elif m in ("VdcQ", "VdcV"):
            # root E_k* = (-s + sqrt(alpha)) / (2 G_kk), s = sum_{m!=k} G_km E_m, c = P_conv + P_loss - p_dc_spec
            Gkk = G[k, k]
            s, al = ev.lin[i], ev.alpha[i]
            ds = A_Idc[k] - Gkk * unit(2 * na + k)
            dc = A_P[l] + gloss
            Jf[r1, :] = (-ds + (s * ds - 2 * Gkk * dc) / np.sqrt(al)) / (2 * Gkk)
            if m == "VdcV":
                Jf[r2, :] = gV2(l)                               # |E_l|^2 - V_set^2
        elif m == "GF" and lay.pos_dc[k] >= 0:
            # root E'_l* of G_ll E'^2 + a E' + f = 0 ([T] (3.41)-(3.42)); same branch/form choice as in evaluate()
            Gll, Bll = Y[l, l].real, Y[l, l].imag
            a, al, sg, f = ev.lin[i], ev.alpha[i], ev.sigma[i], ev.f[i]
            ei = E[l].imag
            b = I[l].imag - (Bll * E[l].real + Gll * ei)
            da = A_Ire[l] - Gll * unit(l) + Bll * unit(na + l)   # a = Re(I_l) - (G_ll E'_l - B_ll E''_l)
            db = A_Iim[l] - Bll * unit(l) - Gll * unit(na + l)   # b = Im(I_l) - (B_ll E'_l + G_ll E''_l)
            df = (2 * Gll * ei + b) * unit(na + l) + ei * db + A_dc[k] + gloss
            dsq = (a * da - 2 * Gll * df) / np.sqrt(al)          # d sqrt(alpha)
            D = -a - sg * np.sqrt(al)
            if abs(D) > abs(2 * Gll):                            # root = 2f / D
                droot = 2 * df / D - 2 * f * (-da - sg * dsq) / D**2
            else:                                                # root = (-a + sigma sqrt(alpha)) / (2 G_ll)
                droot = (-da + sg * dsq) / (2 * Gll)
            Jf[rk, :] = droot - unit(l)                          # E'_l* - E'_l,set

    sel = np.r_[lay.u_ac, na + lay.u_ac, 2 * na + lay.u_dc]
    return Jf.tocsr()[sel][:, sel].tocsc()


def _loss_gradient(model, ev, i, gP, gQ, gV2):
    """d P_loss for converter i (loss model 3.62, |I| = |S_conv| / |E_l|)."""
    c = model.conv
    nf = gP.shape[1]
    kb = c["lb"][i] + 2 * c["lc"][i] * ev.Iabs[i]   # dP_loss/d|I|
    Iabs = ev.Iabs[i]
    if kb == 0.0 or Iabs < 1e-12:                   # lossless, or |I| = 0 where |I| is not differentiable
        return sp.csr_matrix((1, nf))
    l = c["ac"][i]
    Sc, V2 = ev.Sc[i], abs(ev.E_ac[l]) ** 2
    dQ = 0 * gQ if model.ac_pv[l] else gQ           # on a PV-generator bus Q_conv = Q_set (constant)
    dI2 = (2 * Sc.real * gP + 2 * Sc.imag * dQ) / V2 - Iabs**2 * gV2 / V2
    return kb * dI2 / (2 * Iabs)
