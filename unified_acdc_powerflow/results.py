"""Load-flow result container and output table."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Result:
    converged: bool
    iterations: int
    max_mismatch: float
    E_ac: np.ndarray
    E_dc: np.ndarray
    S_ac: np.ndarray       # AC nodal injections (p.u.)
    P_dc: np.ndarray       # DC nodal injections (p.u.)
    conv_p_ac: np.ndarray  # converter injection into the AC grid (p.u.)
    conv_q_ac: np.ndarray
    conv_p_dc: np.ndarray  # converter injection into the DC grid (p.u.)
    conv_loss: np.ndarray
    conv_i: np.ndarray     # converter current magnitude used for the loss (p.u.)
    J: object
    message: str
    model: object = None
    solve_time_s: float = float("nan")  # wall time of solve() (Newton iterations incl. any re-initialisation)

    def table(self):
        """One row per node: AC nodes then DC nodes. P/Q are nodal injections in p.u."""
        m = self.model
        ac_type = np.where(m.ac_ref, "slack", np.where(m.ac_pv, "PV", "PQ")).astype(object)
        dc_type = np.where(m.dc_fixed, "Vdc", "P").astype(object)
        for mode, l, k in zip(m.conv["mode"], m.conv["ac"], m.conv["dc"]):
            ac_type[l] = f"conv {mode}"
            dc_type[k] = f"conv {mode}"
        ac = pd.DataFrame({"side": "ac", "bus": m.ac_bus_id, "name": m.ac_names, "type": ac_type,
                           "E_re": self.E_ac.real, "E_im": self.E_ac.imag,
                           "P_pu": self.S_ac.real, "Q_pu": self.S_ac.imag})
        dc = pd.DataFrame({"side": "dc", "bus": m.dc_bus_id, "name": m.dc_names, "type": dc_type,
                           "E_re": self.E_dc, "E_im": 0.0, "P_pu": self.P_dc, "Q_pu": 0.0})
        return pd.concat([ac, dc], ignore_index=True)

    def converter_table(self):
        m = self.model
        return pd.DataFrame({"vsc": m.conv["index"], "name": m.conv["name"], "mode": m.conv["mode"],
                             "P_ac_pu": self.conv_p_ac, "Q_ac_pu": self.conv_q_ac, "P_dc_pu": self.conv_p_dc,
                             "loss_pu": self.conv_loss, "I_pu": self.conv_i})
