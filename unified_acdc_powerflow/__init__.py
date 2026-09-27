"""Unified AC/DC Newton-Raphson power flow.

References:
  [P] W. Lambrichts and M. Paolone, "General and Unified Model of the Power Flow Problem in Multiterminal AC/DC
      Networks", IEEE Transactions on Power Systems, 2024, doi:10.1109/TPWRS.2024.3378926.
  [T] W. Lambrichts, "Computational Methods for Hybrid AC/DC Networks", EPFL thesis 11001, 2025, chapter 3
      (adds the grid-forming converter).
"""
from .network import Model, add_converter, build_model, build_ybus, expand_converter_impedances
from .results import Result
from .solver import run_pf, solve

__all__ = ["Model", "Result", "add_converter", "build_model", "build_ybus", "expand_converter_impedances",
           "run_pf", "solve"]
