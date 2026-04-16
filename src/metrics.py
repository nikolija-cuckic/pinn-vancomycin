"""
metrics.py
----------
Metrike za poredenje PINN-a i benchmark metoda.

Funkcije:
  relative_error  — relativna greska jednog parametra (%)
  param_errors    — relativna greska svih parametara odjednom
  curve_rmse      — RMSE rekonstrukcije krive na test tackama
  curve_mae       — MAE rekonstrukcije krive
"""

import numpy as np
from typing import Callable


def relative_error(estimated: float, true_val: float) -> float:
    """
    Relativna greska procene parametra (%).

    Returns
    -------
    float
        |estimated - true_val| / |true_val| * 100
    """
    if true_val == 0:
        raise ValueError("true_val ne sme biti nula.")
    return abs(estimated - true_val) / abs(true_val) * 100.0


def param_errors(estimated: dict, true_vals: dict) -> dict:
    """
    Relativna greska za sve parametre koji su prisutni u oba recnika.

    Returns
    -------
    dict
        {"err_<param>": float, ...}
    """
    errors = {}
    for key in true_vals:
        if key in estimated:
            errors[f"err_{key}"] = relative_error(estimated[key], true_vals[key])
    return errors


def curve_rmse(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    t_test:     np.ndarray,
    C_test:     np.ndarray,
) -> float:
    """
    RMSE rekonstrukcije krive.

    Parameters
    ----------
    predict_fn : callable
        t -> C_pred (np.ndarray -> np.ndarray)
    t_test, C_test : np.ndarray
        Test tacke i referentne koncentracije (mg/L).

    Returns
    -------
    float
        sqrt(mean((C_pred - C_test)^2))
    """
    C_pred = predict_fn(t_test)
    return float(np.sqrt(np.mean((C_pred - C_test) ** 2)))


def curve_mae(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    t_test:     np.ndarray,
    C_test:     np.ndarray,
) -> float:
    """MAE rekonstrukcije krive (mg/L)."""
    C_pred = predict_fn(t_test)
    return float(np.mean(np.abs(C_pred - C_test)))
