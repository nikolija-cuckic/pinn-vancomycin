"""
inverse_problem.py
------------------
Wrapper za inverzni problem: uzorkovanje merenja, dodavanje suma,
procena parametara (PINN + benchmark).

Funkcije:
  subsample_1comp   — uzorkuje N tacaka iz 1-odeljnog profila
  subsample_2comp   — uzorkuje N tacaka iz 2-odeljnog profila (samo C1)
  run_inverse_1comp — PINN + curve_fit za 1-odeljni model
  run_inverse_2comp — PINN + NLS za 2-odeljni model

Tipicna upotreba (sensitivity analysis petlja):
    t_sp, C_sp = subsample_1comp(profiles_df, "S001", N=5, sigma=0.10, seed=42)
    result = run_inverse_1comp(t_sp, C_sp, gt_row)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinn_model import OneCompartmentPINN, TwoCompartmentPINN, train_pinn
from benchmark import fit_one_compartment, fit_two_compartment
from metrics import param_errors
from data_processing import DOSE_MG


# ── Uzorkovanje ───────────────────────────────────────────────────────────────

def subsample_1comp(
    profiles_df: pd.DataFrame,
    subject_id:  str,
    N:           int,
    sigma:       float,
    seed:        int,
) -> tuple:
    """
    Uzorkuje N merenja iz 1-odeljnog profila i dodaje multiplikativni Gaussov sum.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Normalizovani profili (subject_profiles_1comp.csv) ili raw (vancomycin_1comp_raw.csv).
        Mora imati kolone "subject_id", "time_h", "C_true".
    N : int
        Broj merenja koje uzorkujemo.
    sigma : float
        Relativna SD suma (0 = bez suma).
    seed : int
        Seed za reproduktivnost.

    Returns
    -------
    (t_sparse_h, C_noisy) : np.ndarray, np.ndarray
    """
    rng  = np.random.default_rng(seed)
    subj = profiles_df[profiles_df["subject_id"] == subject_id]
    if len(subj) == 0:
        raise ValueError(f"Subjekt {subject_id} nije pronadjen.")
    if N > len(subj):
        raise ValueError(f"N={N} vece od dostupnih merenja ({len(subj)}).")

    idx      = np.sort(rng.choice(len(subj), N, replace=False))
    t_sparse = subj["time_h"].values[idx]
    C_true   = subj["C_true"].values[idx]

    noise   = rng.normal(0, sigma, N) if sigma > 0 else np.zeros(N)
    C_noisy = np.maximum(C_true * (1 + noise), 1e-6)
    return t_sparse, C_noisy


def subsample_2comp(
    profiles_df: pd.DataFrame,
    subject_id:  str,
    N:           int,
    sigma:       float,
    seed:        int,
) -> tuple:
    """
    Uzorkuje N C1 merenja iz 2-odeljnog profila.

    Returns
    -------
    (t_sparse_h, C1_noisy) : np.ndarray, np.ndarray
    """
    rng  = np.random.default_rng(seed)
    subj = profiles_df[profiles_df["subject_id"] == subject_id]
    if len(subj) == 0:
        raise ValueError(f"Subjekt {subject_id} nije pronadjen.")
    if N > len(subj):
        raise ValueError(f"N={N} vece od dostupnih merenja ({len(subj)}).")

    idx      = np.sort(rng.choice(len(subj), N, replace=False))
    t_sparse = subj["time_h"].values[idx]
    C1_true  = subj["C1_true"].values[idx]

    noise    = rng.normal(0, sigma, N) if sigma > 0 else np.zeros(N)
    C1_noisy = np.maximum(C1_true * (1 + noise), 1e-6)
    return t_sparse, C1_noisy


# ── Inverzni problem — 1-odeljni ──────────────────────────────────────────────

def run_inverse_1comp(
    t_sparse:     np.ndarray,
    C_noisy:      np.ndarray,
    gt_params:    dict,
    dose_mg:      float = DOSE_MG,
    t_max_h:      float = 24.0,
    epochs_adam:  int   = 5000,
    epochs_lbfgs: int   = 500,
    device:       str   = "cpu",
    verbose:      bool  = False,
) -> dict:
    """
    Procena parametara 1-odeljnog modela iz sparnih merenja.

    Parameters
    ----------
    t_sparse, C_noisy : np.ndarray
        Sparna merenja sa sumom (originalne jedinice: h, mg/L).
    gt_params : dict
        Ground truth — mora imati "k10" i "Vd".

    Returns
    -------
    dict s kljucevima:
        pinn_k10, pinn_Vd, pinn_CL, pinn_err_k10, pinn_err_Vd
        bench_k10, bench_Vd, bench_CL, bench_err_k10, bench_err_Vd, bench_success
    """
    C_max  = C_noisy.max()
    t_norm = t_sparse / t_max_h
    C_norm = C_noisy  / C_max

    # ── PINN ──────────────────────────────────────────────────────────────────
    model = OneCompartmentPINN(dose_mg=dose_mg, t_max_h=t_max_h, device=device)
    train_pinn(model, t_norm, C_norm, t_max_h, C_max,
               epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs, verbose=verbose)
    pinn_p = model.get_parameters()
    pinn_e = param_errors(pinn_p, {k: gt_params[k] for k in ["k10", "Vd"]})

    # ── Benchmark ─────────────────────────────────────────────────────────────
    bench   = fit_one_compartment(t_sparse, C_noisy, dose_mg=dose_mg)
    bench_e = (param_errors({"k10": bench["k10"], "Vd": bench["Vd"]},
                             {k: gt_params[k] for k in ["k10", "Vd"]})
               if bench["success"]
               else {"err_k10": np.nan, "err_Vd": np.nan})

    return {
        "pinn_k10":      pinn_p["k10"],
        "pinn_Vd":       pinn_p["Vd"],
        "pinn_CL":       pinn_p["CL"],
        "pinn_err_k10":  pinn_e.get("err_k10", np.nan),
        "pinn_err_Vd":   pinn_e.get("err_Vd",  np.nan),
        "bench_k10":     bench.get("k10", np.nan),
        "bench_Vd":      bench.get("Vd",  np.nan),
        "bench_CL":      bench.get("CL",  np.nan),
        "bench_err_k10": bench_e.get("err_k10", np.nan),
        "bench_err_Vd":  bench_e.get("err_Vd",  np.nan),
        "bench_success": bench["success"],
    }


# ── Inverzni problem — 2-odeljni ──────────────────────────────────────────────

def run_inverse_2comp(
    t_sparse:     np.ndarray,
    C1_noisy:     np.ndarray,
    gt_params:    dict,
    dose_mg:      float = DOSE_MG,
    t_max_h:      float = 24.0,
    epochs_adam:  int   = 8000,
    epochs_lbfgs: int   = 1000,
    device:       str   = "cpu",
    verbose:      bool  = False,
) -> dict:
    """
    Procena parametara 2-odeljnog modela iz sparnih C1 merenja.

    Parameters
    ----------
    gt_params : dict
        Ground truth — mora imati "k10", "k12", "k21", "V1".

    Returns
    -------
    dict s PINN i benchmark procenama + greskama za sve 4 parametra.
    """
    C1_max  = C1_noisy.max()
    t_norm  = t_sparse / t_max_h
    C1_norm = C1_noisy / C1_max

    # ── PINN ──────────────────────────────────────────────────────────────────
    model = TwoCompartmentPINN(dose_mg=dose_mg, t_max_h=t_max_h, device=device)
    train_pinn(model, t_norm, C1_norm, t_max_h, C1_max,
               epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs, verbose=verbose)
    pinn_p = model.get_parameters()
    pinn_e = param_errors(pinn_p, {k: gt_params[k] for k in ["k10", "k12", "k21", "V1"]})

    # ── Benchmark ─────────────────────────────────────────────────────────────
    bench   = fit_two_compartment(t_sparse, C1_noisy, dose_mg=dose_mg)
    bench_e = (param_errors({k: bench[k] for k in ["k10", "k12", "k21", "V1"]},
                             {k: gt_params[k] for k in ["k10", "k12", "k21", "V1"]})
               if bench["success"]
               else {f"err_{k}": np.nan for k in ["k10", "k12", "k21", "V1"]})

    return {
        "pinn_k10":      pinn_p["k10"],
        "pinn_k12":      pinn_p["k12"],
        "pinn_k21":      pinn_p["k21"],
        "pinn_V1":       pinn_p["V1"],
        "pinn_CL":       pinn_p["CL"],
        "pinn_err_k10":  pinn_e.get("err_k10", np.nan),
        "pinn_err_k12":  pinn_e.get("err_k12", np.nan),
        "pinn_err_k21":  pinn_e.get("err_k21", np.nan),
        "pinn_err_V1":   pinn_e.get("err_V1",  np.nan),
        "bench_k10":     bench.get("k10", np.nan),
        "bench_k12":     bench.get("k12", np.nan),
        "bench_k21":     bench.get("k21", np.nan),
        "bench_V1":      bench.get("V1",  np.nan),
        "bench_CL":      bench.get("CL",  np.nan),
        "bench_err_k10": bench_e.get("err_k10", np.nan),
        "bench_err_k12": bench_e.get("err_k12", np.nan),
        "bench_err_k21": bench_e.get("err_k21", np.nan),
        "bench_err_V1":  bench_e.get("err_V1",  np.nan),
        "bench_success": bench["success"],
    }
