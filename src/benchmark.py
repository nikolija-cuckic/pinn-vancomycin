"""
benchmark.py
------------
Klasicni NLS benchmark za jednoodeljni i dvoodeljni PK model.

Funkcije:
  fit_one_compartment  — curve_fit na C(t) = C0*exp(-k10*t)
  fit_two_compartment  — minimize(NLS) + solve_ivp za 2-odeljni model
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing import one_compartment, two_compartment_ode, DOSE_MG


# ── 1-odeljni benchmark ───────────────────────────────────────────────────────

def fit_one_compartment(
    t_data:     np.ndarray,
    C_data:     np.ndarray,
    dose_mg:    float = DOSE_MG,
    n_restarts: int   = 5,
) -> dict:
    """
    Procena k10 i Vd curve_fit-om (multi-start).

    Parametrizacija: C(t) = (D/Vd) * exp(-k10 * t)

    Returns
    -------
    dict
        {"k10", "Vd", "CL", "C0", "success", "residual"}
    """
    best_result = None
    best_res    = np.inf

    C0_inits  = [C_data[0], C_data.max(), dose_mg / 30.0, dose_mg / 60.0, dose_mg / 20.0]
    k10_inits = [0.05,      0.08,         0.12,            0.15,           0.10]

    for C0_init, k10_init in zip(C0_inits, k10_inits):
        try:
            (C0_fit, k10_fit), _ = curve_fit(
                one_compartment,
                t_data, C_data,
                p0=[C0_init, k10_init],
                bounds=([1e-3, 1e-4], [1e4, 5.0]),
                maxfev=10000,
            )
            res = float(np.sum((one_compartment(t_data, C0_fit, k10_fit) - C_data) ** 2))
            if res < best_res:
                best_res    = res
                best_result = (C0_fit, k10_fit)
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return {"k10": np.nan, "Vd": np.nan, "CL": np.nan, "C0": np.nan,
                "success": False, "residual": np.nan}

    C0_fit, k10_fit = best_result
    Vd_fit = dose_mg / C0_fit
    return {
        "k10":      float(k10_fit),
        "Vd":       float(Vd_fit),
        "CL":       float(k10_fit * Vd_fit),
        "C0":       float(C0_fit),
        "success":  True,
        "residual": float(best_res),
    }


# ── 2-odeljni benchmark ───────────────────────────────────────────────────────

def _simulate_2comp(
    log_params: np.ndarray,
    t_data:     np.ndarray,
    dose_mg:    float,
) -> np.ndarray:
    """Simulira C1(t) za log_params = [log_k10, log_k12, log_k21, log_V1]."""
    k10, k12, k21, V1 = np.exp(log_params)
    C1_0 = dose_mg / V1
    try:
        sol = solve_ivp(
            two_compartment_ode,
            t_span=(0.0, t_data.max() + 1e-6),
            y0=[C1_0, 0.0],
            args=(k10, k12, k21),
            t_eval=t_data,
            method="RK45",
            rtol=1e-7, atol=1e-9,
        )
        if sol.success and sol.y.shape[1] == len(t_data):
            return sol.y[0]
    except Exception:
        pass
    return np.full(len(t_data), np.nan)


def _nls_2comp(
    log_params: np.ndarray,
    t_data:     np.ndarray,
    C_data:     np.ndarray,
    dose_mg:    float,
) -> float:
    C1_pred = _simulate_2comp(log_params, t_data, dose_mg)
    if np.any(np.isnan(C1_pred)):
        return 1e10
    return float(np.sum((C1_pred - C_data) ** 2))


def fit_two_compartment(
    t_data:     np.ndarray,
    C_data:     np.ndarray,
    dose_mg:    float = DOSE_MG,
    n_restarts: int   = 8,
) -> dict:
    """
    Procena parametara 2-odeljnog modela minimizacijom NLS (multi-start L-BFGS-B).
    Optimizacija u log-prostoru garantuje pozitivnost.

    Returns
    -------
    dict
        {"k10", "k12", "k21", "V1", "CL", "V2", "Vd_ss", "success", "residual"}
    """
    log_lo = np.log([1e-4, 1e-4, 1e-4,  5.0])
    log_hi = np.log([2.0,  3.0,  3.0,  80.0])

    rng = np.random.default_rng(0)
    starts = [np.log([0.10, 0.20, 0.10, 20.0])]   # populaciona sredina
    for _ in range(n_restarts - 1):
        starts.append(rng.uniform(log_lo, log_hi))

    best_x   = None
    best_res = np.inf

    for x0 in starts:
        try:
            res = minimize(
                _nls_2comp,
                x0=x0,
                args=(t_data, C_data, dose_mg),
                method="L-BFGS-B",
                bounds=list(zip(log_lo, log_hi)),
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_res:
                best_res = res.fun
                best_x   = res.x
        except Exception:
            continue

    if best_x is None or np.isinf(best_res):
        return {"k10": np.nan, "k12": np.nan, "k21": np.nan, "V1": np.nan,
                "CL": np.nan, "V2": np.nan, "Vd_ss": np.nan,
                "success": False, "residual": np.nan}

    k10, k12, k21, V1 = np.exp(best_x)
    V2    = V1 * k12 / k21
    Vd_ss = V1 + V2
    return {
        "k10":      float(k10),
        "k12":      float(k12),
        "k21":      float(k21),
        "V1":       float(V1),
        "CL":       float(k10 * V1),
        "V2":       float(V2),
        "Vd_ss":    float(Vd_ss),
        "success":  True,
        "residual": float(best_res),
    }


# ── Brzi test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_processing import solve_two_compartment

    t = np.array([0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24], dtype=float)

    print("Test: fit_one_compartment")
    k10_t, Vd_t = 0.08, 50.0
    C = (DOSE_MG / Vd_t) * np.exp(-k10_t * t)
    r1 = fit_one_compartment(t, C)
    print(f"  k10: {r1['k10']:.5f} (istina {k10_t}), "
          f"Vd: {r1['Vd']:.2f} (istina {Vd_t}), uspesno: {r1['success']}")

    print()
    print("Test: fit_two_compartment")
    k10_t, k12_t, k21_t, V1_t = 0.10, 0.20, 0.10, 20.0
    C1, _ = solve_two_compartment(t, k10_t, k12_t, k21_t, V1_t)
    r2 = fit_two_compartment(t, C1)
    for key, true_val in [("k10", k10_t), ("k12", k12_t), ("k21", k21_t), ("V1", V1_t)]:
        err = abs(r2[key] - true_val) / true_val * 100
        print(f"  {key}: {r2[key]:.5f} (istina {true_val}, greska {err:.2f}%)")
    print(f"  uspesno: {r2['success']}")
