"""
02_inverse_full_data.py
-----------------------
Proof-of-concept: inverzni problem na PUNOM profilu (N=12, sigma=0).

Za oba modela (1-odeljni i 2-odeljni):
  - Ucitava prvi subjekt iz dataseta
  - Pokrece PINN i benchmark
  - Stampa procenjene parametre i greske
  - Crta profil: istiniti vs PINN vs benchmark

Rezultati se cuvaju u:
  results/figures/02_poc_1comp.png
  results/figures/02_poc_2comp.png
  results/tables/02_poc_results.csv

Pokretanje:
  cd pinn-vancomycin
  python experiments/02_inverse_full_data.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing import (
    one_compartment, solve_two_compartment, DOSE_MG, OBS_TIMES
)
from pinn_model import (
    OneCompartmentPINN, TwoCompartmentPINN, train_pinn
)
from benchmark import fit_one_compartment, fit_two_compartment
from metrics import param_errors, curve_rmse

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PROC_DIR = ROOT / "data" / "processed"

T_MAX_H = 24.0
T_DENSE = np.linspace(0, T_MAX_H, 500)     # za crtanje glatke krive


# ═══════════════════════════════════════════════════════════════════════════════
# 1-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_1comp():
    print("=" * 60)
    print("1-ODELJNI MODEL — proof of concept (N=12, sigma=0)")
    print("=" * 60)

    # Ucitaj podatke
    profiles = pd.read_csv(PROC_DIR / "subject_profiles_1comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_1comp.csv")

    # Uzmi prvog subjekta
    sid      = profiles["subject_id"].unique()[0]
    subj     = profiles[profiles["subject_id"] == sid]
    gt_row   = gt_all[gt_all["subject_id"] == sid].iloc[0]

    t_h    = subj["time_h"].values
    C_true = subj["C_true"].values
    C_max  = C_true.max()
    t_norm = t_h / T_MAX_H
    C_norm = C_true / C_max

    gt = {"k10": gt_row["k10"], "Vd": gt_row["Vd"]}
    print(f"\nSubjekt: {sid}")
    print(f"  ground truth  k10={gt['k10']:.5f} 1/h,  Vd={gt['Vd']:.2f} L")

    # ── PINN ──────────────────────────────────────────────────────────────────
    print("\nTreniranje PINN-a...")
    model = OneCompartmentPINN(dose_mg=DOSE_MG, t_max_h=T_MAX_H)
    history = train_pinn(
        model, t_norm, C_norm, T_MAX_H, C_max,
        epochs_adam=5000, epochs_lbfgs=500, verbose=True,
    )
    pinn_p = model.get_parameters()
    pinn_e = param_errors(pinn_p, gt)
    print(f"\nPINN procena: k10={pinn_p['k10']:.5f}, Vd={pinn_p['Vd']:.2f}")
    print(f"  err_k10={pinn_e['err_k10']:.3f}%, err_Vd={pinn_e['err_Vd']:.3f}%")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\nBenchmark (curve_fit)...")
    bench   = fit_one_compartment(t_h, C_true)
    bench_e = param_errors({"k10": bench["k10"], "Vd": bench["Vd"]}, gt)
    print(f"Benchmark:    k10={bench['k10']:.5f}, Vd={bench['Vd']:.2f}")
    print(f"  err_k10={bench_e['err_k10']:.3f}%, err_Vd={bench_e['err_Vd']:.3f}%")

    # ── Predvidjanje na gustoj mrezi ───────────────────────────────────────────
    import torch
    T_norm_dense = torch.tensor(T_DENSE / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        C_pinn_norm = model(T_norm_dense).numpy().flatten()
    C_pinn = C_pinn_norm * C_max

    # curve_fit predvidjanje
    C0_bench = DOSE_MG / bench["Vd"]
    C_bench  = one_compartment(T_DENSE, C0_bench, bench["k10"])

    # RMSE na merenim tackama
    def pinn_pred(t): return (
        model(torch.tensor(t / T_MAX_H, dtype=torch.float32).reshape(-1, 1))
        .detach().numpy().flatten() * C_max
    )
    rmse_pinn  = curve_rmse(pinn_pred,  t_h, C_true)
    rmse_bench = curve_rmse(lambda t: one_compartment(t, C0_bench, bench["k10"]), t_h, C_true)

    # ── Vizualizacija ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Kriva
    ax = axes[0]
    ax.plot(T_DENSE, C_pinn,  "b-",  lw=2,   label=f"PINN (k10={pinn_p['k10']:.4f})", zorder=3)
    ax.plot(T_DENSE, C_bench, "r--", lw=2,   label=f"curve_fit (k10={bench['k10']:.4f})", zorder=2)
    ax.scatter(t_h, C_true, color="k", s=50, zorder=5, label="Merenja (puni profil)")
    ax.set_xlabel("Vreme (h)")
    ax.set_ylabel("Koncentracija (mg/L)")
    ax.set_title(f"1-odeljni model — {sid}\nRMSE: PINN={rmse_pinn:.3f}, bench={rmse_bench:.3f} mg/L")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Loss kriva (Adam faza)
    ax = axes[1]
    ax.semilogy(history, lw=1.2, color="steelblue")
    ax.set_xlabel("Epoha (Adam)")
    ax.set_ylabel("Ukupni loss")
    ax.set_title("Konvergencija PINN-a (Adam faza)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_poc_1comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrafik sacuvan: {FIG_DIR / '02_poc_1comp.png'}")

    return {
        "model": "1comp",
        "subject_id": sid,
        "gt_k10": gt["k10"], "gt_Vd": gt["Vd"],
        "pinn_k10": pinn_p["k10"], "pinn_Vd": pinn_p["Vd"],
        "pinn_err_k10": pinn_e["err_k10"], "pinn_err_Vd": pinn_e["err_Vd"],
        "bench_k10": bench["k10"], "bench_Vd": bench["Vd"],
        "bench_err_k10": bench_e["err_k10"], "bench_err_Vd": bench_e["err_Vd"],
        "pinn_rmse": rmse_pinn, "bench_rmse": rmse_bench,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_2comp():
    print()
    print("=" * 60)
    print("2-ODELJNI MODEL — proof of concept (N=12, sigma=0)")
    print("=" * 60)

    profiles = pd.read_csv(PROC_DIR / "subject_profiles_2comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_2comp.csv")

    sid    = profiles["subject_id"].unique()[0]
    subj   = profiles[profiles["subject_id"] == sid]
    gt_row = gt_all[gt_all["subject_id"] == sid].iloc[0]

    t_h     = subj["time_h"].values
    C1_true = subj["C1_true"].values
    C1_max  = C1_true.max()
    t_norm  = t_h / T_MAX_H
    C1_norm = C1_true / C1_max

    gt = {"k10": gt_row["k10"], "k12": gt_row["k12"],
          "k21": gt_row["k21"], "V1":  gt_row["V1"]}
    print(f"\nSubjekt: {sid}")
    print(f"  ground truth  k10={gt['k10']:.5f}, k12={gt['k12']:.5f}, "
          f"k21={gt['k21']:.5f}, V1={gt['V1']:.2f} L")

    # ── PINN ──────────────────────────────────────────────────────────────────
    print("\nTreniranje PINN-a (2-odeljni)...")
    model = TwoCompartmentPINN(dose_mg=DOSE_MG, t_max_h=T_MAX_H)
    history = train_pinn(
        model, t_norm, C1_norm, T_MAX_H, C1_max,
        epochs_adam=8000, epochs_lbfgs=1000, verbose=True,
    )
    pinn_p = model.get_parameters()
    pinn_e = param_errors(pinn_p, gt)
    print(f"\nPINN procena:")
    for k in ["k10", "k12", "k21", "V1"]:
        print(f"  {k}: {pinn_p[k]:.5f}  (istina {gt[k]:.5f}, "
              f"greska {pinn_e.get('err_'+k, float('nan')):.2f}%)")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\nBenchmark (NLS + solve_ivp)...")
    bench   = fit_two_compartment(t_h, C1_true)
    bench_e = (param_errors({k: bench[k] for k in ["k10", "k12", "k21", "V1"]}, gt)
               if bench["success"] else {f"err_{k}": np.nan for k in gt})
    print(f"Benchmark procena (uspesno={bench['success']}):")
    for k in ["k10", "k12", "k21", "V1"]:
        print(f"  {k}: {bench.get(k, float('nan')):.5f}  "
              f"(greska {bench_e.get('err_'+k, float('nan')):.2f}%)")

    # ── Predvidjanje ──────────────────────────────────────────────────────────
    import torch
    T_norm_dense = torch.tensor(T_DENSE / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        out_dense = model(T_norm_dense).numpy()
    C1_pinn = out_dense[:, 0] * C1_max
    C2_pinn = out_dense[:, 1] * C1_max

    # Istiniti C1 i C2 na gustoj mrezi (za referencu)
    C1_ref, C2_ref = solve_two_compartment(T_DENSE, gt["k10"], gt["k12"], gt["k21"], gt["V1"])

    # Benchmark predvidjanje
    if bench["success"]:
        C1_bench, _ = solve_two_compartment(
            T_DENSE, bench["k10"], bench["k12"], bench["k21"], bench["V1"]
        )
    else:
        C1_bench = np.full(len(T_DENSE), np.nan)

    def pinn_pred_C1(t):
        import torch as _torch
        tn = _torch.tensor(t / T_MAX_H, dtype=_torch.float32).reshape(-1, 1)
        with _torch.no_grad():
            return model(tn).numpy()[:, 0] * C1_max

    rmse_pinn  = curve_rmse(pinn_pred_C1, t_h, C1_true)
    rmse_bench = (curve_rmse(
        lambda t: solve_two_compartment(t, bench["k10"], bench["k12"], bench["k21"], bench["V1"])[0],
        t_h, C1_true
    ) if bench["success"] else np.nan)

    # ── Vizualizacija ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # C1 kriva
    ax = axes[0]
    ax.plot(T_DENSE, C1_ref,   "k-",  lw=1.5, alpha=0.5, label="Istiniti C1", zorder=1)
    ax.plot(T_DENSE, C1_pinn,  "b-",  lw=2,              label=f"PINN C1 (RMSE={rmse_pinn:.2f})", zorder=3)
    ax.plot(T_DENSE, C1_bench, "r--", lw=2,              label=f"benchmark C1 (RMSE={rmse_bench:.2f})", zorder=2)
    ax.plot(T_DENSE, C2_pinn,  "b:",  lw=1.5,            label="PINN C2 (nemerljivo)", zorder=3)
    ax.plot(T_DENSE, C2_ref,   "g:",  lw=1.5, alpha=0.6, label="Istiniti C2 (ref)", zorder=1)
    ax.scatter(t_h, C1_true, color="k", s=50, zorder=5, label="Merenja C1")
    ax.set_xlabel("Vreme (h)")
    ax.set_ylabel("Koncentracija (mg/L)")
    ax.set_title(f"2-odeljni model — {sid}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Loss kriva
    ax = axes[1]
    ax.semilogy(history, lw=1.2, color="steelblue")
    ax.set_xlabel("Epoha (Adam)")
    ax.set_ylabel("Ukupni loss")
    ax.set_title("Konvergencija PINN-a (Adam faza)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_poc_2comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrafik sacuvan: {FIG_DIR / '02_poc_2comp.png'}")

    return {
        "model": "2comp",
        "subject_id": sid,
        **{f"gt_{k}":    gt[k]                           for k in ["k10", "k12", "k21", "V1"]},
        **{f"pinn_{k}":  pinn_p[k]                       for k in ["k10", "k12", "k21", "V1"]},
        **{f"pinn_err_{k}": pinn_e.get(f"err_{k}", np.nan) for k in ["k10", "k12", "k21", "V1"]},
        **{f"bench_{k}": bench.get(k, np.nan)            for k in ["k10", "k12", "k21", "V1"]},
        **{f"bench_err_{k}": bench_e.get(f"err_{k}", np.nan) for k in ["k10", "k12", "k21", "V1"]},
        "pinn_rmse": rmse_pinn,
        "bench_rmse": rmse_bench,
        "bench_success": bench["success"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = []

    r1 = run_1comp()
    results.append(r1)

    r2 = run_2comp()
    results.append(r2)

    # Sacuvaj tabelu
    out_path = TAB_DIR / "02_poc_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nRezultati sacuvani: {out_path}")

    print()
    print("=" * 60)
    print("REZIME — proof of concept (puni profil, bez suma)")
    print("=" * 60)
    print(f"\n1-odeljni model:")
    print(f"  PINN:      err_k10={r1['pinn_err_k10']:.3f}%,  err_Vd={r1['pinn_err_Vd']:.3f}%,  RMSE={r1['pinn_rmse']:.4f} mg/L")
    print(f"  curve_fit: err_k10={r1['bench_err_k10']:.3f}%, err_Vd={r1['bench_err_Vd']:.3f}%, RMSE={r1['bench_rmse']:.4f} mg/L")
    print(f"\n2-odeljni model:")
    print(f"  PINN:      err_k10={r2['pinn_err_k10']:.2f}%, err_k12={r2['pinn_err_k12']:.2f}%, "
          f"err_k21={r2['pinn_err_k21']:.2f}%, err_V1={r2['pinn_err_V1']:.2f}%,  RMSE={r2['pinn_rmse']:.4f} mg/L")
    print(f"  NLS bench: err_k10={r2['bench_err_k10']:.2f}%, err_k12={r2['bench_err_k12']:.2f}%, "
          f"err_k21={r2['bench_err_k21']:.2f}%, err_V1={r2['bench_err_V1']:.2f}%, RMSE={r2['bench_rmse']:.4f} mg/L")
    print(f"  bench_success={r2['bench_success']}")
