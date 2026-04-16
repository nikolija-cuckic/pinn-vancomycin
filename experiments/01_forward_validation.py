"""
01_forward_validation.py
------------------------
Validacija PINN-a na forward problemu:
  parametri su POZNATI i fiksni — treniramo samo tezine mreze
  da zadovolji ODJ i pocetni uslov, bez mjernih podataka.

Ovo je nezavisan test od inverznog problema:
  - Ako forward ne radi -> greska u physics loss implementaciji
  - Ako forward radi a inverzni ne -> greska u gradijentima parametara

Sto se radi:
  1-odeljni: treniraj mrezu na ODJ dC/dt = -k10*C, C(0)=D/Vd
             uporedjuje sa analitickim rjesenjem C(t) = C0*exp(-k10*t)
  2-odeljni: treniraj mrezu na sistemu 2 ODJ,
             uporedjuje sa numerickim rjesenjem (solve_ivp)

  Za svaki model: testira 3 subjekta sa razlicitim parametrima.
  Mjeri RMSE po rezidualu fizike i po krivoj vs. referentno rjesenje.

Izlaz:
  results/figures/01_forward_1comp.png
  results/figures/01_forward_2comp.png
  results/tables/01_forward_validation.csv

Pokretanje:
  python experiments/01_forward_validation.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_model import PKNet, _grad
from data_processing import (
    one_compartment, solve_two_compartment,
    DOSE_MG, OBS_TIMES,
)

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

T_MAX_H  = 24.0
T_DENSE  = np.linspace(0, T_MAX_H, 500)
N_COL    = 2000    # kolokacione tacke za forward problem


# ══════════════════════════════════════════════════════════════════════════════
# Forward PINN trening (parametri fiksirani)
# ══════════════════════════════════════════════════════════════════════════════

def train_forward_1comp(
    k10:          float,
    Vd:           float,
    dose_mg:      float = DOSE_MG,
    t_max_h:      float = T_MAX_H,
    hidden_layers: int  = 3,
    hidden_size:   int  = 64,
    epochs_adam:   int  = 5000,
    epochs_lbfgs:  int  = 500,
    lambda_ic:     float = 10.0,
    device:        str  = "cpu",
    verbose:       bool = False,
) -> tuple:
    """
    Trenira mrezu da aproksimira rjesenje 1-odeljnog ODJ.
    Parametri k10 i Vd su fiksirani (nisu trenabilni).

    Returns
    -------
    (net, history) — trenirana mreza i lista loss vrijednosti po eposi
    """
    dev = torch.device(device)
    net = PKNet(hidden_layers, hidden_size, n_outputs=1).to(dev)

    # Kolokacione tacke [0, 1]
    t_col = torch.linspace(0, 1, N_COL, device=dev).reshape(-1, 1)

    k10_t = torch.tensor(k10, dtype=torch.float32, device=dev)
    C0    = dose_mg / Vd
    C_max = C0  # normalizacija: C(0) = D/Vd je maksimum za mono-eksponencijal

    def compute_loss():
        # Physics loss: dC_norm/dt_norm + k10*t_max*C_norm = 0
        tc = t_col.requires_grad_(True)
        C_norm = net(tc)
        dC     = _grad(C_norm, tc)
        r      = dC + k10_t * t_max_h * C_norm
        L_phys = (r ** 2).mean()

        # IC loss: C_norm(0) = 1.0  (jer normalizujemo sa C0)
        t0    = torch.zeros(1, 1, device=dev)
        C_ic  = net(t0)
        L_ic  = (C_ic - 1.0) ** 2

        return L_phys + lambda_ic * L_ic.squeeze(), L_phys, L_ic.squeeze()

    # Adam
    opt     = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs_adam):
        opt.zero_grad()
        loss, Lp, Lic = compute_loss()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and epoch % 500 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.4e} "
                  f"phys={Lp.item():.4e} ic={Lic.item():.4e}")

    # L-BFGS
    lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=epochs_lbfgs,
        history_size=50, line_search_fn="strong_wolfe",
    )
    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss()
        loss.backward()
        return loss
    lbfgs.step(closure)

    if verbose:
        final, Lp, Lic = compute_loss()
        print(f"  L-BFGS final | loss={final.item():.4e} "
              f"phys={Lp.item():.4e} ic={Lic.item():.4e}")

    return net, history, C_max


def train_forward_2comp(
    k10:          float,
    k12:          float,
    k21:          float,
    V1:           float,
    dose_mg:      float = DOSE_MG,
    t_max_h:      float = T_MAX_H,
    hidden_layers: int  = 4,
    hidden_size:   int  = 64,
    epochs_adam:   int  = 8000,
    epochs_lbfgs:  int  = 1000,
    lambda_ic:     float = 10.0,
    device:        str  = "cpu",
    verbose:       bool = False,
) -> tuple:
    """
    Trenira mrezu da aproksimira rjesenje 2-odeljnog ODJ sistema.
    Mreza ima 2 izlaza: [C1_norm, C2_norm].

    Returns
    -------
    (net, history, C1_max)
    """
    dev = torch.device(device)
    net = PKNet(hidden_layers, hidden_size, n_outputs=2).to(dev)

    t_col = torch.linspace(0, 1, N_COL, device=dev).reshape(-1, 1)

    k10_t = torch.tensor(k10, dtype=torch.float32, device=dev)
    k12_t = torch.tensor(k12, dtype=torch.float32, device=dev)
    k21_t = torch.tensor(k21, dtype=torch.float32, device=dev)

    C1_0  = dose_mg / V1
    # Normalizacija: koristimo C1(0) = D/V1 kao C1_max
    C1_max = C1_0

    def compute_loss():
        tc  = t_col.requires_grad_(True)
        out = net(tc)               # (N, 2)
        C1  = out[:, 0:1]
        C2  = out[:, 1:2]

        dC1 = _grad(C1, tc)
        dC2 = _grad(C2, tc)

        # Reziduali u normalizovanom prostoru
        r1 = dC1 + (k10_t + k12_t) * t_max_h * C1 - k21_t * t_max_h * C2
        r2 = dC2 - k12_t * t_max_h * C1 + k21_t * t_max_h * C2

        L_phys = (r1 ** 2).mean() + (r2 ** 2).mean()

        # IC: C1_norm(0) = 1.0, C2_norm(0) = 0.0
        t0   = torch.zeros(1, 1, device=dev)
        out0 = net(t0)
        L_ic = (out0[0, 0] - 1.0) ** 2 + out0[0, 1] ** 2

        return L_phys + lambda_ic * L_ic, L_phys, L_ic

    opt     = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs_adam):
        opt.zero_grad()
        loss, Lp, Lic = compute_loss()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and epoch % 1000 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.4e} "
                  f"phys={Lp.item():.4e} ic={Lic.item():.4e}")

    lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=epochs_lbfgs,
        history_size=50, line_search_fn="strong_wolfe",
    )
    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss()
        loss.backward()
        return loss
    lbfgs.step(closure)

    if verbose:
        final, Lp, Lic = compute_loss()
        print(f"  L-BFGS final | loss={final.item():.4e} "
              f"phys={Lp.item():.4e} ic={Lic.item():.4e}")

    return net, history, C1_max


# ══════════════════════════════════════════════════════════════════════════════
# Validacija — 1-odeljni model
# ══════════════════════════════════════════════════════════════════════════════

def validate_1comp():
    print("=" * 60)
    print("FORWARD VALIDACIJA — 1-odeljni model")
    print("=" * 60)

    # 3 subjekta sa razlicitim parametrima (pokriva raspon populacije)
    test_cases = [
        {"label": "Spor eliminatorer",  "k10": 0.042, "Vd": 70.0},
        {"label": "Populaciona sredina","k10": 0.078, "Vd": 47.0},
        {"label": "Brz eliminatorer",   "k10": 0.135, "Vd": 32.0},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rows = []

    for col_i, tc in enumerate(test_cases):
        k10, Vd = tc["k10"], tc["Vd"]
        C0      = DOSE_MG / Vd
        C_true  = one_compartment(T_DENSE, C0, k10)

        print(f"\n  [{col_i+1}/3] {tc['label']} | k10={k10}, Vd={Vd}")
        net, history, C_max = train_forward_1comp(
            k10, Vd, verbose=True,
            epochs_adam=5000, epochs_lbfgs=500,
        )

        # Predikcija na gustoj mrezi
        T_tn = torch.tensor(T_DENSE / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            C_pinn_norm = net(T_tn).numpy().flatten()
        C_pinn = C_pinn_norm * C_max

        # Metrike
        rmse = float(np.sqrt(np.mean((C_pinn - C_true) ** 2)))
        mae  = float(np.mean(np.abs(C_pinn - C_true)))
        max_err = float(np.max(np.abs(C_pinn - C_true)))
        # Relativna greska na tackama merenja
        C_obs_true = one_compartment(OBS_TIMES, C0, k10)
        T_obs_tn = torch.tensor(OBS_TIMES / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            C_obs_pinn = net(T_obs_tn).numpy().flatten() * C_max
        rel_err_pct = float(np.mean(np.abs(C_obs_pinn - C_obs_true) / C_obs_true) * 100)

        print(f"    RMSE={rmse:.4f} mg/L, MAE={mae:.4f} mg/L, "
              f"max_err={max_err:.4f} mg/L, rel_err={rel_err_pct:.3f}%")

        rows.append({
            "model": "1comp", "label": tc["label"],
            "k10": k10, "Vd": Vd,
            "rmse_mg_per_L": rmse,
            "mae_mg_per_L":  mae,
            "max_err_mg_per_L": max_err,
            "rel_err_pct": rel_err_pct,
            "final_loss": history[-1],
        })

        # Gornji red: kriva
        ax = axes[0, col_i]
        ax.plot(T_DENSE, C_true, "k-",  lw=2,   label="Analitičko rješenje", zorder=1)
        ax.plot(T_DENSE, C_pinn, "b--", lw=2,   label=f"PINN (RMSE={rmse:.3f})", zorder=2)
        ax.scatter(OBS_TIMES, C_obs_true, color="k", s=30, zorder=5)
        ax.set_xlabel("Vreme (h)")
        ax.set_ylabel("C (mg/L)")
        ax.set_title(tc["label"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Donji red: loss kriva
        ax = axes[1, col_i]
        ax.semilogy(history, lw=1.2, color="steelblue")
        ax.set_xlabel("Epoha (Adam)")
        ax.set_ylabel("Loss")
        ax.set_title(f"Konvergencija (final={history[-1]:.2e})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Forward validacija — 1-odeljni model\n"
                 "(fiksirani parametri, samo physics + IC loss)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_forward_1comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grafik: {FIG_DIR / '01_forward_1comp.png'}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Validacija — 2-odeljni model
# ══════════════════════════════════════════════════════════════════════════════

def validate_2comp():
    print()
    print("=" * 60)
    print("FORWARD VALIDACIJA — 2-odeljni model")
    print("=" * 60)

    # 3 subjekta (min, median, max raspon populacionih parametara)
    test_cases = [
        {"label": "Mala distribucija",  "k10": 0.06, "k12": 0.12, "k21": 0.06, "V1": 15.0},
        {"label": "Populaciona sredina","k10": 0.10, "k12": 0.20, "k21": 0.10, "V1": 20.0},
        {"label": "Velika distribucija","k10": 0.16, "k12": 0.35, "k21": 0.18, "V1": 28.0},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rows = []

    for col_i, tc in enumerate(test_cases):
        k10, k12, k21, V1 = tc["k10"], tc["k12"], tc["k21"], tc["V1"]

        # Referentno numericko rjesenje
        C1_ref, C2_ref = solve_two_compartment(T_DENSE, k10, k12, k21, V1)
        C1_max = DOSE_MG / V1  # = C1(0)

        print(f"\n  [{col_i+1}/3] {tc['label']} | "
              f"k10={k10}, k12={k12}, k21={k21}, V1={V1}")
        net, history, _ = train_forward_2comp(
            k10, k12, k21, V1, verbose=True,
            epochs_adam=8000, epochs_lbfgs=1000,
        )

        # Predikcija
        T_tn = torch.tensor(T_DENSE / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            out = net(T_tn).numpy()
        C1_pinn = out[:, 0] * C1_max
        C2_pinn = out[:, 1] * C1_max

        # Metrike (samo C1 — merljivo)
        rmse_C1  = float(np.sqrt(np.mean((C1_pinn - C1_ref) ** 2)))
        mae_C1   = float(np.mean(np.abs(C1_pinn - C1_ref)))
        rmse_C2  = float(np.sqrt(np.mean((C2_pinn - C2_ref) ** 2)))
        C1_obs, _ = solve_two_compartment(OBS_TIMES, k10, k12, k21, V1)
        T_obs_tn  = torch.tensor(OBS_TIMES / T_MAX_H, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            C1_obs_pinn = net(T_obs_tn).numpy()[:, 0] * C1_max
        rel_err_pct = float(np.mean(np.abs(C1_obs_pinn - C1_obs) / C1_obs) * 100)

        print(f"    C1 RMSE={rmse_C1:.4f} mg/L, MAE={mae_C1:.4f} mg/L, "
              f"C2 RMSE={rmse_C2:.4f} mg/L, rel_err_C1={rel_err_pct:.3f}%")

        rows.append({
            "model": "2comp", "label": tc["label"],
            "k10": k10, "k12": k12, "k21": k21, "V1": V1,
            "rmse_C1_mg_per_L":  rmse_C1,
            "mae_C1_mg_per_L":   mae_C1,
            "rmse_C2_mg_per_L":  rmse_C2,
            "rel_err_C1_pct":    rel_err_pct,
            "final_loss": history[-1],
        })

        # Gornji red: kriva
        ax = axes[0, col_i]
        ax.plot(T_DENSE, C1_ref,  "k-",   lw=2,   label="C1 numeričko", zorder=1)
        ax.plot(T_DENSE, C1_pinn, "b--",  lw=2,   label=f"C1 PINN (RMSE={rmse_C1:.3f})", zorder=2)
        ax.plot(T_DENSE, C2_ref,  "g-",   lw=1.5, alpha=0.6, label="C2 numeričko", zorder=1)
        ax.plot(T_DENSE, C2_pinn, "g:",   lw=1.5, label="C2 PINN", zorder=2)
        ax.scatter(OBS_TIMES, C1_obs, color="k", s=30, zorder=5, label="C1 tacke")
        ax.set_xlabel("Vreme (h)")
        ax.set_ylabel("C (mg/L)")
        ax.set_title(tc["label"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Donji red: loss kriva
        ax = axes[1, col_i]
        ax.semilogy(history, lw=1.2, color="darkorange")
        ax.set_xlabel("Epoha (Adam)")
        ax.set_ylabel("Loss")
        ax.set_title(f"Konvergencija (final={history[-1]:.2e})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Forward validacija — 2-odeljni model\n"
                 "(fiksirani parametri, physics + IC loss, C2 nemerljivo)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_forward_2comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grafik: {FIG_DIR / '01_forward_2comp.png'}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_rows = []
    all_rows += validate_1comp()
    all_rows += validate_2comp()

    df = pd.DataFrame(all_rows)
    out = TAB_DIR / "01_forward_validation.csv"
    df.to_csv(out, index=False)

    print()
    print("=" * 60)
    print("REZIME — forward validacija")
    print("=" * 60)
    print()
    print("1-odeljni model:")
    for _, r in df[df["model"] == "1comp"].iterrows():
        print(f"  {r['label']:25s} | RMSE={r['rmse_mg_per_L']:.4f} mg/L, "
              f"rel_err={r['rel_err_pct']:.3f}%")
    print()
    print("2-odeljni model:")
    for _, r in df[df["model"] == "2comp"].iterrows():
        print(f"  {r['label']:25s} | C1 RMSE={r['rmse_C1_mg_per_L']:.4f} mg/L, "
              f"C2 RMSE={r['rmse_C2_mg_per_L']:.4f} mg/L, "
              f"rel_err_C1={r['rel_err_C1_pct']:.3f}%")
    print()
    print(f"Rezultati: {out}")
