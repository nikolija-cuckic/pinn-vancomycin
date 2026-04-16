"""
03_sensitivity_analysis.py
--------------------------
Centralni eksperiment rada: kako tacnost procene PK parametara
zavisi od broja merenja (N) i nivoa suma (sigma)?

Petlja: za svaki (N, sigma, seed):
  - nasumicno bira subjekta iz dataseta
  - uzorkuje N tacaka iz punog profila
  - dodaje multiplikativni Gaussov sum sigma
  - pokrece PINN i NLS benchmark
  - belezi greske i upisuje u checkpoint CSV

Parametri eksperimenta:
  N_values     = [3, 5, 8, 10, 12]
  sigma_values = [0.0, 0.05, 0.10, 0.20]
  n_seeds      = 30
  => 5 x 4 x 30 = 600 kombinacija po modelu, 1200 ukupno

Procena trajanja (orijentaciono):
  1-odeljni PINN (~30s/run): ~5h za 600 runova
  2-odeljni PINN (~90s/run): ~15h za 600 runova
  Preporuka: pokrenuti 1comp i 2comp u dva odvojena terminala

Pokretanje:
  # Aktiviraj venv
  source .venv/Scripts/activate        # Windows Git Bash
  # ili: .venv\\Scripts\\activate.bat  # Windows cmd

  # Samo 1-odeljni model
  python experiments/03_sensitivity_analysis.py --model 1comp

  # Samo 2-odeljni model (u drugom terminalu)
  python experiments/03_sensitivity_analysis.py --model 2comp

  # Oba modela sekvencijalno (ne preporucuje se, veoma dugo)
  python experiments/03_sensitivity_analysis.py --model both

  # Test mod: 2 seeda, jedan subjekt (provjera da kod radi)
  python experiments/03_sensitivity_analysis.py --model 1comp --test

Izlaz:
  results/tables/sensitivity_1comp.csv
  results/tables/sensitivity_2comp.csv

Checkpoint: svaki run se odmah upisuje u CSV (append), tako da
prekidanje i nastavljanje ne gubi prethodne rezultate.
Ako CSV vec postoji, preskace se vec zavrsene kombinacije.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")   # scipy konvergencija, torch deprecation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing import DOSE_MG
from inverse_problem import (
    subsample_1comp, subsample_2comp,
    run_inverse_1comp, run_inverse_2comp,
)

PROC_DIR = ROOT / "data" / "processed"
TAB_DIR  = ROOT / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Parametri eksperimenta ────────────────────────────────────────────────────

N_VALUES     = [3, 5, 8, 10, 12]
SIGMA_VALUES = [0.0, 0.05, 0.10, 0.20]
N_SEEDS      = 30
T_MAX_H      = 24.0

# Trajanje treninga po modelu
# Manje epoha nego u PoC-u — ubrzanje x2, minimalna degradacija tacnosti
ADAM_1COMP   = 3000
LBFGS_1COMP  = 200
ADAM_2COMP   = 5000
LBFGS_2COMP  = 500

# Test mod (krace trajanje)
ADAM_1COMP_TEST  = 500
LBFGS_1COMP_TEST = 50
ADAM_2COMP_TEST  = 500
LBFGS_2COMP_TEST = 50


# ── Pomocne funkcije ──────────────────────────────────────────────────────────

def already_done(out_path: Path, N: int, sigma: float, seed: int) -> bool:
    """Provjeri da li je ova kombinacija vec u checkpoint CSV-u."""
    if not out_path.exists():
        return False
    try:
        df = pd.read_csv(out_path)
        mask = (df["N"] == N) & (df["sigma"] == sigma) & (df["seed"] == seed)
        return mask.any()
    except Exception:
        return False


def append_row(out_path: Path, row: dict) -> None:
    """Dodaj jedan red u CSV (header samo ako fajl ne postoji)."""
    df = pd.DataFrame([row])
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def total_runs(n_values, sigma_values, n_seeds):
    return len(n_values) * len(sigma_values) * n_seeds


# ═══════════════════════════════════════════════════════════════════════════════
# 1-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_1comp(
    test_mode: bool = False,
    out_dir:   "Path | None" = None,
    device:    str = "cpu",
) -> None:
    eff_dir  = Path(out_dir) if out_dir else TAB_DIR
    eff_dir.mkdir(parents=True, exist_ok=True)
    out_path = eff_dir / "sensitivity_1comp.csv"

    profiles = pd.read_csv(PROC_DIR / "subject_profiles_1comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_1comp.csv")
    subjects = profiles["subject_id"].unique()
    n_subj   = len(subjects)

    epochs_adam  = ADAM_1COMP_TEST  if test_mode else ADAM_1COMP
    epochs_lbfgs = LBFGS_1COMP_TEST if test_mode else LBFGS_1COMP

    n_values     = N_VALUES[:2]     if test_mode else N_VALUES
    sigma_values = SIGMA_VALUES[:2] if test_mode else SIGMA_VALUES
    n_seeds      = 2                if test_mode else N_SEEDS

    total = total_runs(n_values, sigma_values, n_seeds)
    done  = 0

    print(f"\n{'='*60}")
    print(f"1-ODELJNI MODEL — sensitivity analysis")
    print(f"  N: {n_values}")
    print(f"  sigma: {sigma_values}")
    print(f"  seeds: {n_seeds}   => {total} runova")
    print(f"  checkpoint: {out_path}")
    if test_mode:
        print("  [TEST MOD]")
    print(f"{'='*60}\n")

    t_start = time.time()

    for N in n_values:
        for sigma in sigma_values:
            for seed in range(n_seeds):

                if already_done(out_path, N, sigma, seed):
                    done += 1
                    continue

                # Odabir subjekta: rotiramo kroz sve subjekte po seed-u
                sid    = subjects[seed % n_subj]
                gt_row = gt_all[gt_all["subject_id"] == sid].iloc[0]
                gt     = {"k10": gt_row["k10"], "Vd": gt_row["Vd"]}

                t0 = time.time()
                try:
                    t_sp, C_sp = subsample_1comp(profiles, sid, N, sigma, seed)
                    result = run_inverse_1comp(
                        t_sp, C_sp, gt,
                        dose_mg      = DOSE_MG,
                        t_max_h      = T_MAX_H,
                        epochs_adam  = epochs_adam,
                        epochs_lbfgs = epochs_lbfgs,
                        device       = device,
                        verbose      = False,
                    )
                    status = "ok"
                except Exception as e:
                    result = {k: np.nan for k in [
                        "pinn_k10","pinn_Vd","pinn_CL",
                        "pinn_err_k10","pinn_err_Vd",
                        "bench_k10","bench_Vd","bench_CL",
                        "bench_err_k10","bench_err_Vd","bench_success",
                    ]}
                    status = f"error: {e}"

                elapsed = time.time() - t0
                done   += 1

                row = {
                    "N": N, "sigma": sigma, "seed": seed,
                    "subject_id": sid,
                    "gt_k10": gt["k10"], "gt_Vd": gt["Vd"],
                    "elapsed_s": round(elapsed, 1),
                    "status": status,
                    **result,
                }
                append_row(out_path, row)

                # Progres
                elapsed_total = time.time() - t_start
                rate = done / elapsed_total if elapsed_total > 0 else 1
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{total}] N={N:2d} sigma={sigma:.2f} "
                      f"seed={seed:2d} | {elapsed:.1f}s | "
                      f"PINN err_k10={result.get('pinn_err_k10', float('nan')):.1f}% "
                      f"bench={result.get('bench_err_k10', float('nan')):.1f}% | "
                      f"ETA {eta/60:.0f}min")

    print(f"\n1-odeljni zavrseno. Sacuvano: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_2comp(
    test_mode: bool = False,
    out_dir:   "Path | None" = None,
    device:    str = "cpu",
) -> None:
    eff_dir  = Path(out_dir) if out_dir else TAB_DIR
    eff_dir.mkdir(parents=True, exist_ok=True)
    out_path = eff_dir / "sensitivity_2comp.csv"

    profiles = pd.read_csv(PROC_DIR / "subject_profiles_2comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_2comp.csv")
    subjects = profiles["subject_id"].unique()
    n_subj   = len(subjects)

    epochs_adam  = ADAM_2COMP_TEST  if test_mode else ADAM_2COMP
    epochs_lbfgs = LBFGS_2COMP_TEST if test_mode else LBFGS_2COMP

    n_values     = N_VALUES[:2]     if test_mode else N_VALUES
    sigma_values = SIGMA_VALUES[:2] if test_mode else SIGMA_VALUES
    n_seeds      = 2                if test_mode else N_SEEDS

    total = total_runs(n_values, sigma_values, n_seeds)
    done  = 0

    print(f"\n{'='*60}")
    print(f"2-ODELJNI MODEL — sensitivity analysis")
    print(f"  N: {n_values}")
    print(f"  sigma: {sigma_values}")
    print(f"  seeds: {n_seeds}   => {total} runova")
    print(f"  checkpoint: {out_path}")
    if test_mode:
        print("  [TEST MOD]")
    print(f"{'='*60}\n")

    t_start = time.time()

    for N in n_values:
        for sigma in sigma_values:
            for seed in range(n_seeds):

                if already_done(out_path, N, sigma, seed):
                    done += 1
                    continue

                sid    = subjects[seed % n_subj]
                gt_row = gt_all[gt_all["subject_id"] == sid].iloc[0]
                gt     = {
                    "k10": gt_row["k10"], "k12": gt_row["k12"],
                    "k21": gt_row["k21"], "V1":  gt_row["V1"],
                }

                t0 = time.time()
                try:
                    t_sp, C_sp = subsample_2comp(profiles, sid, N, sigma, seed)
                    result = run_inverse_2comp(
                        t_sp, C_sp, gt,
                        dose_mg      = DOSE_MG,
                        t_max_h      = T_MAX_H,
                        epochs_adam  = epochs_adam,
                        epochs_lbfgs = epochs_lbfgs,
                        device       = device,
                        verbose      = False,
                    )
                    status = "ok"
                except Exception as e:
                    result = {k: np.nan for k in [
                        "pinn_k10","pinn_k12","pinn_k21","pinn_V1","pinn_CL",
                        "pinn_err_k10","pinn_err_k12","pinn_err_k21","pinn_err_V1",
                        "bench_k10","bench_k12","bench_k21","bench_V1","bench_CL",
                        "bench_err_k10","bench_err_k12","bench_err_k21","bench_err_V1",
                        "bench_success",
                    ]}
                    status = f"error: {e}"

                elapsed = time.time() - t0
                done   += 1

                row = {
                    "N": N, "sigma": sigma, "seed": seed,
                    "subject_id": sid,
                    "gt_k10": gt["k10"], "gt_k12": gt["k12"],
                    "gt_k21": gt["k21"], "gt_V1":  gt["V1"],
                    "elapsed_s": round(elapsed, 1),
                    "status": status,
                    **result,
                }
                append_row(out_path, row)

                elapsed_total = time.time() - t_start
                rate = done / elapsed_total if elapsed_total > 0 else 1
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{total}] N={N:2d} sigma={sigma:.2f} "
                      f"seed={seed:2d} | {elapsed:.1f}s | "
                      f"PINN err_k10={result.get('pinn_err_k10', float('nan')):.1f}% "
                      f"bench={result.get('bench_err_k10', float('nan')):.1f}% | "
                      f"ETA {eta/60:.0f}min")

    print(f"\n2-odeljni zavrseno. Sacuvano: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: PINN vs benchmark, N x sigma x seed"
    )
    parser.add_argument(
        "--model",
        choices=["1comp", "2comp", "both"],
        default="both",
        help="Koji model pokrenuti (default: both)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mod: 2 seeda, 2 N vrednosti, 500 Adam epoha",
    )
    args = parser.parse_args()

    if args.model in ("1comp", "both"):
        run_sensitivity_1comp(test_mode=args.test)

    if args.model in ("2comp", "both"):
        run_sensitivity_2comp(test_mode=args.test)

    print("\nSensitivity analysis zavrsena.")
    print(f"Rezultati su u: {TAB_DIR}")
