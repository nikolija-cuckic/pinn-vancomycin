"""
04_pinn_ablation.py
-------------------
Pokreće samo PINN (bez benchmarka) na ISTIM uzorcima kao
03_sensitivity_analysis.py, ali sa novim početnim vrednostima parametara.

Koristi se kada je promijenjena inicijalizacija u pinn_model.py i treba
porediti nove PINN rezultate sa benchmark-om iz originalnog CSV-a.
Benchmark se ne računa ponovo — benchmark redovi ostaju nepromijenjeni.

Parametri eksperimenta identični kao u 03:
  N_values     = [3, 5, 8, 10, 12]
  sigma_values = [0.0, 0.05, 0.10, 0.20]
  n_seeds      = 30

Uzorci su reproduktivni: isti (subject_id, N, sigma, seed) → isti subsampled podaci.

Pokretanje:
  python experiments/04_pinn_ablation.py --model 1comp
  python experiments/04_pinn_ablation.py --model 2comp
  python experiments/04_pinn_ablation.py --model 2comp --seeds 0 1
  python experiments/04_pinn_ablation.py --model 2comp --seeds 2 3 4 ... 29
  python experiments/04_pinn_ablation.py --model both --test

Izlaz:
  results/tables/sensitivity_1comp_ablation.csv
  results/tables/sensitivity_2comp_ablation.csv

Checkpoint: svaki run se odmah upisuje u CSV (append) — prekid i nastavak
ne gube prethodne rezultate. Već završene (N, sigma, seed) kombinacije se preskaču.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing import DOSE_MG
from inverse_problem import subsample_1comp, subsample_2comp
from pinn_model import OneCompartmentPINN, TwoCompartmentPINN, train_pinn
from metrics import param_errors

PROC_DIR = ROOT / "data" / "processed"
TAB_DIR  = ROOT / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Parametri eksperimenta (identični 03) ─────────────────────────────────────

N_VALUES     = [3, 5, 8, 10, 12]
SIGMA_VALUES = [0.0, 0.05, 0.10, 0.20]
N_SEEDS      = 30
T_MAX_H      = 24.0

ADAM_1COMP   = 3000
LBFGS_1COMP  = 200
ADAM_2COMP   = 5000
LBFGS_2COMP  = 500

ADAM_1COMP_TEST  = 500
LBFGS_1COMP_TEST = 50
ADAM_2COMP_TEST  = 500
LBFGS_2COMP_TEST = 50


# ── Pomoćne funkcije ──────────────────────────────────────────────────────────

def already_done(out_path: Path, N: int, sigma: float, seed: int) -> bool:
    if not out_path.exists():
        return False
    try:
        df   = pd.read_csv(out_path)
        mask = (df["N"] == N) & (df["sigma"] == sigma) & (df["seed"] == seed)
        return mask.any()
    except Exception:
        return False


def append_row(out_path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def total_runs(n_values, sigma_values, seeds):
    return len(n_values) * len(sigma_values) * len(seeds)


# ═══════════════════════════════════════════════════════════════════════════════
# 1-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_1comp(
    test_mode:   bool = False,
    out_dir:     "Path | None" = None,
    device:      str  = "cpu",
    seed_subset: "list | None" = None,
) -> None:
    """
    Pokreće PINN za 1-odeljni model na istim uzorcima kao 03_sensitivity_analysis.

    Parameters
    ----------
    seed_subset : list, optional
        Ako je zadato, pokreće samo seedove iz ove liste (npr. [0, 1]).
        Korisno za raspodjelu po GPU-ima.
    """
    eff_dir  = Path(out_dir) if out_dir else TAB_DIR
    eff_dir.mkdir(parents=True, exist_ok=True)
    out_path = eff_dir / "sensitivity_1comp_ablation.csv"

    profiles = pd.read_csv(PROC_DIR / "subject_profiles_1comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_1comp.csv")
    subjects = profiles["subject_id"].unique()
    n_subj   = len(subjects)

    epochs_adam  = ADAM_1COMP_TEST  if test_mode else ADAM_1COMP
    epochs_lbfgs = LBFGS_1COMP_TEST if test_mode else LBFGS_1COMP

    n_values     = N_VALUES[:2]     if test_mode else N_VALUES
    sigma_values = SIGMA_VALUES[:2] if test_mode else SIGMA_VALUES
    seeds        = list(range(2 if test_mode else N_SEEDS))

    if seed_subset is not None:
        seeds = [s for s in seeds if s in seed_subset]

    total = total_runs(n_values, sigma_values, seeds)
    done  = 0

    print(f"\n{'='*60}")
    print(f"1-ODELJNI MODEL — PINN ablation")
    print(f"  N: {n_values}")
    print(f"  sigma: {sigma_values}")
    print(f"  seeds: {seeds}  => {total} runova")
    print(f"  device: {device}")
    print(f"  checkpoint: {out_path}")
    if test_mode:
        print("  [TEST MOD]")
    print(f"{'='*60}\n")

    t_start = time.time()

    for N in n_values:
        for sigma in sigma_values:
            for seed in seeds:

                if already_done(out_path, N, sigma, seed):
                    done += 1
                    continue

                # Isti subject kao u 03 — reproduktivno
                sid    = subjects[seed % n_subj]
                gt_row = gt_all[gt_all["subject_id"] == sid].iloc[0]
                gt     = {"k10": gt_row["k10"], "Vd": gt_row["Vd"]}

                t0 = time.time()
                try:
                    # Isti uzorci: isti seed → isti random izbor tačaka i šum
                    t_sp, C_sp = subsample_1comp(profiles, sid, N, sigma, seed)

                    C_max  = C_sp.max()
                    t_norm = t_sp / T_MAX_H
                    C_norm = C_sp / C_max

                    model = OneCompartmentPINN(
                        dose_mg=DOSE_MG, t_max_h=T_MAX_H, device=device
                    )
                    train_pinn(
                        model, t_norm, C_norm, T_MAX_H, C_max,
                        epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs,
                        verbose=False,
                    )
                    pinn_p = model.get_parameters()
                    pinn_e = param_errors(pinn_p, {"k10": gt["k10"], "Vd": gt["Vd"]})

                    result = {
                        "pinn_k10":     pinn_p["k10"],
                        "pinn_Vd":      pinn_p["Vd"],
                        "pinn_CL":      pinn_p["CL"],
                        "pinn_err_k10": pinn_e.get("err_k10", np.nan),
                        "pinn_err_Vd":  pinn_e.get("err_Vd",  np.nan),
                    }
                    status = "ok"

                except Exception as e:
                    result = {k: np.nan for k in [
                        "pinn_k10", "pinn_Vd", "pinn_CL",
                        "pinn_err_k10", "pinn_err_Vd",
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

                elapsed_total = time.time() - t_start
                rate = done / elapsed_total if elapsed_total > 0 else 1
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{total}] N={N:2d} sigma={sigma:.2f} "
                      f"seed={seed:2d} | {elapsed:.1f}s | "
                      f"pinn_err_k10={result.get('pinn_err_k10', float('nan')):.1f}% | "
                      f"ETA {eta/60:.0f}min")

    print(f"\n1-odeljni ablation završen. Sačuvano: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2-ODELJNI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_2comp(
    test_mode:   bool = False,
    out_dir:     "Path | None" = None,
    device:      str  = "cpu",
    seed_subset: "list | None" = None,
    n_subset:    "list | None" = None,
) -> None:
    """
    Pokreće PINN za 2-odeljni model na istim uzorcima kao 03_sensitivity_analysis.

    Parameters
    ----------
    seed_subset : list, optional
        Pokreće samo seedove iz ove liste (npr. [0, 1] za prvu grupu,
        list(range(2, 30)) za drugu grupu).
    n_subset : list, optional
        Pokreće samo zadane N vrednosti (npr. [10, 12]).
    """
    eff_dir  = Path(out_dir) if out_dir else TAB_DIR
    eff_dir.mkdir(parents=True, exist_ok=True)
    out_path = eff_dir / "sensitivity_2comp_ablation.csv"

    profiles = pd.read_csv(PROC_DIR / "subject_profiles_2comp.csv")
    gt_all   = pd.read_csv(PROC_DIR / "ground_truth_params_2comp.csv")
    subjects = profiles["subject_id"].unique()
    n_subj   = len(subjects)

    epochs_adam  = ADAM_2COMP_TEST  if test_mode else ADAM_2COMP
    epochs_lbfgs = LBFGS_2COMP_TEST if test_mode else LBFGS_2COMP

    n_values     = N_VALUES[:2]     if test_mode else N_VALUES
    sigma_values = SIGMA_VALUES[:2] if test_mode else SIGMA_VALUES
    seeds        = list(range(2 if test_mode else N_SEEDS))

    if n_subset is not None:
        n_values = [n for n in n_values if n in n_subset]
    if seed_subset is not None:
        seeds = [s for s in seeds if s in seed_subset]

    total = total_runs(n_values, sigma_values, seeds)
    done  = 0

    print(f"\n{'='*60}")
    print(f"2-ODELJNI MODEL — PINN ablation")
    print(f"  N: {n_values}")
    print(f"  sigma: {sigma_values}")
    print(f"  seeds: {seeds}  => {total} runova")
    print(f"  device: {device}")
    print(f"  checkpoint: {out_path}")
    if test_mode:
        print("  [TEST MOD]")
    print(f"{'='*60}\n")

    t_start = time.time()

    for N in n_values:
        for sigma in sigma_values:
            for seed in seeds:

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

                    C1_max  = C_sp.max()
                    t_norm  = t_sp / T_MAX_H
                    C1_norm = C_sp / C1_max

                    model = TwoCompartmentPINN(
                        dose_mg=DOSE_MG, t_max_h=T_MAX_H, device=device
                    )
                    train_pinn(
                        model, t_norm, C1_norm, T_MAX_H, C1_max,
                        epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs,
                        verbose=False,
                    )
                    pinn_p = model.get_parameters()
                    pinn_e = param_errors(
                        pinn_p,
                        {"k10": gt["k10"], "k12": gt["k12"],
                         "k21": gt["k21"], "V1":  gt["V1"]},
                    )

                    result = {
                        "pinn_k10":     pinn_p["k10"],
                        "pinn_k12":     pinn_p["k12"],
                        "pinn_k21":     pinn_p["k21"],
                        "pinn_V1":      pinn_p["V1"],
                        "pinn_CL":      pinn_p["CL"],
                        "pinn_err_k10": pinn_e.get("err_k10", np.nan),
                        "pinn_err_k12": pinn_e.get("err_k12", np.nan),
                        "pinn_err_k21": pinn_e.get("err_k21", np.nan),
                        "pinn_err_V1":  pinn_e.get("err_V1",  np.nan),
                    }
                    status = "ok"

                except Exception as e:
                    result = {k: np.nan for k in [
                        "pinn_k10", "pinn_k12", "pinn_k21", "pinn_V1", "pinn_CL",
                        "pinn_err_k10", "pinn_err_k12", "pinn_err_k21", "pinn_err_V1",
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
                      f"pinn_err_k10={result.get('pinn_err_k10', float('nan')):.1f}% | "
                      f"ETA {eta/60:.0f}min")

    print(f"\n2-odeljni ablation završen. Sačuvano: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PINN ablation: pokreće samo PINN na istim uzorcima kao 03"
    )
    parser.add_argument(
        "--model",
        choices=["1comp", "2comp", "both"],
        default="both",
        help="Koji model pokrenuti (default: both)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="Pokrenuti samo ove seed-ove (npr. --seeds 0 1  ili  --seeds 2 3 4 ... 29)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mod: 2 seeda, 2 N vrednosti, 500 Adam epoha",
    )
    parser.add_argument(
        "--device0",
        default="cpu",
        help="PyTorch device za 1-comp (default: cpu)",
    )
    parser.add_argument(
        "--device1",
        default="cpu",
        help="PyTorch device za 2-comp (default: cpu)",
    )
    args = parser.parse_args()

    if args.model in ("1comp", "both"):
        run_ablation_1comp(
            test_mode=args.test,
            device=args.device0,
            seed_subset=args.seeds,
        )

    if args.model in ("2comp", "both"):
        run_ablation_2comp(
            test_mode=args.test,
            device=args.device1,
            seed_subset=args.seeds,
        )

    print("\nPINN ablation završena.")
    print(f"Rezultati su u: {TAB_DIR}")
