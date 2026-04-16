"""
data_processing.py
------------------
Generisanje sintetičkih farmakokinetičkih profila vankomicina.

Opcija B — dva odvojena dataseta:
  1) Jednoodeljni model (1-comp): analitičko rešenje C(t) = C0*exp(-k10*t)
  2) Dvoodeljni model (2-comp): numerička integracija (solve_ivp)

Literatura:
  1-comp: Rybak et al. (2020): CL≈3.5 L/h, Vd≈50 L, CV(CL)=30%, CV(Vd)=25%
  2-comp: Vancomycin pop-PK (Matzke 1984, Boeckmann 1992):
          k10≈0.10 h⁻¹, k12≈0.20 h⁻¹, k21≈0.10 h⁻¹, V1≈20 L

Izlaz:
  data/processed/subject_profiles_1comp.csv
  data/processed/ground_truth_params_1comp.csv
  data/processed/subject_profiles_2comp.csv
  data/processed/ground_truth_params_2comp.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
from pathlib import Path

# ── Putanje ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Zajednički parametri ─────────────────────────────────────────────────────
DOSE_MG    = 1000.0   # mg IV bolusna doza
N_SUBJECTS = 50       # broj virtualnih pacijenata po modelu

# Tačke merenja (sati) — 12 tačaka, omogućava proređivanje na 3–10
OBS_TIMES = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0])

# ── 1-odeljni populacioni parametri (Rybak et al. 2020) ──────────────────────
POP1_CL_MEAN = 3.5    # L/h
POP1_CL_CV   = 0.30
POP1_VD_MEAN = 50.0   # L
POP1_VD_CV   = 0.25

# ── 2-odeljni populacioni parametri (Matzke 1984, Boeckmann 1992) ────────────
# k10 — eliminacija iz centralnog odeljka [1/h]
# k12 — distribucija centralni → periferni [1/h]
# k21 — redistribucija periferni → centralni [1/h]
# V1  — volumen centralnog odeljka [L]
POP2_K10_MEAN = 0.10;  POP2_K10_CV = 0.30
POP2_K12_MEAN = 0.20;  POP2_K12_CV = 0.35
POP2_K21_MEAN = 0.10;  POP2_K21_CV = 0.30
POP2_V1_MEAN  = 20.0;  POP2_V1_CV  = 0.25


# ── Modeli ───────────────────────────────────────────────────────────────────

def one_compartment(t, C0, k10):
    """Analitičko rešenje jednoodeljnog IV modela: C(t) = C0 * exp(-k10 * t)"""
    return C0 * np.exp(-k10 * t)


def two_compartment_ode(t, y, k10, k12, k21):
    """
    ODJ sistem za 2-odeljni IV model (bez unosa — bolus dat kao IC).
      y[0] = C1(t)  — koncentracija u centralnom odeljku
      y[1] = C2(t)  — koncentracija u perifernom odeljku
    dC1/dt = -(k10 + k12)*C1 + k21*C2
    dC2/dt =  k12*C1 - k21*C2
    """
    C1, C2 = y
    dC1 = -(k10 + k12) * C1 + k21 * C2
    dC2 =   k12 * C1   - k21 * C2
    return [dC1, dC2]


def solve_two_compartment(t_eval, k10, k12, k21, V1, dose=DOSE_MG):
    """
    Numerički integriše 2-odeljni model i vraća (C1, C2) u tačkama t_eval.
    Početni uslovi: C1(0) = dose/V1, C2(0) = 0.
    """
    C1_0 = dose / V1
    sol = solve_ivp(
        two_compartment_ode,
        t_span=(0.0, t_eval.max()),
        y0=[C1_0, 0.0],
        args=(k10, k12, k21),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8, atol=1e-10,
    )
    return sol.y[0], sol.y[1]   # C1, C2


def log_normal_sample(mean, cv, n_samples, rng):
    """
    Uzorkuje n_samples vrednosti iz log-normalne raspodele.
    mean — aritmetička sredina
    cv   — koeficijent varijacije (sigma_rel)
    """
    sigma = np.sqrt(np.log(1 + cv**2))
    mu    = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, n_samples)


# ── Generisanje podataka ──────────────────────────────────────────────────────

def generate_synthetic_profiles(
    n_subjects: int = N_SUBJECTS,
    seed: int = 42,
) -> tuple:
    """
    Generiše sintetičke PK profile i vraća:
      profiles_df   — jedan red po (subject, timepoint)
      gt_params_df  — ground truth parametri po subjektu
    """
    rng = np.random.default_rng(seed)

    CL_values  = log_normal_sample(POP1_CL_MEAN, POP1_CL_CV, n_subjects, rng)
    Vd_values  = log_normal_sample(POP1_VD_MEAN, POP1_VD_CV, n_subjects, rng)
    k10_values = CL_values / Vd_values
    C0_values  = DOSE_MG / Vd_values

    profile_rows = []
    gt_rows = []

    for i in range(n_subjects):
        subject_id = f"S{i+1:03d}"
        k10 = k10_values[i]
        Vd  = Vd_values[i]
        CL  = CL_values[i]
        C0  = C0_values[i]

        C_true = one_compartment(OBS_TIMES, C0, k10)

        for t, C in zip(OBS_TIMES, C_true):
            profile_rows.append({
                "subject_id": subject_id,
                "time_h":     t,
                "C_true":     C,
                "dose_mg":    DOSE_MG,
            })

        gt_rows.append({
            "subject_id":  subject_id,
            "k10":         k10,
            "Vd":          Vd,
            "CL":          CL,
            "C0":          C0,
            "half_life_h": np.log(2) / k10,
            "AUC":         C0 / k10,
        })

    profiles_df  = pd.DataFrame(profile_rows)
    gt_params_df = pd.DataFrame(gt_rows)
    return profiles_df, gt_params_df


def generate_synthetic_profiles_2comp(
    n_subjects: int = N_SUBJECTS,
    seed: int = 123,
) -> tuple:
    """
    Generiše sintetičke 2-odeljne PK profile i vraća:
      profiles_df  — jedan red po (subject, timepoint); sadrži C1_true i C2_true
      gt_params_df — ground truth parametri (k10, k12, k21, V1, CL, Vd_ss) po subjektu

    Napomena: samo C1 je "merljivo" — C2 je latentna promenljiva perifernog odeljka.
    """
    rng = np.random.default_rng(seed)

    k10_vals = log_normal_sample(POP2_K10_MEAN, POP2_K10_CV, n_subjects, rng)
    k12_vals = log_normal_sample(POP2_K12_MEAN, POP2_K12_CV, n_subjects, rng)
    k21_vals = log_normal_sample(POP2_K21_MEAN, POP2_K21_CV, n_subjects, rng)
    V1_vals  = log_normal_sample(POP2_V1_MEAN,  POP2_V1_CV,  n_subjects, rng)

    profile_rows = []
    gt_rows      = []

    for i in range(n_subjects):
        subject_id = f"T{i+1:03d}"   # prefiks T = two-compartment
        k10 = k10_vals[i]
        k12 = k12_vals[i]
        k21 = k21_vals[i]
        V1  = V1_vals[i]

        C1, C2 = solve_two_compartment(OBS_TIMES, k10, k12, k21, V1)

        for t, c1, c2 in zip(OBS_TIMES, C1, C2):
            profile_rows.append({
                "subject_id": subject_id,
                "time_h":     t,
                "C1_true":    c1,   # merljivo (centralni odeljak)
                "C2_true":    c2,   # nemerljivo (periferni odeljak)
                "dose_mg":    DOSE_MG,
            })

        # Klinički izvedeni parametri
        CL    = k10 * V1
        V2    = V1 * k12 / k21           # Vd perifernog odeljka
        Vd_ss = V1 + V2                  # Vd u stacionarnom stanju
        C1_0  = DOSE_MG / V1

        gt_rows.append({
            "subject_id": subject_id,
            "k10": k10, "k12": k12, "k21": k21, "V1": V1,
            "CL":  CL,  "V2":  V2,  "Vd_ss": Vd_ss,
            "C1_0": C1_0,
        })

    return pd.DataFrame(profile_rows), pd.DataFrame(gt_rows)


def normalize_profiles_2comp(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje vreme i C1 na [0,1] po subjektu. C2 normalizuje istim faktorom."""
    t_max   = OBS_TIMES.max()
    records = []
    for sid, grp in profiles_df.groupby("subject_id"):
        C1_max = grp["C1_true"].max()
        grp    = grp.copy()
        grp["time_norm"] = grp["time_h"]  / t_max
        grp["C1_norm"]   = grp["C1_true"] / C1_max
        grp["C2_norm"]   = grp["C2_true"] / C1_max   # isti faktor — čuva odnos
        grp["C1_max"]    = C1_max
        grp["t_max"]     = t_max
        records.append(grp)
    return pd.concat(records, ignore_index=True)


def fit_ground_truth_check(profiles_df: pd.DataFrame, gt_params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifikacija: curve_fit na punom profilu treba da reprodukuje ground truth k10.
    Dodaje kolone fit_k10, fit_Vd, fit_err_k10 u gt_params_df.
    """
    rows = []
    for _, gt_row in gt_params_df.iterrows():
        sid   = gt_row["subject_id"]
        subj  = profiles_df[profiles_df["subject_id"] == sid]
        t     = subj["time_h"].values
        C     = subj["C_true"].values

        try:
            (C0_fit, k10_fit), _ = curve_fit(
                one_compartment, t, C,
                p0=[C[0], 0.1],
                bounds=([0, 1e-4], [1e4, 10.0]),
                maxfev=5000,
            )
            Vd_fit = DOSE_MG / C0_fit
        except RuntimeError:
            k10_fit = np.nan
            Vd_fit  = np.nan

        row = gt_row.to_dict()
        row["fit_k10"]     = k10_fit
        row["fit_Vd"]      = Vd_fit
        row["fit_err_k10"] = abs(k10_fit - gt_row["k10"]) / gt_row["k10"] * 100
        rows.append(row)

    return pd.DataFrame(rows)


# ── Normalizacija ─────────────────────────────────────────────────────────────

def normalize_profiles(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje vreme na [0, 1] i koncentraciju na [0, 1] po subjektu.
    Čuva originalne vrednosti u kolonama time_h i C_true.
    """
    t_max   = OBS_TIMES.max()
    records = []
    for sid, grp in profiles_df.groupby("subject_id"):
        C_max       = grp["C_true"].max()
        grp         = grp.copy()
        grp["time_norm"] = grp["time_h"] / t_max
        grp["C_norm"]    = grp["C_true"] / C_max
        grp["C_max"]     = C_max
        grp["t_max"]     = t_max
        records.append(grp)
    return pd.concat(records, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1-odeljni model ───────────────────────────────────────────────────────
    print("=" * 55)
    print("1-ODELJNI MODEL")
    print("=" * 55)
    profiles_1, gt_1 = generate_synthetic_profiles()
    print(f"  Subjekata: {len(gt_1)},  merenja: {len(profiles_1)}")
    print(f"  k10  : [{gt_1['k10'].min():.4f}, {gt_1['k10'].max():.4f}] 1/h")
    print(f"  Vd   : [{gt_1['Vd'].min():.1f},  {gt_1['Vd'].max():.1f}] L")
    print(f"  t1/2 : [{gt_1['half_life_h'].min():.1f},  {gt_1['half_life_h'].max():.1f}] h")

    print("  Verifikacija curve_fit...")
    gt_1_checked = fit_ground_truth_check(profiles_1, gt_1)
    err = gt_1_checked["fit_err_k10"].mean()
    print(f"  Greska k10 (pun profil): {err:.4f}%")

    profiles_1_norm = normalize_profiles(profiles_1)

    profiles_1.to_csv(RAW_DIR  / "vancomycin_1comp_raw.csv",          index=False)
    profiles_1_norm.to_csv(PROC_DIR / "subject_profiles_1comp.csv",   index=False)
    gt_1_checked.to_csv(PROC_DIR   / "ground_truth_params_1comp.csv", index=False)

    # ── 2-odeljni model ───────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("2-ODELJNI MODEL")
    print("=" * 55)
    profiles_2, gt_2 = generate_synthetic_profiles_2comp()
    print(f"  Subjekata: {len(gt_2)},  merenja: {len(profiles_2)}")
    print(f"  k10  : [{gt_2['k10'].min():.4f}, {gt_2['k10'].max():.4f}] 1/h")
    print(f"  k12  : [{gt_2['k12'].min():.4f}, {gt_2['k12'].max():.4f}] 1/h")
    print(f"  k21  : [{gt_2['k21'].min():.4f}, {gt_2['k21'].max():.4f}] 1/h")
    print(f"  V1   : [{gt_2['V1'].min():.1f},  {gt_2['V1'].max():.1f}] L")
    print(f"  Vd_ss: [{gt_2['Vd_ss'].min():.1f}, {gt_2['Vd_ss'].max():.1f}] L")

    # Sanity check: C1 treba opadati dugorocno (eliminacija dominira)
    last_C1  = profiles_2.groupby("subject_id")["C1_true"].last()
    first_C1 = profiles_2.groupby("subject_id")["C1_true"].first()
    pct_declining = (last_C1 < first_C1).mean() * 100
    print(f"  Profili sa C1(24h) < C1(0.5h): {pct_declining:.0f}% (ocekivano ~100%)")

    profiles_2_norm = normalize_profiles_2comp(profiles_2)

    profiles_2.to_csv(RAW_DIR  / "vancomycin_2comp_raw.csv",          index=False)
    profiles_2_norm.to_csv(PROC_DIR / "subject_profiles_2comp.csv",   index=False)
    gt_2.to_csv(PROC_DIR       / "ground_truth_params_2comp.csv",     index=False)

    print()
    print("Sacuvano:")
    for p in [
        RAW_DIR  / "vancomycin_1comp_raw.csv",
        RAW_DIR  / "vancomycin_2comp_raw.csv",
        PROC_DIR / "subject_profiles_1comp.csv",
        PROC_DIR / "ground_truth_params_1comp.csv",
        PROC_DIR / "subject_profiles_2comp.csv",
        PROC_DIR / "ground_truth_params_2comp.csv",
    ]:
        print(f"  {p}")
    print("\nFaza 1 zavrsena (oba modela).")


if __name__ == "__main__":
    main()
