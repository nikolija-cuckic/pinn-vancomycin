# Struktura projekta: PINN-Vankomicin

```
pinn_vancomycin/
│
├── data/
│   ├── raw/
│   │   ├── vancomycin_synthetic.csv      # generisano iz data_processing.py
│   │   └── vancomycin_neonatal.csv       # opciono: Lee et al. 2021 MOESM4
│   └── processed/
│       ├── subject_profiles.csv          # jedan red po merenju, normalizovano
│       ├── ground_truth_params.csv       # k10, Vd, CL po subjektu
│       └── neonatal_profiles.csv         # opciono: preprocessirani neonatalni podaci
│
├── src/
│   ├── data_processing.py        # generisanje sintetickih profila (odrasli)
│   ├── data_processing_neo.py    # opciono: preprocessing neonatalnog dataseta
│   ├── benchmark.py              # curve_fit benchmark (1-odeljni i 2-odeljni)
│   ├── pinn_model.py             # definicija mreze, loss funkcija, trening
│   ├── inverse_problem.py        # wrapper: uzorkuj N tacaka, dodaj sum, proceni parametre
│   └── metrics.py                # relativna greska parametara, RMSE krive
│
├── experiments/
│   ├── 01_forward_validation.py      # validacija PINN forward problema vs. analiticko resenje
│   ├── 02_inverse_full_data.py       # inverzni problem na punom profilu (proof of concept)
│   ├── 03_sensitivity_analysis.py    # glavni eksperiment: N x sigma x 30 seed-ova
│   └── 04_neonatal_validation.py     # opciono: validacija na realnim neonatalnim podacima
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_visualization.ipynb
│
├── results/
│   ├── figures/                  # PNG grafikoni za rad
│   └── tables/                   # CSV rezultati eksperimenata
│
├── paper/
│   └── main.docx / main.tex
│
├── requirements.txt
└── README.md
```

---

## Sadrzaj kljucnih fajlova

### `requirements.txt`

```
torch>=2.0
deepxde>=1.10
scipy
numpy
pandas
matplotlib
seaborn
```

---

### `src/data_processing.py`

```python
# Odgovornosti:
# - Generise sinteticke PK profile na osnovu populacionih parametara (Aljutayli et al. 2020)
# - CL ~ LogNormal(3.5 L/h, CV=30%), Vd ~ LogNormal(50 L, CV=25%)
# - 50 virtualnih pacijenata, 12 tacaka merenja (0.5-24 h)
# - Normalizuje vreme na [0, 1] i koncentraciju na [0, 1] po subjektu
# - Verifikuje generisanje curve_fit-om na punom profilu (greska treba biti <1%)
# - Cuva:
#     data/raw/vancomycin_synthetic.csv
#     data/processed/subject_profiles.csv
#     data/processed/ground_truth_params.csv

def generate_synthetic_profiles(n_subjects, seed) -> tuple: ...
def fit_ground_truth_check(profiles_df, gt_params_df) -> pd.DataFrame: ...
def normalize_profiles(profiles_df) -> pd.DataFrame: ...
```

---

### `src/pinn_model.py`

```python
# Odgovornosti:
# - Definisanje neuronske mreze (FNN, tanh aktivacija, 3-4 sloja, 32-64 neurona)
# - Trenabilni parametri: log_k10, log_Vd (log-prostor garantuje pozitivnost)
# - Loss funkcija: L_data + lambda * L_physics
# - Physics loss: rezidual dC/dt + k10 * C na kolokacionim tackama (autograd)
# - Trening petlja: Adam -> L-BFGS
# - Podrzava jednoodeljni i dvoodeljni model (parametar model_type)

class PKNet(nn.Module):
    def __init__(self, hidden_layers=3, hidden_size=32): ...

class PINN:
    def __init__(self, net: PKNet, n_collocation=1000, model_type='one'): ...
    def physics_loss(self, t_col): ...
    def data_loss(self, t_data, C_data): ...
    def train(self, t_data, C_data, epochs=5000): ...
    def get_parameters(self) -> dict: ...   # vraca k10, Vd
```

---

### `src/benchmark.py`

```python
# Odgovornosti:
# - Klasicni ODE fit kao benchmark za poredenje sa PINN-om
# - Oba modela su obavezna za eksperiment

def fit_one_compartment(t_data, C_data) -> dict:
    # scipy.optimize.curve_fit na C(t) = C0 * exp(-k10 * t)

def fit_two_compartment(t_data, C_data) -> dict:
    # solve_ivp za dvoodeljni sistem + minimize(residual)
```

---

### `src/inverse_problem.py`

```python
# Odgovornosti:
# - Uzorkuje N tacaka iz punog profila (random ili uniformno)
# - Dodaje Gaussov sum: C_noisy = C_true * (1 + N(0, sigma))
# - Poziva PINN i benchmark, vraca procenjene parametre i greske

def subsample(profile, N, sigma, seed) -> tuple: ...
def run_inverse(t_sparse, C_noisy, gt_params, model_type='two') -> dict: ...
```

---

### `src/metrics.py`

```python
# Odgovornosti:
# - Relativna greska parametara
# - RMSE rekonstrukcije krive na test tackama

def relative_error(estimated, true_val) -> float: ...
def curve_rmse(model, t_test, C_test) -> float: ...
```

---

### `experiments/03_sensitivity_analysis.py`

```python
# Centralni eksperiment rada
# Petlja: za svaki N i sigma, 30 seed-ova
# Rezultat: results/tables/sensitivity_results.csv

N_values     = [3, 5, 8, 10, 12]        # za sinteticki dataset
sigma_values = [0.0, 0.05, 0.10, 0.20]
n_seeds      = 30

results = []
for N in N_values:
    for sigma in sigma_values:
        for seed in range(n_seeds):
            t_sparse, C_noisy = subsample(full_profile, N, sigma, seed)

            pinn_params  = PINN(...).train(t_sparse, C_noisy).get_parameters()
            bench_params = fit_two_compartment(t_sparse, C_noisy)

            results.append({
                "N": N, "sigma": sigma, "seed": seed,
                "pinn_err_k10":  relative_error(pinn_params["k10"],  gt["k10"]),
                "bench_err_k10": relative_error(bench_params["k10"], gt["k10"]),
            })

pd.DataFrame(results).to_csv("results/tables/sensitivity_results.csv")

# Za neonatalni dataset (04_neonatal_validation.py):
# N_values = [2, 3, 4, 5]
```

---

## Redosled pokretanja

```
1. python src/data_processing.py               # generiši sinteticke podatke
2. python experiments/01_forward_validation.py  # verifikuj PINN forward problem
3. python experiments/02_inverse_full_data.py   # proof of concept inverznog problema
4. python experiments/03_sensitivity_analysis.py # glavni eksperiment
5. [opciono] python experiments/04_neonatal_validation.py
6. jupyter notebook notebooks/02_results_visualization.ipynb
```

---

## Napomene o dizajnu

### Sinteticki podaci kao primarni izbor

PK-DB ne sadrzi vankomicin podatke sa dovoljno merenja po subjektu. Sinteticki podaci generisani na osnovu Aljutayli et al. (2020) su standardan pristup u PINN PK radovima i daju potpunu kontrolu nad ground truth vrednostima.

### Dvoodeljni model kao primarni

Jednoodeljni model ima analiticko resenje, sto znaci da curve_fit radi gotovo savrseno cak i sa malo podataka. Dvoodeljni model nema zatvorenu formu i numericka nestabilnost je realna — tu PINN regularizacija ima veci potencijal da pokaze prednost.

### src/ vs experiments/

`src/` sadrzi reusable kod (model, benchmark, metrike) koji se ne menja izmedju eksperimenata. `experiments/` su skripte koje se pokrecu jednom i proizvode rezultate. Ako nesto krene naopako u eksperimentu, ne diras model — greska je izolovana u eksperimentalnoj skripti.

### 30 seed-ova po kombinaciji

Bez ovoga, jedan los run (losa inicijalizacija, nesrecan uzorak merenja) moze da izgleda kao da je cela metoda losa. Sa 30 seed-ova imas srednju vrednost i standardnu devijaciju greske — to je minimum za kredibilan eksperiment.

### results/ odvojen od koda

Cuvas CSV sa svim rezultatima posebno od koda koji ih generise. Prednost: ne moraš ponovo pokretati dugacke eksperimente svaki put kada prilagodavas grafikon ili formatiranje tabele za rad.

### log_k10 i log_Vd umesto direktnih vrednosti

Ako PINN tokom optimizacije nauci negativnu vrednost za k10, eliminacija bi isla u pogresnom smeru. Treniranjem u log-prostoru (k10 = exp(log_k10)) garantujes da parametri ostanu pozitivni tokom celog treninga bez eksplicitnih ogranicenja ili clippinga.
