# Plan rada: PINN za procenu farmakokinetičkih parametara vankomicina iz retkih kliničkih merenja

**Konferencija:** AAI2026 — 5. srpska međunarodna konferencija o primenjenoj veštačkoj inteligenciji
**Datum konferencije:** 20–21. maj 2026, Kragujevac
**Rok za predaju:** 20. april 2026
**Format:** 6+ strana, DOCX, Springer zbornik

---

## 1. Motivacija

Vankomicin je antibiotik koji se koristi za lecenje teskih bakterijskih infekcija (MRSA, endokarditis, sepsa). Ima uzak terapijski prozor — nedovoljna koncentracija znaci neuspesno lecenje, a predoziranje uzrokuje nefrotoksicnost (ostecenje bubrega). Zbog toga je Therapeutic Drug Monitoring (TDM) standardna praksa: prate se koncentracije leka u krvi i na osnovu njih se prilagodava doza.

**Problem:** U klinickoj praksi lekar ima na raspolaganju samo 3-8 uzoraka krvi po pacijentu (eticka i prakticna ogranicenja). Klasicni metodi fitovanja (npr. `scipy.optimize.curve_fit`, NONMEM) imaju nestabilne procene parametara kada je merenja malo i kada su zasumljena.

**Predlog resenja:** Physics-Informed Neural Networks (PINNs) ugraduju diferencijalne jednacine direktno u loss funkciju, cime fizicki zakon sluzi kao regularizacija. Hipoteza je da ova regularizacija omogucava pouzdaniju procenu parametara pri malom broju merenja.

**Istrazivacko pitanje:**
> *Pri kom minimalnom broju merenja i maksimalnom nivou suma PINN inverzni problem daje pouzdanije procene farmakokinetičkih parametara vankomicina od klasicnog nelinearnog fitovanja?*

---

## 2. Teorijska osnova

### 2.1 Jednoodeljni farmakokineticki model

Jednoodeljni model pretpostavlja ravnomernu distribuciju leka u organizmu. Za intravensku primenu, dinamika koncentracije opisana je ODJ prvog reda:

```
dC/dt = -k10 * C,   C(0) = D / Vd
```

gde su:
- C(t) — koncentracija leka u plazmi [mg/L]
- k10  — stopa eliminacije [h^-1]
- D    — primenjena doza [mg]
- Vd   — volumen distribucije [L]

Analiticko resenje: C(t) = C0 * exp(-k10 * t)

Klinicki relevantni parametri koji se izvode:
- Poluzivot:  t_1/2 = ln(2) / k10
- Klirens:    CL = k10 * Vd
- AUC:        AUC = C0 / k10

### 2.2 Dvoodeljni farmakokineticki model

Dvoodeljni model uvodi centralni odeljak (krv) i periferni odeljak (tkiva):

```
dC1/dt = -k10*C1 - k12*C1 + k21*C2 + D(t)/V1
dC2/dt =  k12*C1 - k21*C2
```

Samo C1(t) je direktno merljivo (iz uzorka krvi). Dvoodeljni model je primarni model u ovom radu jer realisticnije opisuje distribuciju vankomicina u tkiva.

### 2.3 Physics-Informed Neural Networks (PINNs)

PINN aproksimira resenje ODJ neuronom mrezom C_hat(t; theta) minimizacijom kompozitnog gubitka:

```
L(theta) = L_data + lambda * L_physics
```

Data loss — greska na izmerenim tackama:
```
L_data = (1/N) * sum_i ( C_hat(t_i) - C_i_meas )^2
```

Physics loss — rezidual diferencijalne jednacine na kolokacionim tackama:
```
L_physics = (1/Nc) * sum_j ( dC_hat/dt(t_j) + k10 * C_hat(t_j) )^2
```

Gradijent dC_hat/dt racuna se automatskim diferenciranjem (autograd). Parametri k10 i Vd tretiraju se kao trenabilni skalari uz parametre mreze theta — to je inverzni problem.

### 2.4 Benchmark metod: scipy.optimize.curve_fit

Nelinearni least-squares fit koji minimizuje:
```
sum_i ( C_i_meas - C0 * exp(-k10 * t_i) )^2
```
Nema fizicku regularizaciju — oslanja se iskljucivo na podatke.

---

## 3. Podaci

### 3.1 Primarni dataset: sinteticki virtualni pacijenti (odrasli)

Sinteticki profili generisu se na osnovu populacionih PK parametara vankomicina iz literature (Aljutayli et al., 2020):

- CL ~ LogNormal(mean=3.5 L/h, CV=30%)
- Vd ~ LogNormal(mean=50 L, CV=25%)
- k10 = CL / Vd
- Doza D = 1000 mg IV bolus
- 50 virtualnih pacijenata, 12 tacaka merenja po profilu (0.5-24 h)

Prednosti sintetickog pristupa: poznat ground truth po subjektu, kontrolisana varijabilnost, reproduktivnost eksperimenta.

### 3.2 Opcioni dataset: neonatalni TDM podaci (Lee et al., 2021)

Ukoliko rezultati na sintetickim podacima budu pozitivni i ostane vremena, koristice se realni TDM podaci iz rada:

Lee et al. (2021). Population pharmacokinetics and dose optimization of vancomycin in neonates. Scientific Reports. DOI: 10.1038/s41598-021-85529-3

Dataset: 900 merenja vankomicina kod 207 novorodjencadi, dostupan kao Supplementary Dataset 4 (CSV, CC BY 4.0). Za neonatalnu populaciju eksperimentalni dizajn se prilagodava: N u {2, 3, 4, 5} merenja po pacijentu, sto odgovara realnim klinickim ogranicenjima.

### 3.3 Preprocessing

- Normalizacija vremenske ose na [0, 1]
- Normalizacija koncentracije na [0, 1] po subjektu
- Podela: pun profil -> ground truth parametri (fitovanje na svim tackama), proredeni profil -> ulaz za eksperiment

---

## 4. Eksperimentalni dizajn

### 4.1 Forward problem (validacija)

Pre inverznog problema, verifikuj da PINN moze da resi jednacinu kada su parametri poznati:
- Zadaj k10 = 0.1 h^-1, Vd = 50 L
- PINN treba da rekonstruise C(t) bez merenja, samo sa physics lossom
- Poredi sa analitickim resenjem

### 4.2 Inverzni problem — centralni eksperiment

Za svaku kombinaciju (N, sigma) gde je:
- N     u {3, 5, 8, 10, 12}        — broj merenja
- sigma u {0%, 5%, 10%, 20%}       — nivo Gaussovog suma (% od vrednosti)

Postupak (30 ponavljanja po kombinaciji sa razlicitim seed-ovima):

1. Uzorkuj N tacaka iz sintetickog profila
2. Dodaj Gaussov sum: C_i_noisy = C_i * (1 + N(0, sigma))
3. Proceni parametre sa PINN-om
4. Proceni parametre sa curve_fit-om
5. Izracunaj relativnu gresku u odnosu na ground truth

Metrika:
```
eps_k10 = |k10_hat - k10_star| / k10_star * 100%
```

Centralni rezultat: 2D grafikon greska vs. N (po nivoima suma) sa krivama PINN i curve_fit.

### 4.3 Rekonstrukcija krive

Pored greske parametara, evaluiraj i kvalitet rekonstruisane krive na test tackama (koje nisu bile u treningu):

```
RMSE = sqrt( (1/M) * sum_j ( C_hat(t_j) - C_j_true )^2 )
```

### 4.4 Opciona validacija na neonatalnom datasetu

Ako rezultati u sekciji 4.2 budu pozitivni, primeni isti PINN pristup na realne neonatalne TDM podatke. Ground truth parametri se procenjuju fitovanjem na punom profilu pacijenata sa 5+ merenja. Eksperimentalni dizajn se prilagodava: N u {2, 3, 4, 5}.

---

## 5. Implementacija

Stack:
- Python 3.10+
- PyTorch (autograd za physics loss)
- DeepXDE (opciono, za brzi razvoj PINN-a)
- scipy (curve_fit benchmark)
- pandas, matplotlib, seaborn

Arhitektura PINN-a:
- Ulaz:  t u R
- Izlaz: C_hat(t) u R
- Skriveni slojevi: 3-4 sloja, 32-64 neurona, aktivacija tanh
- Trenabilni parametri: log_k10, log_Vd (log-prostor za pozitivnost)
- Optimizer: Adam (lr=1e-3), zatim L-BFGS za fine-tuning
- Kolokacione tacke: 1000 uniformnih tacaka u [0, T]
- Podrzava jednoodeljni i dvoodeljni model (parametar model_type)

Primer loss funkcije (pseudokod):

```python
def loss(model, t_data, C_data, t_col, lambda_phys=1.0):
    C_pred = model(t_data)
    L_data = mse(C_pred, C_data)

    t_col.requires_grad_(True)
    C_col = model(t_col)
    dC_dt = autograd.grad(C_col.sum(), t_col, create_graph=True)[0]
    k10 = torch.exp(model.log_k10)
    residual = dC_dt + k10 * C_col
    L_phys = (residual**2).mean()

    return L_data + lambda_phys * L_phys
```

---

## 6. Struktura rada

1. Introduction (~0.5 str.)       — TDM vankomicina, problem retkih merenja, motivacija za PINNs
2. Mathematical Model (~1 str.)   — jednoodeljni i dvoodeljni PK model, PINN formulacija, loss funkcija
3. Data and Methods (~1 str.)     — sinteticki dataset, eksperimentalni dizajn, benchmark metod
4. Results (~2 str.)              — forward validacija, greska parametara vs. N i sigma, RMSE rekonstrukcije, tabele i grafikoni; opciono: rezultati na neonatalnom datasetu
5. Discussion (~0.5 str.)         — kada koristiti PINN vs. curve_fit, klinicka interpretacija, ogranicenja
6. Conclusion and Future Work (~0.5 str.) — populacioni PINN, vise pacijenata, bayesijanski pristup

---

## 7. Reference (preliminarne)

1. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Daryakenari, N.A. et al. (2024). CMINNs: Compartment Model Informed Neural Networks — Unlocking Drug Dynamics. arXiv:2409.12998.

3. Aljutayli, A., Marsot, A., Nekka, F. (2020). An Update on Population Pharmacokinetic Analyses of Vancomycin, Part I: In Adults. Clinical Pharmacokinetics, 59(6), 671-698.

4. Lee, C.R. et al. (2021). Population pharmacokinetics and dose optimization of vancomycin in neonates. Scientific Reports, 11, 6497.

5. Marino, I.P. et al. (2026). A physics-informed neural network approach for estimating pharmacokinetic parameters. PMC12909361.

6. Lu, L. et al. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1), 208-228.

7. Ivanovic, M. et al. (2024). Treatment of Non-Physical Solutions of the Oxygen Diffusion in Soil by Physics-Informed Neural Network. AAI2024 Proceedings.

---

## 8. Vremenski plan

| Dani  | Zadatak                                                                         |
|-------|---------------------------------------------------------------------------------|
| 1     | Finalizacija parametara, pokretanje data_processing.py, verifikacija            |
| 2-4   | Implementacija PINN forward problema, validacija na analitickom resenju          |
| 5-10  | Implementacija inverznog problema, jednoodeljni + dvoodeljni model, debug        |
| 11-16 | Sistematski eksperiment: petlja po N i sigma, 30 seed-ova, curve_fit benchmark   |
| 17-18 | Opciono: neonatalni dataset (samo ako rezultati na sintetickim budu dobri)       |
| 19-21 | Vizualizacija rezultata (grafikoni, tabele)                                      |
| 22-24 | Pisanje rada u LaTeX-u / DOCX-u                                                  |
| 25    | Korektura, formatiranje, predaja                                                 |
