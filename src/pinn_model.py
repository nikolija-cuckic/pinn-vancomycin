"""
pinn_model.py
-------------
Physics-Informed Neural Networks za procenu farmakokinetičkih parametara
vankomicina — inverzni problem.

Klase:
  PKNet              — zajednička FNN arhitektura (tanh, 3-4 sloja)
  OneCompartmentPINN — 1-odeljni model, 2 parametra (k10, Vd)
  TwoCompartmentPINN — 2-odeljni model, 4 parametra (k10, k12, k21, V1)

Trening:
  Adam (brza konvergencija) → L-BFGS (fine-tuning)
  Physics loss na kolokacionim tačkama (autograd)
  Parametri u log-prostoru (garantuje pozitivnost)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# ── Arhitektura mreže ─────────────────────────────────────────────────────────

class PKNet(nn.Module):
    """
    Fully-connected mreža: t → C(t) ili [C1(t), C2(t)].
    Ulaz: 1D vreme (normalizovano na [0,1])
    Izlaz: n_outputs neurona (1 za 1-comp, 2 za 2-comp)
    """

    def __init__(
        self,
        hidden_layers: int = 3,
        hidden_size:   int = 64,
        n_outputs:     int = 1,
    ):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, n_outputs)]
        self.net = nn.Sequential(*layers)

        # Xavier inicijalizacija
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ── Pomoćna funkcija ──────────────────────────────────────────────────────────

def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """dOutput/dInp korišćenjem autograd-a."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
    )[0]


# ── 1-odeljni PINN ────────────────────────────────────────────────────────────

class OneCompartmentPINN(nn.Module):
    """
    Inverzni problem za jednoodeljni IV model:
        dC/dt = -k10 * C,   C(0) = D / Vd

    Trenabilni parametri (log-prostor → uvek pozitivni):
        log_k10, log_Vd

    Loss:
        L = L_data + lambda_phys * L_phys + lambda_ic * L_ic
    """

    def __init__(
        self,
        dose_mg:       float = 1000.0,
        hidden_layers: int   = 3,
        hidden_size:   int   = 64,
        lambda_phys:   float = 1.0,
        lambda_ic:     float = 10.0,
        n_collocation: int   = 1000,
        t_max_h:       float = 24.0,
        device:        str   = "cpu",
    ):
        super().__init__()
        self.dose       = dose_mg
        self.lam_phys   = lambda_phys
        self.lam_ic     = lambda_ic
        self.n_col      = n_collocation
        self.t_max      = t_max_h
        self.device     = torch.device(device)

        self.net = PKNet(hidden_layers, hidden_size, n_outputs=1)

        # Parametri u log-prostoru; inicijalizacija na tipičnim vrednostima
        self.log_k10 = nn.Parameter(torch.tensor([np.log(0.08)]))  # k10 ~ 0.08 1/h
        self.log_Vd  = nn.Parameter(torch.tensor([np.log(50.0)]))  # Vd  ~ 50 L

        self.to(self.device)

    @property
    def k10(self) -> torch.Tensor:
        return torch.exp(self.log_k10)

    @property
    def Vd(self) -> torch.Tensor:
        return torch.exp(self.log_Vd)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Predviđa normalizovanu koncentraciju za normalizovano vreme."""
        return self.net(t_norm)

    def physics_loss(self, t_norm: torch.Tensor, t_max_h: float) -> torch.Tensor:
        """
        Rezidual ODJ na kolokacionim tačkama.
        t_norm ∈ [0,1] — normalizovano vreme
        t_max_h        — faktor skaliranja za dC/dt
        """
        t_norm = t_norm.detach().requires_grad_(True)  # svež list, grad se ne akumulira na t_col
        C_norm = self.net(t_norm)
        dC_dtnorm = _grad(C_norm, t_norm)

        # dC/dt u originalnim jedinicama = dC_norm/dt_norm * (C_max / t_max)
        # Ali u loss-u radimo u normalizovanom prostoru:
        # dC_norm/dt_norm = -k10 * t_max * C_norm  (chain rule)
        residual = dC_dtnorm + self.k10 * t_max_h * C_norm
        return (residual ** 2).mean()

    def ic_loss(self, C_max: float) -> torch.Tensor:
        """
        Početni uslov: C(0) = D/Vd.
        U normalizovanom prostoru: C_norm(0) = (D/Vd) / C_max.
        """
        t0    = torch.zeros(1, 1, device=self.device)
        C_pred = self.net(t0)
        # (self.dose / self.Vd) čuva Vd u computation graph → gradijent teče ka log_Vd
        C0_norm_expected = (self.dose / self.Vd) / C_max
        return (C_pred - C0_norm_expected) ** 2

    def data_loss(
        self,
        t_norm: torch.Tensor,
        C_norm: torch.Tensor,
    ) -> torch.Tensor:
        C_pred = self.net(t_norm)
        return ((C_pred - C_norm) ** 2).mean()

    def compute_loss(
        self,
        t_data_norm: torch.Tensor,
        C_data_norm: torch.Tensor,
        t_col_norm:  torch.Tensor,
        t_max_h:     float,
        C_max:       float,
    ) -> tuple:
        L_data = self.data_loss(t_data_norm, C_data_norm)
        L_phys = self.physics_loss(t_col_norm, t_max_h)
        L_ic   = self.ic_loss(C_max).squeeze()
        total  = L_data + self.lam_phys * L_phys + self.lam_ic * L_ic
        return total, L_data, L_phys, L_ic

    def get_parameters(self) -> dict:
        return {
            "k10": self.k10.item(),
            "Vd":  self.Vd.item(),
            "CL":  (self.k10 * self.Vd).item(),
        }


# ── 2-odeljni PINN ────────────────────────────────────────────────────────────

class TwoCompartmentPINN(nn.Module):
    """
    Inverzni problem za dvoodeljni IV model:
        dC1/dt = -(k10+k12)*C1 + k21*C2
        dC2/dt =  k12*C1 - k21*C2
        C1(0) = D/V1,  C2(0) = 0

    Mreža ima 2 izlaza: [C1_norm, C2_norm].
    Data loss samo na C1 (C2 nije merljivo!).
    Physics loss na oba izlaza.

    Trenabilni parametri (log-prostor):
        log_k10, log_k12, log_k21, log_V1
    """

    def __init__(
        self,
        dose_mg:       float = 1000.0,
        hidden_layers: int   = 4,
        hidden_size:   int   = 64,
        lambda_phys:   float = 1.0,
        lambda_ic:     float = 10.0,
        n_collocation: int   = 1000,
        t_max_h:       float = 24.0,
        device:        str   = "cpu",
    ):
        super().__init__()
        self.dose     = dose_mg
        self.lam_phys = lambda_phys
        self.lam_ic   = lambda_ic
        self.n_col    = n_collocation
        self.t_max    = t_max_h
        self.device   = torch.device(device)

        self.net = PKNet(hidden_layers, hidden_size, n_outputs=2)

        # Inicijalizacija na populacionim sredinama
        self.log_k10 = nn.Parameter(torch.tensor([np.log(0.10)]))
        self.log_k12 = nn.Parameter(torch.tensor([np.log(0.20)]))
        self.log_k21 = nn.Parameter(torch.tensor([np.log(0.10)]))
        self.log_V1  = nn.Parameter(torch.tensor([np.log(20.0)]))

        self.to(self.device)

    @property
    def k10(self): return torch.exp(self.log_k10)
    @property
    def k12(self): return torch.exp(self.log_k12)
    @property
    def k21(self): return torch.exp(self.log_k21)
    @property
    def V1(self):  return torch.exp(self.log_V1)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        return self.net(t_norm)

    def physics_loss(self, t_norm: torch.Tensor, t_max_h: float) -> torch.Tensor:
        """
        Reziduali oba ODJ u normalizovanom prostoru.
        C1_norm, C2_norm normalizovani istim faktorom C1_max.
        """
        t_norm = t_norm.detach().requires_grad_(True)  # svež list, grad se ne akumulira na t_col
        out    = self.net(t_norm)              # (N, 2)
        C1_norm = out[:, 0:1]
        C2_norm = out[:, 1:2]

        dC1_dtnorm = _grad(C1_norm, t_norm)
        dC2_dtnorm = _grad(C2_norm, t_norm)

        # ODJ u normalizovanom prostoru (t_max skalira derivaciju):
        # dC1/dt = dC1_norm/dt_norm / t_max * C1_max
        # Rezidual u normalizovanim jedinicama (delimo sa C1_max → ostaje normalizovano)
        r1 = dC1_dtnorm + (self.k10 + self.k12) * t_max_h * C1_norm \
             - self.k21 * t_max_h * C2_norm
        r2 = dC2_dtnorm - self.k12 * t_max_h * C1_norm \
             + self.k21 * t_max_h * C2_norm

        return (r1 ** 2).mean() + (r2 ** 2).mean()

    def ic_loss(self, C1_max: float) -> torch.Tensor:
        """
        IC: C1(0) = D/V1 (normalizovano),  C2(0) = 0.
        """
        t0  = torch.zeros(1, 1, device=self.device)
        out = self.net(t0)
        C1_pred_norm = out[0, 0]
        C2_pred_norm = out[0, 1]

        C1_ic_norm = (self.dose / self.V1) / C1_max
        ic1 = (C1_pred_norm - C1_ic_norm) ** 2
        ic2 = C2_pred_norm ** 2
        return ic1 + ic2

    def data_loss(
        self,
        t_norm:  torch.Tensor,
        C1_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Data loss samo na C1 — C2 nije merljivo."""
        out    = self.net(t_norm)
        C1_pred = out[:, 0:1]
        return ((C1_pred - C1_norm) ** 2).mean()

    def compute_loss(
        self,
        t_data_norm:  torch.Tensor,
        C1_data_norm: torch.Tensor,
        t_col_norm:   torch.Tensor,
        t_max_h:      float,
        C1_max:       float,
    ) -> tuple:
        L_data = self.data_loss(t_data_norm, C1_data_norm)
        L_phys = self.physics_loss(t_col_norm, t_max_h)
        L_ic   = self.ic_loss(C1_max).squeeze()
        total  = L_data + self.lam_phys * L_phys + self.lam_ic * L_ic
        return total, L_data, L_phys, L_ic

    def get_parameters(self) -> dict:
        return {
            "k10": self.k10.item(),
            "k12": self.k12.item(),
            "k21": self.k21.item(),
            "V1":  self.V1.item(),
            "CL":  (self.k10 * self.V1).item(),
            "Vd_ss": (self.V1 * (1 + self.k12 / self.k21)).item(),
        }


# ── Trening funkcija ──────────────────────────────────────────────────────────

def train_pinn(
    model:        nn.Module,
    t_data_norm:  np.ndarray,
    C_data_norm:  np.ndarray,
    t_max_h:      float,
    C_max:        float,
    epochs_adam:  int   = 5000,
    epochs_lbfgs: int   = 500,
    lr_adam:      float = 1e-3,
    verbose:      bool  = False,
) -> list:
    """
    Trenira PINN (OneCompartmentPINN ili TwoCompartmentPINN).

    Vraća listu rečnika po eposi (Adam faza):
        [{"total": float, "data": float, "phys": float, "ic": float}, ...]
    L-BFGS faza koristi closure pattern koji PyTorch zahteva.
    """
    device = model.device

    # Konvertuj ulaze u tenzore
    t_tn = torch.tensor(t_data_norm, dtype=torch.float32, device=device).reshape(-1, 1)
    C_tn = torch.tensor(C_data_norm, dtype=torch.float32, device=device).reshape(-1, 1)

    # Kolokacione tačke — uniformne u [0, 1]
    t_col = torch.linspace(0, 1, model.n_col, device=device).reshape(-1, 1)

    # ── Adam faza ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    history   = []

    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        total, L_data, L_phys, L_ic = model.compute_loss(
            t_tn, C_tn, t_col, t_max_h, C_max
        )
        total.backward()
        optimizer.step()
        history.append({
            "total": total.item(),
            "data":  L_data.item(),
            "phys":  L_phys.item(),
            "ic":    L_ic.item(),
        })

        if verbose and epoch % 500 == 0:
            print(f"  Adam {epoch:5d} | total={total.item():.4e} "
                  f"data={L_data.item():.4e} phys={L_phys.item():.4e} "
                  f"ic={L_ic.item():.4e}")

    # ── L-BFGS faza ───────────────────────────────────────────────────────────
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=epochs_lbfgs,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        total, _, _, _ = model.compute_loss(t_tn, C_tn, t_col, t_max_h, C_max)
        total.backward()
        return total

    lbfgs.step(closure)

    if verbose:
        final, *_ = model.compute_loss(t_tn, C_tn, t_col, t_max_h, C_max)
        print(f"  L-BFGS final loss: {final.item():.4e}")

    return history


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

    print("Smoke test: 1-odeljni PINN")
    # Sinteticki profil: k10=0.08, Vd=50, D=1000
    k10_true, Vd_true, dose = 0.08, 50.0, 1000.0
    t_h    = np.array([0.5,1,2,3,4,6,8,10,12,16,20,24], dtype=np.float32)
    C_true = (dose/Vd_true) * np.exp(-k10_true * t_h)

    t_max_h = 24.0;  C_max = C_true.max()
    t_norm  = t_h / t_max_h
    C_norm  = C_true / C_max

    model1 = OneCompartmentPINN(dose_mg=dose)
    train_pinn(model1, t_norm, C_norm, t_max_h, C_max,
               epochs_adam=3000, epochs_lbfgs=200, verbose=True)

    params = model1.get_parameters()
    err_k10 = abs(params["k10"] - k10_true) / k10_true * 100
    err_Vd  = abs(params["Vd"]  - Vd_true)  / Vd_true  * 100
    print(f"  k10: procena={params['k10']:.5f}, istina={k10_true}, greska={err_k10:.2f}%")
    print(f"  Vd:  procena={params['Vd']:.2f},  istina={Vd_true},  greska={err_Vd:.2f}%")

    print()
    print("Smoke test: 2-odeljni PINN")
    from data_processing import solve_two_compartment
    k10_t, k12_t, k21_t, V1_t = 0.10, 0.20, 0.10, 20.0
    C1_true, _ = solve_two_compartment(t_h, k10_t, k12_t, k21_t, V1_t, dose)
    C1_max = C1_true.max()
    C1_norm = C1_true / C1_max

    model2 = TwoCompartmentPINN(dose_mg=dose)
    train_pinn(model2, t_norm, C1_norm, t_max_h, C1_max,
               epochs_adam=3000, epochs_lbfgs=200, verbose=True)

    params2 = model2.get_parameters()
    for key, true_val in [("k10",k10_t),("k12",k12_t),("k21",k21_t),("V1",V1_t)]:
        err = abs(params2[key] - true_val) / true_val * 100
        print(f"  {key}: procena={params2[key]:.4f}, istina={true_val}, greska={err:.2f}%")
