"""
Map the RESURF design point across oxide thickness:
    V_bd,max(T_ox)   and   N_D,opt(T_ox)
for both breakdown criteria (Ec, ionization integral).

For each T_ox a dense N_D sweep is run and the peak is parabolic-refined
(same approach as vbd_zoom.py). Reuses the solver in vbd_vs_nd.py.
"""

import numpy as np
import vbd_vs_nd as m
from vbd_zoom import peak          # parabolic peak refiner

Tox_list = np.linspace(10e-9, 300e-9, 16)     # 10 .. 300 nm
Nd_grid  = np.logspace(np.log10(3e15), np.log10(1e17), 55)   # cm^-3

def curve(nx=120, nyv=60, margin=0.06):
    out = {"Ec_V": [], "Ec_Nd": [], "II_V": [], "II_Nd": []}
    for Tox in Tox_list:
        vEc, vII = [], []
        for Nd in Nd_grid:
            x, y, F = m.field_components(Nd, Tox, nx=nx, nyv=nyv)
            X = np.meshgrid(x, y, indexing="ij")[0]
            mask = m.corner_mask(X, drain_margin=margin)
            vEc.append(m.Vbd_Ec(F, m.Ec_of_Nd(Nd), mask))
            vII.append(m.Vbd_II(F, x))
        nEc, VEc = peak(Nd_grid, vEc)
        nII, VII = peak(Nd_grid, vII)
        out["Ec_V"].append(VEc); out["Ec_Nd"].append(nEc)
        out["II_V"].append(VII); out["II_Nd"].append(nII)
        print(f"Tox={Tox*1e9:5.1f} nm :  Ec {VEc:5.1f}V@{nEc:.2e}   II {VII:5.1f}V@{nII:.2e}")
    return {k: np.array(v) for k, v in out.items()}

if __name__ == "__main__":
    d = curve()
    tnm = Tox_list * 1e9

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    ax[0].plot(tnm, d["Ec_V"], "o-", color="#1f77b4", label="Ec method")
    ax[0].plot(tnm, d["II_V"], "s-", color="#d62728", label="Integral method")
    ax[0].set_xlabel("T_ox (nm)"); ax[0].set_ylabel("V_bd,max (V)")
    ax[0].set_title("Peak breakdown voltage vs oxide thickness")
    ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].semilogy(tnm, d["Ec_Nd"], "o-", color="#1f77b4", label="Ec method")
    ax[1].semilogy(tnm, d["II_Nd"], "s-", color="#d62728", label="Integral method")
    ax[1].set_xlabel("T_ox (nm)"); ax[1].set_ylabel("N_D,opt (cm$^{-3}$)")
    ax[1].set_title("Optimal doping vs oxide thickness")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend()

    fig.suptitle("RESURF design map  (L=2 um, a=0.5 um)")
    fig.tight_layout(); fig.savefig("map_tox.png", dpi=150)
    print("saved map_tox.png")

    # linear-fit the near-linear Vbd,max(Tox) trend
    for meth in ("Ec", "II"):
        p = np.polyfit(tnm, d[meth+"_V"], 1)
        print(f"{meth}: Vbd,max ~ {p[0]:.3f}*Tox[nm] + {p[1]:.2f}  V")
