"""
Electrostatic bias sweep of the partial-depletion obstacle PDE (report Sec. 7-8).
Fixed geometry a=0.9 um, t_ox=0.4 um, L=3 um.  For several N_D, sweep the applied
drain bias V and extract depletion geometry from the free-boundary solution
(u = V - phi >= 0; neutral set Z = {u=0}, drain-connected; depleted D = {u>0}):

  Wcl  : centerline depletion depth      = (depleted cells on row y~0) * dx     -> L
  Wlat : lateral depletion width at x=L/2 = (depleted cells in column x=L/2)*dy  -> a
  frac : depleted silicon area fraction  = mean(depleted) = 1 - neutral fraction -> 1

Points above the avalanche voltage are electrostatic continuations, not a blocking state.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freeboundary as fb
from vbd_vdm import eps_s, eps_I, qe, um

a, tox, L = 0.9*um, 0.4*um, 3.0*um
b = (eps_s/eps_I)*tox
nx, ny = 150, 50
A, r, dx, dy = fb.build_operator(nx, ny, L, a, b)

Nd_list = [0.5e16, 1e16, 2e16, 5e16, 10e16]
labels  = ["0.5", "1", "2", "5", "10"]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
Vgrid   = np.linspace(0.0, 100.0, 81)

def metrics(Nd):
    k = qe*Nd/eps_s
    Wcl, Wlat, frac = [], [], []
    icut = nx//2                                  # x = L/2 column
    for V in Vgrid:
        u, Z = fb.solve_obstacle(A, V*r - k)
        D = (~Z).reshape(nx, ny)                  # depleted mask [i=x, j=y]
        Wcl.append(D[:, 0].sum() * dx / um)       # centerline depth (um)
        Wlat.append(D[icut, :].sum() * dy / um)   # lateral width at L/2 (um)
        frac.append(D.mean())
    return np.array(Wcl), np.array(Wlat), np.array(frac)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
for Nd, lab, col in zip(Nd_list, labels, colors):
    Wcl, Wlat, frac = metrics(Nd)
    ax[0].plot(Vgrid, Wcl,  color=col, lw=1.9, label=lab)
    ax[1].plot(Vgrid, Wlat, color=col, lw=1.9, label=lab)
    ax[2].plot(Vgrid, frac, color=col, lw=1.9, label=lab)
    print(f"Nd={lab}e16: Wcl(100V)={Wcl[-1]:.2f} um  Wlat(100V)={Wlat[-1]:.2f} um  "
          f"frac(100V)={frac[-1]:.2f}")

ax[0].axhline(L/um, ls=":", color="0.5"); ax[0].set_ylabel("Wcl (um)")
ax[0].set_title("Centerline depletion depth")
ax[1].axhline(a/um, ls=":", color="0.5"); ax[1].set_ylabel("Wlat (um)")
ax[1].set_title("Lateral width at x = L/2")
ax[2].set_ylabel("Area fraction"); ax[2].set_title("Depleted silicon area fraction")
for a_ in ax:
    a_.set_xlabel("Applied drain bias V (V)"); a_.grid(alpha=0.3); a_.set_xlim(0, 100)
ax[2].legend(title=r"Nd ($10^{16}$ cm$^{-3}$)", loc="lower right")
fig.suptitle(f"Partial-depletion obstacle PDE | a=0.9 um, tox=0.4 um, L=3 um | "
             f"Electrostatic bias sweep; points above avalanche are not a blocking-state prediction")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("biassweep.png", dpi=140)
print("saved biassweep.png")
