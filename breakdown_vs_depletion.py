"""
Breakdown vs full-depletion threshold, and the depletion state at breakdown.
Fixed a=0.9 um, tox=0.4 um, L=3 um, 300 K, grounded field plate.

Top panel:
  - Breakdown V_BD(Nd): moving-boundary obstacle PDE + vOdM ionization integral.
  - Full-depletion threshold V_FD(Nd): the field-reversal / admissibility floor
    Vstart of the fully-depleted analytic solution (below it a neutral region
    forms), linear in Nd.
  - star  : breakdown maximum.
  - dot   : onset of partial depletion at breakdown = crossing V_BD = V_FD.
  - shaded: Nd beyond onset -> device is only partially depleted at breakdown.
Bottom panel: at V = V_BD(Nd), centerline depleted depth / L and depleted-area
fraction (both = 1 while fully depleted, falling after onset).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vbd_vdm as vd
import freeboundary as fb
from vbd_vdm import eps_s, eps_I, qe, um

a, tox, L = 0.9*um, 0.4*um, 3.0*um
b = (eps_s/eps_I)*tox
nx, ny = 120, 40

# --- full-depletion threshold VFD = Vstart (analytic, fine grid) ---
g_an = vd.centerline(L, a, b)
Nd_fine = np.linspace(0.4e16, 3.8e16, 200)
VFD = np.array([vd.Vstart(qe*n/eps_s, g_an) for n in Nd_fine])

# --- breakdown + depletion state at breakdown (moving boundary) ---
Nd = np.linspace(0.4e16, 3.8e16, 42)
Vbd, depth_frac, area_frac = [], [], []
for n in Nd:
    Vb, nf, (u, Z, dx, dy) = fb.Vbd_free(n, nx, ny, L, a, b)
    Vbd.append(Vb)
    D = (~Z).reshape(nx, ny)
    depth_frac.append(D[:, 0].sum()*dx / L)       # centerline depleted depth / L
    area_frac.append(1.0 - nf)                     # depleted area / silicon area
    print(f"Nd={n/1e16:.2f}e16  Vbd={Vb:6.2f}  depth/L={depth_frac[-1]:.2f}  area={area_frac[-1]:.2f}")
Vbd = np.array(Vbd); depth_frac = np.array(depth_frac); area_frac = np.array(area_frac)

# --- breakdown maximum ---
imax = np.nanargmax(Vbd); Nd_star, V_star = Nd[imax], Vbd[imax]

# --- onset of partial depletion at breakdown: crossing Vbd = VFD ---
VFD_at = np.interp(Nd, Nd_fine, VFD)
d = Vbd - VFD_at
s = np.where(np.diff(np.sign(d)) != 0)[0]
if len(s):
    i = s[0]
    t = d[i] / (d[i] - d[i+1])
    Nd_on = Nd[i] + t*(Nd[i+1]-Nd[i]); V_on = Vbd[i] + t*(Vbd[i+1]-Vbd[i])
else:
    Nd_on = V_on = np.nan

# ---------------- plot ----------------
fig, (axT, axB) = plt.subplots(2, 1, figsize=(10, 8.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
x = Nd/1e16; xf = Nd_fine/1e16; xon = Nd_on/1e16

axT.plot(x, Vbd, "-", color="#1f77b4", lw=2.4,
         label="Breakdown: obstacle PDE + ionization integral")
axT.plot(xf, VFD, "--", color="#d62728", lw=2.2, label="Full-depletion threshold VFD")
axT.plot([Nd_star/1e16], [V_star], "*", color="#1f77b4", ms=20, mec="k", mew=0.5,
         label="Breakdown maximum (Green series)")
axT.plot([xon], [V_on], "o", color="#d62728", ms=11, mec="k", mew=0.5,
         label="Onset of partial depletion at breakdown")
axT.axvline(Nd_star/1e16, ls=":", color="#1f77b4", lw=1.2)
axT.axvline(xon, ls=":", color="#d62728", lw=1.2)
axT.axvspan(xon, x[-1], color="#ffe8d6", alpha=0.6, zorder=0)
axT.set_ylabel("Voltage (V)"); axT.set_ylim(0, 100); axT.grid(alpha=0.3)
axT.legend(loc="lower center", fontsize=9)
axT.set_title("a=0.9 um, tox=0.4 um, L=3 um | 300 K | grounded field plate\n"
              "Shading: partially depleted at breakdown; no fitting\n"
              "The breakdown maximum occurs BEFORE partial depletion begins", fontsize=11)

axB.plot(x, depth_frac, "-", color="#1f77b4", lw=2.2, label="Centerline depleted depth / L")
axB.plot(x, area_frac,  "-", color="#ff7f0e", lw=2.2, label="Depleted area / silicon area")
axB.axvline(Nd_star/1e16, ls=":", color="#1f77b4", lw=1.2)
axB.axvline(xon, ls=":", color="#d62728", lw=1.2)
axB.axvspan(xon, x[-1], color="#ffe8d6", alpha=0.6, zorder=0)
axB.set_ylabel("Depletion at breakdown"); axB.set_ylim(0, 1.05)
axB.set_xlabel(r"Nd ($10^{16}$ cm$^{-3}$)"); axB.grid(alpha=0.3)
axB.legend(loc="lower left", fontsize=9)

fig.tight_layout()
fig.savefig("breakdown_vs_depletion.png", dpi=140)
print(f"\nbreakdown max: {V_star:.2f} V @ Nd={Nd_star/1e16:.3f}e16")
print(f"partial-depletion onset: Nd={Nd_on/1e16:.3f}e16, V={V_on:.2f} V")
print("saved breakdown_vs_depletion.png")
