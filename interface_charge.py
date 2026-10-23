"""
Effect of a fixed Si/oxide INTERFACE CHARGE on breakdown voltage V_BD vs N_D.

A sheet charge Q_f = q*N_f (N_f in cm^-2) at the Si/oxide wall y=a changes the
Gauss/Robin condition to
    phi(x,a) + b phi_y(x,a) = V_Q ,   V_Q = Q_f * t_ox / eps_o ,   b=(eps_s/eps_o)t_ox
(V_Q = 0 is the clean interface used so far).  In u = V - phi this is
    u_y + u/b = (V - V_Q)/b ,
so the finite-volume Robin-wall load scales with (V - V_Q):
    f_V = V*(r_top + r_wall) - V_Q*r_wall - k*1 .

Positive N_f (positive interface charge) accumulates the n-drift sidewall -> less
depletion; negative N_f enhances depletion.  Breakdown from the moving-boundary
obstacle PDE + van Overstraeten-de Man centerline ionization integral.
Geometry a=0.9 um, tox=0.4 um, L=3 um, grounded field plate, 300 K.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freeboundary as fb
from fp_sweep import build_operator_split
from vbd_vdm import eps_s, eps_I, eps0, qe, um

a, tox, L = 0.9*um, 0.4*um, 3.0*um
b = (eps_s/eps_I)*tox
eps_ox = 3.9*eps0
nx, ny = 120, 40
A, r_top, r_wall, dx, dy = build_operator_split(nx, ny, L, a, b)
rbase = r_top + r_wall

def VQ_of_Nf(Nf_cm2):
    return qe*Nf_cm2*tox/eps_ox            # volts

def Vbd_q(Nd, VQ, Vmax=400.0):
    k = qe*Nd/eps_s
    def g(V):
        u, Z = fb.solve_obstacle(A, V*rbase - VQ*r_wall - k)
        return fb.centerline_In(u, V, nx, ny, dx) - 1.0
    Vlo, Vhi = 1.0, 8.0
    while g(Vhi) < 0 and Vhi < Vmax:
        Vhi *= 1.5
    if g(Vhi) < 0:
        return np.nan
    return brentq(g, Vlo, Vhi, xtol=1e-5, rtol=1e-9)

# interface charge set (cm^-2): negative, zero, positive
Nf_list = [-1e12, -5e11, 0.0, 5e11, 1e12]
colors  = ["#08519c", "#6baed6", "#2ca02c", "#fdae6b", "#d62728"]
Nd_list = np.linspace(0.4e16, 3.8e16, 40)

fig, ax = plt.subplots(figsize=(9, 6))
for Nf, col in zip(Nf_list, colors):
    VQ = VQ_of_Nf(Nf)
    V = np.array([Vbd_q(n, VQ) for n in Nd_list])
    sign = "+" if Nf > 0 else ("" if Nf == 0 else "")
    lab = (f"N_f = {Nf/1e12:+.1f}e12 cm$^{{-2}}$  (V_Q={VQ:+.1f} V)"
           if Nf != 0 else "N_f = 0  (clean interface)")
    ax.plot(Nd_list/1e16, np.where(V > 0, V, np.nan), "-", color=col, lw=2.1, label=lab)
    print(f"N_f={Nf:+.2e} cm^-2  VQ={VQ:+6.2f} V  "
          f"Vbd,max={np.nanmax(V):.2f} V @ Nd={Nd_list[np.nanargmax(V)]/1e16:.2f}e16")

ax.set_xlabel(r"N_D ($10^{16}$ cm$^{-3}$)"); ax.set_ylabel("Breakdown voltage V_BD (V)")
ax.set_title("Breakdown vs doping with Si/oxide interface charge\n"
             "obstacle PDE + ionization integral; a=0.9, tox=0.4, L=3 um")
ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, title="interface charge")
fig.tight_layout(); fig.savefig("interface_charge.png", dpi=140)
print("saved interface_charge.png")
