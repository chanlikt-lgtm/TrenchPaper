"""
Fair process comparison:  ionization integral (solid)  vs  fixed Ec (dashed),
on IDENTICAL electrostatics (analytic fully-depleted centerline) and IDENTICAL
admissibility window.  No fitted curves.

Three geometry sweeps x two x-scales (linear / log):
    col 1  Oxide thickness  W_I in {0.2, 0.4, 0.6} um   (a=0.9, L=3.0)
    col 2  Full mesa width  2a  in {1.2, 1.8, 2.4} um   (W_I=0.4, L=3.0)
    col 3  Drift length     L   in {2, 3, 4} um         (W_I=0.4, a=0.9)

Admissibility (report Sec. 5), from E+ = V f_x + k h_x on the centerline:
    E+ >= 0 everywhere  ->  V in [Vstart, Vmax]
        Vstart = max(0, overshoot floor, max_{f_x>0} -k h_x/f_x)
        Vmax   = min_{f_x<0} (-k h_x/f_x)          (field-reversal ceiling)
    empty window  ->  geometry/doping inadmissible (curve terminates).
Integral criterion:  I_n(V)=1 (van Overstraeten-de Man) within the window.
Fixed-Ec criterion:  max_x E+(x) = Ec within the window.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
import vbd_vdm as vd
from vbd_vdm import centerline, alpha_n, alpha_p, eps_s, eps_I, qe, um, trap

def admissible(k, g):
    """Return the admissible bias window [Vstart, Vmax] from E+ = V f_x + k h_x >= 0."""
    f, h, fx, hx = g["f"], g["h"], g["fx"], g["hx"]
    one_f = 1.0 - f; ok = one_f > 1e-9
    over = (h[ok]/one_f[ok]).max() if ok.any() else -np.inf
    over = max(over, -hx[-1]/fx[-1])          # L'Hopital overshoot floor at drain
    pos = fx > 1e-12;  neg = fx < -1e-12
    lo_rev = (-hx[pos]/fx[pos]).max() if pos.any() else -np.inf
    Vstart = max(0.0, k*over, k*lo_rev)
    Vmax = (-k*hx[neg]/fx[neg]).min() if neg.any() else np.inf   # field-reversal ceiling
    return Vstart, Vmax

def In_of_V(V, k, g):
    Ep = V*g["fx"] + k*g["hx"]
    F = np.maximum(Ep, 0.0)
    dx = g["x"][1]-g["x"][0]
    C = np.concatenate([[0.0], cumulative_trapezoid(alpha_p(F)-alpha_n(F), dx=dx)])
    return trap(alpha_n(F)*np.exp(C), dx=dx)

def peakE(V, k, g):
    return np.max(V*g["fx"] + k*g["hx"])

VCAP = 2000.0    # finite search ceiling when no field-reversal bound exists

def Vbd_integral(Nd, g):
    # validated van Overstraeten-de Man solver (reproduces the 34.27 V benchmark)
    return vd.Vbd(Nd, g)[0]

def Vbd_Ec(k, g, Ec):
    V0, V1 = admissible(k, g)
    V1 = min(V1, VCAP)
    if not (V1 > V0):
        return np.nan
    # genuine interior-root branch only: field reaches Ec strictly inside the
    # admissible window.  Onset already past Ec (peakE(V0)>=Ec) or Ec unreachable
    # before field reversal (peakE(V1)<Ec) -> last passing point exceeded -> terminate.
    if peakE(V0, k, g) >= Ec or peakE(V1, k, g) < Ec:
        return np.nan
    return brentq(lambda V: peakE(V, k, g)-Ec, V0, V1, xtol=1e-6)

# ---------------- build sweeps ----------------
def geom_sets():
    base = dict(WI=0.4, a=0.9, L=3.0)
    cols = []
    cols.append(("Oxide thickness", "W_I",
                 [(f"{w} um", dict(WI=w, a=base['a'], L=base['L'])) for w in (0.2, 0.4, 0.6)]))
    cols.append(("Full mesa width", "2a",
                 [(f"{2*aa} um", dict(WI=base['WI'], a=aa, L=base['L'])) for aa in (0.6, 0.9, 1.2)]))
    cols.append(("Drift length", "L",
                 [(f"{LL} um", dict(WI=base['WI'], a=base['a'], L=LL)) for LL in (2.0, 3.0, 4.0)]))
    return cols

def curves(p, Nd):
    g = centerline(p["L"]*um, p["a"]*um, (eps_s/eps_I)*(p["WI"]*um))
    return g

if __name__ == "__main__":
    Nd = np.logspace(14, np.log10(4e16), 150)

    # ---- calibrate Ec from the base oxide case: match integral peak ----
    gbase = curves(dict(WI=0.4, a=0.9, L=3.0), Nd)
    Vi = np.array([Vbd_integral(n, gbase) for n in Nd])
    imax = np.nanargmax(Vi)
    kstar = qe*Nd[imax]/eps_s
    Ec = peakE(Vi[imax], kstar, gbase)
    print(f"calibrated Ec = {Ec:.4e} V/cm  (integral peak {Vi[imax]:.2f} V @ "
          f"Nd={Nd[imax]:.3e}, where both methods agree)")

    cols = geom_sets()
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for c, (title, sym, members) in enumerate(cols):
        for (label, p), col in zip(members, colors):
            g = curves(p, Nd)
            Vi = np.array([Vbd_integral(n, g) for n in Nd])
            Ve = np.array([Vbd_Ec(qe*n/eps_s, g, Ec) for n in Nd])
            x = Nd/1e16
            for row, xscale in enumerate(("linear", "log")):
                ax = axs[row, c]
                ax.plot(x, np.where(Ve > 0, Ve, np.nan), "--", color=col, lw=1.6,
                        label=f"{label}: Ec")
                ax.plot(x, np.where(Vi > 0, Vi, np.nan), "-", color=col, lw=1.8,
                        label=f"{label}: integral")
        for row, xscale in enumerate(("linear", "log")):
            ax = axs[row, c]
            ax.set_xscale(xscale)
            ax.set_title(f"{title} / {xscale}")
            ax.set_xlabel(r"$N_D\ (10^{16}\,cm^{-3})$"); ax.set_ylabel("Threshold voltage (V)")
            ax.set_ylim(0, 115); ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            if xscale == "linear": ax.set_xlim(0, 4)
            else: ax.set_xlim(1e-2, 4)
    fig.suptitle("Fair process comparison: solid = ionization integral; dashed = fixed Ec\n"
                 "Identical electrostatics and admissibility; no fitted curves", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fair_compare.png", dpi=130)
    print("saved fair_compare.png")
