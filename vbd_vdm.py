"""
V_BD by the Green-function centerline avalanche method (report Sections 4-5).

All lengths in cm, fields in V/cm, densities in cm^-3 (report convention).

Rapidly convergent centerline evaluation (Sec. 4), lifting potential
    P(x) = V x/L + (k/2) x(L-x),   -P'' = k,   mu_n = n*pi/L
    v_n = 2(-1)^{n+1}/(n pi),   w_n = 2 L^2 [1-(-1)^n]/(n pi)^3
    R_n(0) = 2 e^{-mu_n a} / [ (1+b mu_n) + (1-b mu_n) e^{-2 mu_n a} ]   (stable)
    phi = V f + k h ,  E+ = V f_x + k h_x   at y=0
      f   = x/L            - sum v_n R_n(0) sin(mu_n x)
      h   = (1/2)x(L-x)    - sum w_n R_n(0) sin(mu_n x)
      f_x = 1/L            - sum v_n mu_n R_n(0) cos(mu_n x)
      h_x = (L/2 - x)      - sum w_n mu_n R_n(0) cos(mu_n x)

Admissibility (Sec. 5): require V(1-f)-k h >= 0 and E+ >= 0 on the sampled
centerline; bottom limit h/(1-f) -> -h_x(L)/f_x(L).

Avalanche (van Overstraeten-de Man, 300 K):
    alpha_n = 7.03e5 exp(-1.231e6/F)
    alpha_p = 1.582e6 exp(-2.036e6/F)  (F<4e5);  6.71e5 exp(-1.693e6/F)  (F>=4e5)
    C(x)=int_0^x (alpha_p-alpha_n) dt ;  I_n=int_0^L alpha_n e^{C} dx = 1  -> V_BD
    check:  I_p=int_0^L alpha_p e^{C-C(L)} dx ,  I_n-1 = e^{C(L)}(I_p-1)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ---- constants (cm units) ----
eps0  = 8.8541878128e-14        # F/cm
eps_s = 11.7 * eps0
eps_I = 3.9  * eps0
qe    = 1.602176634e-19         # C
um    = 1e-4                    # cm per micron

# ---- avalanche coefficients ----
def alpha_n(F):
    Fp = np.where(F > 0, F, 1.0)
    return np.where(F > 0, 7.03e5 * np.exp(-1.231e6 / Fp), 0.0)

def alpha_p(F):
    Fp = np.where(F > 0, F, 1.0)
    lo = 1.582e6 * np.exp(-2.036e6 / Fp)
    hi = 6.71e5  * np.exp(-1.693e6 / Fp)
    return np.where(F > 0, np.where(F < 4e5, lo, hi), 0.0)

# ---- centerline basis (depends only on geometry L, a, b) ----
def centerline(L, a, b, npts=2001, n_modes=400):
    x = np.linspace(0.0, L, npts)
    n = np.arange(1, n_modes + 1)
    mu = n * np.pi / L
    vn = 2.0 * (-1.0)**(n + 1) / (n * np.pi)
    wn = 2.0 * L**2 * (1.0 - (-1.0)**n) / (n * np.pi)**3
    e1 = np.exp(-mu * a); e2 = np.exp(-2.0 * mu * a)
    Rn0 = 2.0 * e1 / ((1.0 + b*mu) + (1.0 - b*mu) * e2)
    cosM = np.cos(np.outer(x, mu))      # (npts, n_modes)
    sinM = np.sin(np.outer(x, mu))
    f   = x / L           - sinM @ (vn * Rn0)
    h   = 0.5 * x*(L - x) - sinM @ (wn * Rn0)
    fx  = 1.0 / L         - cosM @ (vn * mu * Rn0)
    hx  = (L/2.0 - x)     - cosM @ (wn * mu * Rn0)
    return dict(x=x, f=f, h=h, fx=fx, hx=hx, L=L)

def Vstart(k, g):
    f, h, fx, hx = g["f"], g["h"], g["fx"], g["hx"]
    one_f = 1.0 - f
    ok = one_f > 1e-9
    over = (h[ok] / one_f[ok]).max() if ok.any() else -np.inf
    over = max(over, -hx[-1] / fx[-1])              # L'Hopital bottom limit
    okf = fx > 0
    rev = (-hx[okf] / fx[okf]).max() if okf.any() else -np.inf
    return max(0.0, k*over, k*rev)

def _InIp(V, k, g):
    Ep = V * g["fx"] + k * g["hx"]                   # V/cm (native)
    F  = np.maximum(Ep, 0.0)
    an = alpha_n(F); ap = alpha_p(F)
    dx = g["x"][1] - g["x"][0]
    C  = np.concatenate([[0.0], cumulative_trapezoid(ap - an, dx=dx)])
    CL = C[-1]
    In = trap(an * np.exp(C), dx=dx)
    Ip = trap(ap * np.exp(C - CL), dx=dx)
    return In, Ip, CL

def Vbd(Nd_cm3, g, Vmax=1e4):
    k = qe * Nd_cm3 / eps_s
    V0 = Vstart(k, g)
    if _InIp(V0, k, g)[0] >= 1.0:
        return np.nan, V0, np.nan           # avalanche already met at admissible start
    gfun = lambda V: _InIp(V, k, g)[0] - 1.0
    Vhi = max(V0 * 1.5, V0 + 1.0)
    it = 0
    while gfun(Vhi) < 0 and Vhi < Vmax:
        Vhi *= 1.6; it += 1
        if it > 80: break
    if gfun(Vhi) < 0:
        return np.nan, V0, np.nan
    Vb = brentq(gfun, max(V0, 1e-12), Vhi, xtol=1e-9, rtol=1e-13)
    In, Ip, CL = _InIp(Vb, k, g)
    resid = (In - 1.0) - np.exp(CL) * (Ip - 1.0)
    return Vb, V0, resid

# ---------------- validation + sweep ----------------
if __name__ == "__main__":
    # --- experimental-geometry benchmark (report: 34.27 V computed vs 35.4 V meas.) ---
    aB, WI_B, LB, NdB = 0.425*um, 0.10*um, 1.25*um, 6.9e16
    bB = (eps_s / eps_I) * WI_B
    gB = centerline(LB, aB, bB)
    VbB, V0B, resB = Vbd(NdB, gB)
    print(f"[benchmark] a=0.425 WI=0.10 L=1.25 um, Nd=6.9e16: "
          f"Vbd={VbB:.3f} V (report 34.27, meas 35.4), Vstart={V0B:.2f}, "
          f"identity resid={resB:.2e}")

    # --- two plotted geometry families, grouped by silicon half width a ---
    families = {
        "Family 1 (W_I=0.25, L=1.6 um)": dict(WI=0.25*um, L=1.6*um),
        "Family 2 (W_I=0.40, L=3.0 um)": dict(WI=0.40*um, L=3.0*um),
    }
    a_list  = [0.2, 0.3, 0.5, 0.7, 0.9]          # um
    Nd_list = np.logspace(np.log10(1e15), np.log10(1.6e17), 251)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=False)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(a_list)))

    maxres = 0.0
    for axi, (fname, fp) in zip(ax, families.items()):
        b = (eps_s / eps_I) * fp["WI"]
        for aum, col in zip(a_list, colors):
            g = centerline(fp["L"], aum*um, b)
            V = []
            for Nd in Nd_list:
                vb, v0, res = Vbd(Nd, g)
                V.append(vb)
                if np.isfinite(res): maxres = max(maxres, abs(res))
            V = np.array(V)
            axi.semilogx(Nd_list, np.where(V > 0, V, np.nan), "-", color=col, lw=1.9,
                         label=f"a={aum} um")
        axi.set_title(fname); axi.set_xlabel("N_D (cm$^{-3}$)")
        axi.set_ylabel("V_BD (V)"); axi.grid(True, which="both", alpha=0.3)
        axi.legend(title="Si half-width")
    fig.suptitle("V_BD vs N_D  —  van Overstraeten-de Man centerline avalanche (Sec. 4-5)")
    fig.tight_layout(); fig.savefig("vbd_vdm.png", dpi=150)
    print(f"max identity residual over full sweep = {maxres:.2e}")
    print("saved vbd_vdm.png")
