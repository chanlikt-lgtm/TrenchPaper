"""
Breakdown voltage V_bd vs drift doping N_D, from the 2D Green's-function field,
grouped by oxide thickness T_ox. Two breakdown criteria:

  1) Ec method      : breakdown when peak |E| in the silicon reaches the
                      doping-dependent critical field  Ec = 4010 * Nd^(1/8) [V/cm].
  2) Integral method: breakdown when the impact-ionization integral along a
                      vertical path reaches 1, with Chynoweth
                      alpha(E) = A_ii * exp(-B_ii / E).

Field model: phi solves -(phi_xx+phi_yy)=qNd/eps_s with
  x=0 phi=0, x=L phi=V, y=0 Neumann, y=a Robin (phi + b phi_y =0), b=(eps_s/eps_o)Tox.
Because phi is linear in V, the field splits as  E = A + V*B  (A = doping part,
B = unit-bias part), so each (Nd,Tox) needs the modal field built only once.
"""

import numpy as np
from scipy.optimize import brentq

# ---- physical constants ----
q, eps0 = 1.602e-19, 8.854e-12
eps_s, eps_o = 11.7 * eps0, 3.9 * eps0

# ---- geometry ----
L = 2e-6           # drift depth (m)
a = 0.5e-6         # half cell width (m)

# ---- ionization / critical-field models (Si, cgs field units V/cm) ----
A_ii, B_ii = 7.03e5, 1.231e6        # Chynoweth electrons (1/cm , V/cm)
def Ec_of_Nd(Nd_cm3):               # V/cm, Nd in cm^-3
    return 4010.0 * Nd_cm3**0.125

# ---- transverse eigenvalues: cot(l a)=b l ----
def eigs(b, n_modes=60):
    F = lambda l: np.cos(l*a) - b*l*np.sin(l*a)
    r, k = [], 0
    while len(r) < n_modes:
        lo, hi = (k*np.pi+1e-6)/a, ((k+1)*np.pi-1e-6)/a
        if F(lo)*F(hi) < 0:
            r.append(brentq(F, lo, hi, xtol=1e-14))
        k += 1
        if k > n_modes+50: break
    return np.array(r)

def _sr(u, w):   return np.exp(u-w)*(1-np.exp(-2*u))/(1-np.exp(-2*w))   # sinh u/sinh w
def _cr(u, w):   return np.exp(u-w)*(1+np.exp(-2*u))/(1-np.exp(-2*w))   # cosh u/sinh w

def field_components(Nd_cm3, Tox, nx=90, nyv=46):
    """Return grids X,Y and A,B field parts: E = A + V*B  (V/m). dict with x/y comps."""
    f = q*(Nd_cm3*1e6)/eps_s
    b = (eps_s/eps_o)*Tox
    lam = eigs(b)
    x = np.linspace(0, L, nx); y = np.linspace(0, a, nyv)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Ax = np.zeros_like(X); Bx = np.zeros_like(X)
    Ay = np.zeros_like(X); By = np.zeros_like(X)
    for ln in lam:
        Nn = a/2 + np.sin(2*ln*a)/(4*ln)
        cn = np.sin(ln*a)/(ln*Nn)
        gn = f*cn/ln**2
        wL = ln*L
        R1  = _sr(ln*(L-X), wL);  R2  = _sr(ln*X, wL)
        R1p = -ln*_cr(ln*(L-X), wL); R2p = ln*_cr(ln*X, wL)
        cosy = np.cos(ln*Y); siny = np.sin(ln*Y)
        # Ex = -sum X_n' cos ; X_n' = -gn(R1'+R2') + V cn R2'
        Ax += (gn*(R1p+R2p))*cosy
        Bx += (-cn*R2p)*cosy
        # Ey = +sum ln X_n sin ; X_n = gn(1-R1-R2) + V cn R2
        Ay += (ln*gn*(1-R1-R2))*siny
        By += (ln*cn*R2)*siny
    return x, y, dict(Ax=Ax, Bx=Bx, Ay=Ay, By=By)

def corner_mask(X, drain_margin=0.06):
    """True on points to KEEP: exclude the thin band near the drain ohmic
    contact (x=L), where the Dirichlet/oxide corner is a numerical singularity.
    The drain is an ohmic contact, not a physical high-field site."""
    return X < L * (1.0 - drain_margin)

def Emax_of_V(F, V, mask=None):   # peak |E| (V/cm)
    Ex = F["Ax"] + V*F["Bx"]; Ey = F["Ay"] + V*F["By"]
    Emag = np.hypot(Ex, Ey)
    if mask is not None:
        Emag = Emag[mask]
    return np.max(Emag) / 100.0                  # V/m -> V/cm

def ion_integral_of_V(F, x, V):  # max vertical ionization integral (dimensionless)
    Ex = F["Ax"] + V*F["Bx"]; Ey = F["Ay"] + V*F["By"]
    Emag = np.hypot(Ex, Ey) / 100.0              # V/cm
    Emag = np.maximum(Emag, 1.0)
    alpha = A_ii*np.exp(-B_ii/Emag)              # 1/cm
    dx_cm = (x[1]-x[0])*100.0
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    II = trap(alpha, dx=dx_cm, axis=0)           # integrate along x for each y
    return II.max()

def _solve(g, Vmax):
    glo, ghi = g(1e-3), g(Vmax)
    if glo > 0: return 0.0        # charge alone already at/over the limit
    if ghi < 0: return np.nan     # no breakdown up to Vmax
    return brentq(g, 1e-3, Vmax)

def Vbd_Ec(F, Ec, mask, Vmax=2000.0):
    return _solve(lambda V: Emax_of_V(F, V, mask) - Ec, Vmax)

def Vbd_II(F, x, Vmax=2000.0):
    return _solve(lambda V: ion_integral_of_V(F, x, V) - 1.0, Vmax)

# ---------------- sweep ----------------
if __name__ == "__main__":
    Nd_list  = np.logspace(15, 17, 18)          # cm^-3
    Tox_list = [20e-9, 50e-9, 100e-9, 200e-9]

    res_Ec = {t: [] for t in Tox_list}
    res_II = {t: [] for t in Tox_list}
    for Tox in Tox_list:
        for Nd in Nd_list:
            x, y, F = field_components(Nd, Tox)
            X = np.meshgrid(x, y, indexing="ij")[0]
            mask = corner_mask(X)
            res_Ec[Tox].append(Vbd_Ec(F, Ec_of_Nd(Nd), mask))
            res_II[Tox].append(Vbd_II(F, x))
        print(f"Tox={Tox*1e9:4.0f} nm done")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(Tox_list)))
    for t, col in zip(Tox_list, colors):
        ax[0].loglog(Nd_list, res_Ec[t], "o-", color=col, label=f"Tox={t*1e9:.0f} nm")
        ax[1].loglog(Nd_list, res_II[t], "s-", color=col, label=f"Tox={t*1e9:.0f} nm")
    ax[0].set_title("1) Critical-field (Ec) method")
    ax[1].set_title("2) Ionization-integral method")
    for a_ in ax:
        a_.set_xlabel("N_D (cm$^{-3}$)"); a_.grid(True, which="both", alpha=0.3)
        a_.legend()
    ax[0].set_ylabel("V_bd (V)")
    fig.suptitle(f"Breakdown voltage vs doping  (L={L*1e6:.1f} um, a={a*1e6:.2f} um)")
    fig.tight_layout(); fig.savefig("vbd_vs_nd.png", dpi=140)
    print("saved vbd_vs_nd.png")

    # quick table
    print("\nNd(cm^-3)   " + "  ".join(f"Ec/II @{t*1e9:.0f}nm" for t in Tox_list))
    for i, Nd in enumerate(Nd_list):
        row = f"{Nd:9.2e} "
        for t in Tox_list:
            row += f"  {res_Ec[t][i]:6.1f}/{res_II[t][i]:6.1f}"
        print(row)
