"""
Apply the VBD regression concept (reduced features -> polynomial + thin-plate-spline
residual) to OUR moving-boundary breakdown model (freeboundary.py), which -- unlike
the downloaded fully-depleted surrogate -- also covers the PARTIAL-DEPLETION regime.

Model (target): V_BD from the obstacle PDE + van Overstraeten-de Man centerline
ionization, geometry a=0.9 um, L=3 um, grounded-referenced effective bias Veff
(Veff = VFP + q*Nox*tox/eps_ox, the same degeneracy the package uses).

Inputs:  Nd [cm^-3], tox [um], Veff [V].
Reduced features (package style):
    u = (tox - 0.35)/0.25 ,  z = Veff/20 ,  d = log10(Nd) - 16
Fit:  poly(deg 6)(u,z,d)  +  thin-plate-spline residual(u,z,d).
"""

import os, numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

import freeboundary as fb
from fp_sweep import build_operator_split
from vbd_vdm import eps_s, eps_I, eps0, qe, um

a, L = 0.9*um, 3.0*um
eps_ox = 3.9*eps0
NX, NY = 96, 32
CSV = "our_surrogate_train.csv"

# ---------- target model: moving-boundary V_BD(Nd, tox, Veff) ----------
def Vbd_model(Nd, tox_um, Veff, op_cache):
    key = round(tox_um, 4)
    if key not in op_cache:
        b = (eps_s/eps_I)*(tox_um*um)
        op_cache[key] = build_operator_split(NX, NY, L, a, b)
    A, r_top, r_wall, dx, dy = op_cache[key]
    rbase = r_top + r_wall
    k = qe*Nd/eps_s
    def g(V):
        u, Z = fb.solve_obstacle(A, V*rbase - Veff*r_wall - k)
        return fb.centerline_In(u, V, NX, NY, dx) - 1.0
    Vlo, Vhi = 1.0, 8.0
    while g(Vhi) < 0 and Vhi < 400:
        Vhi *= 1.5
    if g(Vhi) < 0:
        return np.nan
    return brentq(g, Vlo, Vhi, xtol=1e-5, rtol=1e-9)

# ---------- training grid (cached) ----------
def generate():
    tox_g  = np.array([0.10, 0.18, 0.28, 0.40, 0.52, 0.65])
    Veff_g = np.array([-20, -10, 0, 10, 20.0])
    Nd_g   = np.logspace(np.log10(1e15), np.log10(1.6e17), 44)
    op_cache = {}
    rows = []
    for tox in tox_g:
        for Veff in Veff_g:
            for Nd in Nd_g:
                V = Vbd_model(Nd, tox, Veff, op_cache)
                if np.isfinite(V):
                    rows.append((Nd, tox, Veff, V))
        print(f"  tox={tox:.2f} um done ({len(rows)} valid so far)")
    arr = np.array(rows)
    np.savetxt(CSV, arr, delimiter=",", header="Nd,tox,Veff,VBD", comments="")
    return arr

def features(Nd, tox, Veff):
    u = (tox - 0.35)/0.25
    z = Veff/20.0
    d = np.log10(Nd) - 16.0
    return np.stack([u, z, d], axis=-1)

if __name__ == "__main__":
    if os.path.exists(CSV):
        arr = np.loadtxt(CSV, delimiter=",", skiprows=1)
        print(f"loaded {len(arr)} cached training rows")
    else:
        print("generating training data (moving-boundary model)...")
        arr = generate()
    Nd, tox, Veff, VBD = arr.T
    X = features(Nd, tox, Veff)

    # train/holdout split
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))
    ntr = int(0.8*len(X))
    tr, te = idx[:ntr], idx[ntr:]

    # --- polynomial (degree 6) + thin-plate-spline residual (package concept) ---
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.interpolate import RBFInterpolator
    poly = PolynomialFeatures(degree=7)
    Phi = poly.fit_transform(X)
    lin = LinearRegression().fit(Phi[tr], VBD[tr])
    res_tr = VBD[tr] - lin.predict(Phi[tr])
    rbf = RBFInterpolator(X[tr], res_tr, neighbors=120,
                          kernel="thin_plate_spline", smoothing=1e-5)

    def predict(Xq):
        return lin.predict(poly.transform(Xq)) + rbf(Xq)

    pred_te = predict(X[te])
    err = pred_te - VBD[te]
    ss = 1 - np.sum(err**2)/np.sum((VBD[te]-VBD[te].mean())**2)
    print(f"\nHoldout ({len(te)} cases):  R2={ss:.6f}  RMSE={np.sqrt((err**2).mean()):.4f} V  "
          f"MAE={np.abs(err).mean():.4f} V  max={np.abs(err).max():.4f} V")

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
    ax[0].scatter(VBD[te], pred_te, s=14, alpha=0.6, color="#1f77b4")
    lo, hi = VBD.min(), VBD.max()
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set_xlabel("model VBD (V)"); ax[0].set_ylabel("surrogate VBD (V)")
    ax[0].set_title(f"Holdout parity  (R2={ss:.5f}, RMSE={np.sqrt((err**2).mean()):.3f} V)")
    ax[0].grid(alpha=0.3)

    # slice: VBD vs Nd for a few tox at Veff=0 (surrogate vs model)
    op_cache = {}
    Nds = np.logspace(np.log10(1e15), np.log10(1.6e17), 50)
    for tox, c in zip([0.18, 0.40, 0.65], ["#1f77b4", "#2ca02c", "#d62728"]):
        Vs = predict(features(Nds, np.full_like(Nds, tox), np.zeros_like(Nds)))
        Vm = np.array([Vbd_model(n, tox, 0.0, op_cache) for n in Nds])
        ax[1].semilogx(Nds, Vs, "-", color=c, lw=2, label=f"surrogate tox={tox}")
        ax[1].semilogx(Nds, Vm, "o", color=c, ms=3, alpha=0.5)
    ax[1].set_xlabel(r"Nd (cm$^{-3}$)"); ax[1].set_ylabel("VBD (V)")
    ax[1].set_title("surrogate (line) vs model (points), Veff=0")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
    fig.suptitle("Regression surrogate for OUR moving-boundary breakdown model "
                 "(covers partial depletion) | a=0.9 um, L=3 um")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("our_surrogate.png", dpi=140)
    print("saved our_surrogate.png")
