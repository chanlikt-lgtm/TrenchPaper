"""
Figure 3 reproduction: B-to-C continuation with a moving depletion boundary.
Fixed a=0.9 um, W_I=0.4 um, L=3.0 um.  Solid line = moving-boundary FV model
(120x40) at every doping; dashed = old fixed fully-depleted analytic Green model.
A and B marked from the analytic solution; C from the refined moving-boundary solve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vbd_vdm as an
import freeboundary as fb

um = an.um
a, WI, L = 0.9*um, 0.4*um, 3.0*um
b = (an.eps_s/an.eps_I)*WI

# --- analytic fully-depleted curve (old model) ---
g_an = an.centerline(L, a, b)
Nd_an = np.logspace(np.log10(3e15), np.log10(5e16), 70)
V_an = np.array([an.Vbd(Nd, g_an)[0] for Nd in Nd_an])

# --- moving-boundary curve (new model), 120x40 ---
Nd_fb = np.logspace(np.log10(3e15), np.log10(5e16), 44)
V_fb, nf_fb = [], []
for Nd in Nd_fb:
    Vb, nf, _ = fb.Vbd_free(Nd, 120, 40, L, a, b)
    V_fb.append(Vb); nf_fb.append(nf)
    print(f"Nd={Nd:.3e}  Vbd={Vb:6.2f} V  neutral={nf*100:5.2f}%")
V_fb = np.array(V_fb)

# --- A / B / C reference points ---
pts = {"A": (3.000e15, 67.215, "#1f77b4"),
       "B": (1.293816e16, 79.762, "#d62728"),
       "C": (3.000e16, 31.542, "#7030a0")}

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.semilogx(Nd_fb, V_fb, "-", color="teal", lw=2.2,
            label="New: moving depletion boundary (120x40)")
ax.semilogx(Nd_an, np.where(V_an > 0, V_an, np.nan), "--", color="gray", lw=1.6,
            label="Old: fixed fully-depleted Green model")
ax.axvline(2.2e16, color="0.6", ls=":", lw=1.2)
ax.text(2.2e16, 20, " Old cutoff", color="0.4", rotation=90, va="bottom", fontsize=9)
for name, (Nd, V, col) in pts.items():
    ax.plot([Nd], [V], "o", color=col, ms=9, mec="k", mew=0.6)
    ax.annotate(f"{name}: {V:.3f} V\nNd={Nd:.4g} cm$^{{-3}}$",
                (Nd, V), textcoords="offset points", xytext=(8, 8),
                fontsize=8.5, color=col)
ax.set_xlabel("N_D (cm$^{-3}$)"); ax.set_ylabel("Centerline avalanche voltage  V_BD (V)")
ax.set_title("B-to-C extension from charge screening and partial depletion\n"
             "Fixed a=0.9 um, W_I=0.4 um, L=3 um; no curve fitting")
ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig3_moving_boundary.png", dpi=150)
print("saved fig3_moving_boundary.png")
