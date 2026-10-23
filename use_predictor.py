"""
Use the 4-input VBD surrogate (vbd_regression_package) for fast breakdown estimates.
Package geometry is fixed at L=1.6 um, a=0.5 um (our Family-1 cell), 300 K.
Inputs: Nd [cm^-3], tox [um], VFP [V], Nox [cm^-2 signed oxide charge].
Validated against the independent vbd_vdm solver (RMSE ~ 0.013 V).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vbd_regression_package"))
import vbd_4input_predict as pred

def vbd(Nd, tox, vfp=0.0, nox=0.0):
    try:
        r = pred.predict(Nd, tox, vfp, nox)
    except ValueError:
        return np.nan                 # outside the model's validity domain
    return r['VBD_V'] if r['VBD_V'] is not None else np.nan

Nd = np.logspace(np.log10(1e15), np.log10(1.55e17), 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# panel 1: oxide thickness
for tox, c in zip([0.10, 0.25, 0.50], ["#1f77b4", "#2ca02c", "#d62728"]):
    V = [vbd(n, tox) for n in Nd]
    ax[0].semilogx(Nd, V, "-", color=c, lw=2, label=f"tox={tox} um")
ax[0].set_title("oxide thickness (VFP=0, Nox=0)"); ax[0].legend()

# panel 2: field-plate bias
for vfp, c in zip([-15, 0, 15], ["#08519c", "#2ca02c", "#d62728"]):
    V = [vbd(n, 0.25, vfp=vfp) for n in Nd]
    ax[1].semilogx(Nd, V, "-", color=c, lw=2, label=f"VFP={vfp:+d} V")
ax[1].set_title("field-plate bias (tox=0.25, Nox=0)"); ax[1].legend()

# panel 3: oxide charge
for nox, c in zip([-5e11, 0, 5e11], ["#08519c", "#2ca02c", "#d62728"]):
    V = [vbd(n, 0.25, nox=nox) for n in Nd]
    ax[2].semilogx(Nd, V, "-", color=c, lw=2, label=f"Nox={nox:+.0e}")
ax[2].set_title("oxide charge (tox=0.25, VFP=0)"); ax[2].legend()

for a_ in ax:
    a_.set_xlabel(r"Nd (cm$^{-3}$)"); a_.grid(True, which="both", alpha=0.3)
ax[0].set_ylabel("VBD (V)")
fig.suptitle("Fast VBD estimates from the 4-input surrogate | L=1.6 um, a=0.5 um, 300 K")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("use_predictor.png", dpi=140)

# representative table
print("Representative estimates (L=1.6 um, a=0.5 um):")
print(f"{'Nd':>10} {'tox':>5} {'VFP':>5} {'Nox':>9}  {'VBD (V)':>8}")
for Nd0, tox0, vfp0, nox0 in [(1e16,0.25,0,0),(3e16,0.25,0,0),(3e16,0.25,15,0),
                               (3e16,0.25,0,5e11),(6e16,0.25,0,0),(1e16,0.10,0,0),
                               (1e16,0.50,0,0),(1e17,0.25,0,0)]:
    v = vbd(Nd0, tox0, vfp0, nox0)
    print(f"{Nd0:>10.1e} {tox0:>5.2f} {vfp0:>5.0f} {nox0:>9.1e}  "
          f"{(f'{v:8.3f}' if np.isfinite(v) else '   null')}")
print("saved use_predictor.png")
