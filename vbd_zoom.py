"""
Zoomed, high-resolution sweep of V_bd vs N_D around the RESURF maximum.
Reuses the field/breakdown machinery from vbd_vs_nd.py but with:
  - a dense N_D grid focused on the optimum band,
  - a finer transverse/axial field grid,
and marks V_bd,max for each T_ox.
"""

import numpy as np
import vbd_vs_nd as m   # field_components, Vbd_Ec, Vbd_II, corner_mask, Ec_of_Nd

# dense doping grid over the optimum region
Nd_list  = np.logspace(np.log10(8e15), np.log10(7e16), 60)   # cm^-3
Tox_list = [20e-9, 50e-9, 100e-9, 200e-9]

def sweep(nx=160, nyv=80, margin=0.06):
    Ec, II = {t: [] for t in Tox_list}, {t: [] for t in Tox_list}
    for Tox in Tox_list:
        for Nd in Nd_list:
            x, y, F = m.field_components(Nd, Tox, nx=nx, nyv=nyv)
            X = np.meshgrid(x, y, indexing="ij")[0]
            mask = m.corner_mask(X, drain_margin=margin)
            Ec[Tox].append(m.Vbd_Ec(F, m.Ec_of_Nd(Nd), mask))
            II[Tox].append(m.Vbd_II(F, x))
        print(f"Tox={Tox*1e9:4.0f} nm done")
    return {k: np.array(v) for k, v in Ec.items()}, {k: np.array(v) for k, v in II.items()}

def peak(Nd, V):
    """refine the maximum with a quadratic fit in log(Nd) around the argmax."""
    V = np.array(V, float)
    good = np.isfinite(V) & (V > 0)
    if good.sum() < 3:
        return np.nan, np.nan
    i = np.nanargmax(np.where(good, V, -np.inf))
    lo, hi = max(i-2, 0), min(i+3, len(V))
    xs, ys = np.log10(Nd[lo:hi]), V[lo:hi]
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 3:
        return Nd[i], V[i]
    c = np.polyfit(xs[ok], ys[ok], 2)
    if c[0] >= 0:
        return Nd[i], V[i]
    xm = -c[1]/(2*c[0])
    return 10**xm, np.polyval(c, xm)

if __name__ == "__main__":
    Ec, II = sweep()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(Tox_list)))

    print("\n--- V_bd,max (parabolic-refined) ---")
    for t, col in zip(Tox_list, colors):
        for a_, data, meth in ((ax[0], Ec[t], "Ec"), (ax[1], II[t], "II")):
            v = np.where(data > 0, data, np.nan)
            a_.semilogx(Nd_list, v, "-", color=col, lw=1.8)
            a_.semilogx(Nd_list, v, ".", color=col, ms=5)
            nmax, vmax = peak(Nd_list, data)
            if np.isfinite(vmax):
                a_.plot([nmax], [vmax], "*", color=col, ms=15, mec="k", mew=0.6)
                print(f"  {meth}  Tox={t*1e9:3.0f}nm :  Vbd,max={vmax:6.2f} V  "
                      f"@ Nd={nmax:.3e} cm^-3")
        # legend handle
        ax[0].plot([], [], "-", color=col, label=f"Tox={t*1e9:.0f} nm")

    ax[0].set_title("1) Ec method (corner-masked) — zoom")
    ax[1].set_title("2) Ionization-integral method — zoom")
    for a_ in ax:
        a_.set_xlabel("N_D (cm$^{-3}$)")
        a_.grid(True, which="both", alpha=0.3)
    ax[0].set_ylabel("V_bd (V)"); ax[0].legend()
    fig.suptitle("Zoom on RESURF optimum (dense Nd, fine grid; ★ = Vbd,max)")
    fig.tight_layout(); fig.savefig("vbd_zoom.png", dpi=150)
    print("\nsaved vbd_zoom.png")
