"""
2D and 3D potential and electric-field maps at breakdown, for the two regions of
the breakdown-vs-depletion figure:

  Region A  (fully depleted at breakdown)   : Nd ~ 1.31e16  (breakdown maximum)
  Region B  (partially depleted at breakdown): Nd = 3.0e16   (shaded / partial-depletion)

Both use the moving-boundary obstacle solution at V = V_BD(Nd).  phi = V - u;
E = -grad(phi).  The neutral (screened) region is outlined on the partial case.
Geometry a=0.9 um, tox=0.4 um, L=3 um, grounded field plate.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import freeboundary as fb
from vbd_vdm import eps_s, eps_I, um

a, tox, L = 0.9*um, 0.4*um, 3.0*um
b = (eps_s/eps_I)*tox
nx, ny = 160, 60

def solve_region(Nd):
    Vb, nf, (u, Z, dx, dy) = fb.Vbd_free(Nd, nx, ny, L, a, b)
    phi = (Vb - u).reshape(nx, ny)                 # V
    Zm = Z.reshape(nx, ny)
    xc = (np.arange(nx)+0.5)*dx                     # cm
    yc = (np.arange(ny)+0.5)*dy
    Ex = -np.gradient(phi, xc, axis=0)             # V/cm
    Ey = -np.gradient(phi, yc, axis=1)
    Emag = np.hypot(Ex, Ey)/1e6                     # MV/cm
    X, Y = np.meshgrid(xc/um, yc/um, indexing="ij")# um
    return dict(Vb=Vb, nf=nf, phi=phi, Emag=Emag, Z=Zm, X=X, Y=Y)

regions = [("A: fully depleted\nNd=1.31e16", 1.31e16),
           ("B: partial depletion\nNd=3.0e16", 3.0e16)]
data = [(lbl, solve_region(Nd)) for lbl, Nd in regions]
for lbl, d in data:
    print(f"{lbl.splitlines()[0]}  Vbd={d['Vb']:.2f} V  neutral={d['nf']*100:.1f}%  "
          f"|E|max={d['Emag'].max():.3f} MV/cm")

fig = plt.figure(figsize=(17, 9))
for row, (lbl, d) in enumerate(data):
    Ecap = np.percentile(d["Emag"], 99.0)          # clip corner singularity for color
    # ---- col 1: potential 2D ----
    ax1 = fig.add_subplot(2, 4, row*4+1)
    pc = ax1.pcolormesh(d["Y"], d["X"], d["phi"], cmap="viridis", shading="auto")
    ax1.contour(d["Y"], d["X"], d["Z"].astype(float), [0.5], colors="r", linewidths=1.5)
    ax1.invert_yaxis(); ax1.set_title(f"{lbl}\nPotential phi (V)", fontsize=9)
    ax1.set_xlabel("y (um)"); ax1.set_ylabel("x depth (um)")
    fig.colorbar(pc, ax=ax1, fraction=0.046)
    # ---- col 2: |E| 2D ----
    ax2 = fig.add_subplot(2, 4, row*4+2)
    pc = ax2.pcolormesh(d["Y"], d["X"], np.clip(d["Emag"], 0, Ecap),
                        cmap="inferno", shading="auto")
    ax2.contour(d["Y"], d["X"], d["Z"].astype(float), [0.5], colors="c", linewidths=1.5)
    ax2.invert_yaxis(); ax2.set_title("|E| (MV/cm)", fontsize=9)
    ax2.set_xlabel("y (um)"); ax2.set_ylabel("x depth (um)")
    fig.colorbar(pc, ax=ax2, fraction=0.046)
    # ---- col 3: potential 3D ----
    ax3 = fig.add_subplot(2, 4, row*4+3, projection="3d")
    ax3.plot_surface(d["X"], d["Y"], d["phi"], cmap="viridis", rstride=2, cstride=2,
                     linewidth=0, antialiased=True)
    ax3.set_title("phi (V) 3D", fontsize=9)
    ax3.set_xlabel("x (um)"); ax3.set_ylabel("y (um)"); ax3.set_zlabel("phi (V)")
    ax3.view_init(elev=28, azim=-120)
    # ---- col 4: |E| 3D ----
    ax4 = fig.add_subplot(2, 4, row*4+4, projection="3d")
    ax4.plot_surface(d["X"], d["Y"], np.clip(d["Emag"], 0, Ecap), cmap="inferno",
                     rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax4.set_title("|E| (MV/cm) 3D", fontsize=9)
    ax4.set_xlabel("x (um)"); ax4.set_ylabel("y (um)"); ax4.set_zlabel("|E|")
    ax4.view_init(elev=28, azim=-120)

fig.suptitle("Potential and electric field at breakdown — fully-depleted vs partial-depletion region\n"
             "red/cyan line = depletion/neutral boundary (partial case); "
             "a=0.9 um, tox=0.4 um, L=3 um", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("field_maps.png", dpi=135)
print("saved field_maps.png")
