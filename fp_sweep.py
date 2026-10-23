"""
Centerline depletion depth vs donor concentration, for different field-plate
potentials V_FP (grounded plate was the special case V_FP = 0).

Field plate at potential V_FP across the oxide changes the Si/oxide Robin
condition to   phi(x,a) + b phi_y(x,a) = V_FP ,  b = (eps_s/eps_o) t_ox.
In u = V - phi this is   u_y + u/b = (V - V_FP)/b , so the finite-volume Robin
wall load scales with (V - V_FP) instead of V:
    f_V = V*(r_top + r_wall) - V_FP*r_wall - k*1 .
A more negative plate (V_FP<0) drives stronger sidewall depletion.

Fixed geometry a=0.9 um, t_ox=0.4 um, L=3 um; fixed drain bias V (below).
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freeboundary as fb
from vbd_vdm import eps_s, eps_I, qe, um

# ---- operator with SEPARATE top and Robin-wall load vectors ----
def build_operator_split(nx, ny, L, a, b):
    dx, dy = L/nx, a/ny
    idx = lambda i, j: i*ny + j
    N = nx*ny
    rows, cols, vals = [], [], []
    r_top = np.zeros(N); r_wall = np.zeros(N)
    ix2, iy2 = 1.0/dx**2, 1.0/dy**2
    wall = 1.0/(dy*(b + dy/2.0))
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j); diag = 0.0
            if i > 0:
                rows.append(p); cols.append(idx(i-1, j)); vals.append(-ix2); diag += ix2
            else:
                diag += 2*ix2; r_top[p] += 2*ix2          # top Dirichlet u=V
            if i < nx-1:
                rows.append(p); cols.append(idx(i+1, j)); vals.append(-ix2); diag += ix2
            else:
                diag += 2*ix2                              # drain Dirichlet u=0
            if j > 0:
                rows.append(p); cols.append(idx(i, j-1)); vals.append(-iy2); diag += iy2
            if j < ny-1:
                rows.append(p); cols.append(idx(i, j+1)); vals.append(-iy2); diag += iy2
            else:
                diag += wall; r_wall[p] += wall            # Robin wall (field plate)
            rows.append(p); cols.append(p); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, r_top, r_wall, dx, dy

a, tox, L = 0.9*um, 0.4*um, 3.0*um
b = (eps_s/eps_I)*tox
nx, ny = 150, 50
A, r_top, r_wall, dx, dy = build_operator_split(nx, ny, L, a, b)

V_drain = 40.0                                   # fixed applied drain bias
VFP_list = [-60, -40, -20, 0, 20, 40, 60]
Nd_list  = np.logspace(np.log10(2e15), np.log10(1e17), 55)
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(VFP_list)))

def Wcl_vs_Nd(VFP):
    out = []
    base = V_drain*(r_top + r_wall) - VFP*r_wall
    for Nd in Nd_list:
        k = qe*Nd/eps_s
        u, Z = fb.solve_obstacle(A, base - k)
        D = (~Z).reshape(nx, ny)
        out.append(D[:, 0].sum() * dx / um)       # centerline depth (um)
    return np.array(out)

fig, ax = plt.subplots(figsize=(9, 6))
for VFP, col in zip(VFP_list, cmap):
    W = Wcl_vs_Nd(VFP)
    ax.semilogx(Nd_list, W, color=col, lw=2.0, label=f"{VFP:+d} V")
    print(f"VFP={VFP:+4d} V:  Wcl(1e17)={W[-1]:.2f} um")
ax.axhline(L/um, ls=":", color="0.6")
ax.set_xlabel(r"Nd (cm$^{-3}$)"); ax.set_ylabel("Centerline depletion depth (um)")
ax.set_title(f"Depletion depth vs donor concentration\n"
             f"field-plate sweep, drain bias V={V_drain:.0f} V, a=0.9, tox=0.4, L=3 um")
ax.set_ylim(0, 3.1); ax.grid(True, which="both", alpha=0.3)
ax.legend(title=r"$V_{FP}$", loc="lower left")
fig.tight_layout(); fig.savefig("fp_sweep.png", dpi=140)
print("saved fp_sweep.png")
