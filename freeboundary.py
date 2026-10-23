"""
Moving depletion-boundary (free-boundary) extension  -- report Sections 7-9.

u = V - phi >= 0.  Depleted set D = {u>0}, neutral set Z = {u=0}.
  In D:  Delta u = k = q N_d / eps_s   (fixed donor charge)
  In Z:  u = 0                         (drain-connected neutral screen, phi=V, E=0)

Obstacle / LCP (cell-centered finite volume, Sec. 8):
  u >= 0,   A u - f_V >= 0,   u .* (A u - f_V) = 0,   f_V = V r - k 1
with  A = discrete (-Delta) + homogeneous boundary terms,  r = unit-bias loads.

Boundary conditions on u:
  u(0,y)=V (top),  u(L,y)=0 (drain),  u_y(x,0)=0 (symmetry),
  u_y(x,a)+u(x,a)/b = V/b (Robin wall).

At every trial V we solve the LCP, take the centerline field E+ = d phi/dx|_{y=0},
and impose the same van Overstraeten-de Man electron criterion I_n(V,N_d)=1.
All in cm / V/cm.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from vbd_vdm import alpha_n, alpha_p, eps_s, eps_I, eps0, qe, um, trap

# ---------------- FV operator (depends only on mesh + geometry) ----------------
def build_operator(nx, ny, L, a, b):
    dx, dy = L/nx, a/ny
    idx = lambda i, j: i*ny + j
    N = nx*ny
    rows, cols, vals = [], [], []
    r = np.zeros(N)
    ix2, iy2 = 1.0/dx**2, 1.0/dy**2
    wall = 1.0/(dy*(b + dy/2.0))
    def add(p, qv, v):
        rows.append(p); cols.append(qv); vals.append(v)
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j); diag = 0.0
            # ---- x direction ----
            if i > 0:
                add(p, idx(i-1, j), -ix2); diag += ix2
            else:                      # top Dirichlet u=V
                diag += 2*ix2; r[p] += 2*ix2
            if i < nx-1:
                add(p, idx(i+1, j), -ix2); diag += ix2
            else:                      # drain Dirichlet u=0
                diag += 2*ix2
            # ---- y direction ----
            if j > 0:
                add(p, idx(i, j-1), -iy2); diag += iy2
            # else j==0 symmetry Neumann: no flux, no term
            if j < ny-1:
                add(p, idx(i, j+1), -iy2); diag += iy2
            else:                      # Robin wall
                diag += wall; r[p] += wall
            add(p, p, diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, r, dx, dy

# ---------------- LCP solve by primal-dual active set ----------------
def solve_obstacle(A, f, maxit=60, tol=1e-12):
    N = A.shape[0]
    Z = np.zeros(N, bool)                       # neutral (u=0) set
    u = np.zeros(N)
    for _ in range(maxit):
        I = ~Z
        AII = A[I][:, I]
        uI = spla.spsolve(AII.tocsc(), f[I])
        u = np.zeros(N); u[I] = uI
        lam = A @ u - f
        neg   = (u < -tol) & I                  # depleted went negative -> neutral
        wrong = (lam < -tol) & Z                # neutral wants to deplete
        if not neg.any() and not wrong.any():
            break
        Z[neg] = True; Z[wrong] = False
    return u, Z

# ---------------- centerline field + avalanche ----------------
def centerline_In(u, V, nx, ny, dx, npts=2001):
    phi_row = V - u.reshape(nx, ny)[:, 0]        # y ~ dy/2 (symmetry row)
    xc = (np.arange(nx) + 0.5) * dx
    Ep = np.gradient(phi_row, xc)                # d phi/dx (V/cm)
    xs = np.linspace(xc[0], xc[-1], npts)
    Eps = interp1d(xc, Ep, kind="cubic")(xs)
    F = np.maximum(Eps, 0.0)
    an = alpha_n(F); ap = alpha_p(F)
    h = xs[1]-xs[0]
    C = np.concatenate([[0.0], cumulative_trapezoid(ap - an, dx=h)])
    return trap(an*np.exp(C), dx=h)

def Vbd_free(Nd, nx, ny, L, a, b, Vmax=400.0):
    A, r, dx, dy = build_operator(nx, ny, L, a, b)
    k = qe*Nd/eps_s
    def g(V):
        u, Z = solve_obstacle(A, V*r - k)
        return centerline_In(u, V, nx, ny, dx) - 1.0
    Vlo, Vhi = 1.0, 8.0
    while g(Vhi) < 0 and Vhi < Vmax:
        Vhi *= 1.5
    Vb = brentq(g, Vlo, Vhi, xtol=1e-6, rtol=1e-10)
    u, Z = solve_obstacle(A, Vb*r - k)
    neutral_frac = Z.mean()
    return Vb, neutral_frac, (u, Z, dx, dy)

# ---------------- validation on A / B / C ----------------
if __name__ == "__main__":
    a  = 0.9*um; WI = 0.4*um; L = 3.0*um
    b  = (eps_s/eps_I)*WI
    targets = [("A rising",  3.000e15, 67.214582, 0.0),
               ("B maximum", 1.293816e16, 79.762223, 0.0),
               ("C partial", 3.000e16, 31.541869, 0.2626)]
    print(f"geometry a=0.9 WI=0.4 L=3.0 um, b={b/um:.3f} um\n")
    for name, Nd, Vref, fref in targets:
        nx, ny = (360, 120) if name.startswith("C") else (240, 80)
        Vb, nf, _ = Vbd_free(Nd, nx, ny, L, a, b)
        print(f"{name}: Nd={Nd:.3e}  Vbd={Vb:7.3f} V (report {Vref:.3f})  "
              f"neutral={nf*100:5.2f}% (report {fref*100:.2f}%)  mesh {nx}x{ny}")
