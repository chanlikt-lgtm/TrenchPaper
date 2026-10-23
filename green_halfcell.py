"""
Green's-function / eigenfunction solution of the reduced half-cell Poisson problem.

    -(phi_xx + phi_yy) = f = q*N_D/eps_s     in  0<x<L, 0<y<a

BCs:
    x=0 : phi = 0                       (top)
    x=L : phi = V                       (drain)
    y=0 : phi_y = 0                     (symmetry / Neumann)
    y=a : phi + b*phi_y = 0             (Si/oxide Robin), b = (eps_s/eps_o)*t_ox

Transverse eigenproblem:  Y_n = cos(l_n y),  cot(l_n a) = b*l_n
Field:   phi = sum_n [ g_n (1 - cosh l_n x) + B_n sinh l_n x ] cos(l_n y)

The analytic series is cross-checked against a direct finite-difference solve.
"""

import numpy as np
from scipy.optimize import brentq

# ---------------- parameters (SI, representative numbers) ----------------
q     = 1.602e-19
eps0  = 8.854e-12
eps_s = 11.7 * eps0
eps_o = 3.9  * eps0
N_D   = 1e22            # m^-3
t_ox  = 50e-9          # m
L     = 2e-6           # depth
a     = 0.5e-6         # half-width
V     = 5.0            # drain bias

f = q * N_D / eps_s
b = (eps_s / eps_o) * t_ox

# ---------------- eigenvalues: cot(l a) = b l  ->  F(l)=cos(la)-b*l*sin(la)=0 ---
def F(l):
    return np.cos(l * a) - b * l * np.sin(l * a)

def eigenvalues(n_modes):
    """Bracket one root of F in each pi-interval of (l*a) and refine."""
    roots = []
    k = 0
    while len(roots) < n_modes:
        lo = (k * np.pi + 1e-6) / a
        hi = ((k + 1) * np.pi - 1e-6) / a
        if F(lo) * F(hi) < 0:
            roots.append(brentq(F, lo, hi, xtol=1e-14, rtol=1e-13))
        k += 1
        if k > n_modes + 50:
            break
    return np.array(roots)

# ---------------- analytic series ----------------
def _sinh_ratio(u, w):
    """sinh(u)/sinh(w) for 0<=u<=w, overflow-free."""
    return np.exp(u - w) * (1 - np.exp(-2 * u)) / (1 - np.exp(-2 * w))

def phi_series(x, y, n_modes=200):
    lam = eigenvalues(n_modes)
    out = np.zeros(np.broadcast(x, y).shape)
    for ln in lam:
        Nn = a / 2 + np.sin(2 * ln * a) / (4 * ln)
        cn = np.sin(ln * a) / (ln * Nn)          # projection of constant 1
        gn = f * cn / ln**2                       # particular amplitude X_n^p
        # stable two-point form:
        # X_n = gn + S(L-x)/S(L)*(0-gn) + S(x)/S(L)*(V cn - gn)
        Xn = gn + _sinh_ratio(ln * (L - x), ln * L) * (-gn) \
                + _sinh_ratio(ln * x, ln * L) * (V * cn - gn)
        out = out + Xn * np.cos(ln * y)
    return out

# ---------------- finite-difference reference ----------------
def phi_fd(nx=161, ny=81):
    x = np.linspace(0, L, nx)
    y = np.linspace(0, a, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    N = nx * ny
    idx = lambda i, j: i * ny + j
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    A = sp.lil_matrix((N, N))
    rhs = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            if i == 0:                       # x=0 Dirichlet
                A[p, p] = 1; rhs[p] = 0.0
            elif i == nx - 1:                # x=L Dirichlet
                A[p, p] = 1; rhs[p] = V
            else:
                # interior Laplacian  phi_xx + phi_yy = -f
                A[p, idx(i-1, j)] += 1/dx**2
                A[p, idx(i+1, j)] += 1/dx**2
                A[p, p]           += -2/dx**2 - 2/dy**2
                rhs[p] = -f
                if j == 0:                   # Neumann phi_y=0 : ghost = node j+1
                    A[p, idx(i, 1)] += 2/dy**2
                elif j == ny - 1:            # Robin phi + b phi_y = 0 at y=a
                    # one-sided: phi_y ~ (phi_j - phi_{j-1})/dy  ->  phi_N = phi_{N-1}/(1+dy/b)
                    A[p, idx(i, j)]   += (1/dy**2) * (1.0 / (1.0 + dy / b))
                    A[p, idx(i, j-1)] += 1/dy**2
                else:
                    A[p, idx(i, j-1)] += 1/dy**2
                    A[p, idx(i, j+1)] += 1/dy**2
    sol = spla.spsolve(A.tocsr(), rhs).reshape(nx, ny)
    return x, y, sol

# ---------------- verify ----------------
if __name__ == "__main__":
    print(f"f = q N_D/eps_s = {f:.4e} V/m^2")
    print(f"b = (eps_s/eps_o) t_ox = {b:.4e} m")
    lam = eigenvalues(6)
    print("first eigenvalues l_n a =", np.round(lam * a, 6))

    x, y, sol = phi_fd()
    X, Y = np.meshgrid(x, y, indexing="ij")
    ser = phi_series(X, Y, n_modes=300)
    err = np.max(np.abs(ser - sol))
    print(f"max |series - FD| = {err:.3e} V   (max |phi| ~ {np.max(np.abs(sol)):.3f} V)")

    # interface normal field E_n = phi_s/b  check
    phi_s = phi_series(X[:, -1], Y[:, -1], n_modes=300)
    print(f"interface phi_s range: [{phi_s.min():.3f}, {phi_s.max():.3f}] V")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        c0 = ax[0].contourf(X*1e6, Y*1e6, ser, 30, cmap="viridis")
        ax[0].set_title("phi(x,y) — Green's series [V]")
        ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
        fig.colorbar(c0, ax=ax[0])
        c1 = ax[1].contourf(X*1e6, Y*1e6, ser - sol, 30, cmap="coolwarm")
        ax[1].set_title("series - FD [V]")
        ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)")
        fig.colorbar(c1, ax=ax[1])
        fig.tight_layout()
        fig.savefig("green_halfcell.png", dpi=130)
        print("saved green_halfcell.png")
    except Exception as e:
        print("plot skipped:", e)
