"""
3D Green's-function / eigenfunction solution of the reduced trench-cell Poisson problem.

    -(phi_xx + phi_yy + phi_zz) = f = q*N_D/eps_s     in  0<x<L, 0<y<a, 0<z<c

BCs:
    x=0        : phi = 0                         (top)
    x=L        : phi = V                         (drain)
    y=0, z=0   : Neumann (symmetry)
    y=a        : phi + b_y phi_y = 0   b_y=(eps_s/eps_o) t_ox_y
    z=c        : phi + b_z phi_z = 0   b_z=(eps_s/eps_o) t_ox_z

Transverse eigenproblems:
    Y_m = cos(mu_m y),  cot(mu_m a)=b_y mu_m
    Z_n = cos(nu_n z),  cot(nu_n c)=b_z nu_n
Field:
    phi = sum_{m,n} X_{mn}(x) cos(mu_m y) cos(nu_n z),  kappa=sqrt(mu^2+nu^2)

Cross-checked against a direct 3D finite-difference solve, then plotted (2D + 3D).
"""

import numpy as np
from scipy.optimize import brentq

# ---------------- parameters (SI) ----------------
q     = 1.602e-19
eps0  = 8.854e-12
eps_s = 11.7 * eps0
eps_o = 3.9  * eps0
N_D   = 1e22
tox_y = 50e-9
tox_z = 50e-9
L     = 2e-6
a     = 0.5e-6
c     = 0.5e-6
V     = 5.0

f  = q * N_D / eps_s
by = (eps_s / eps_o) * tox_y
bz = (eps_s / eps_o) * tox_z

# ---------------- eigenvalues of  cot(l*w)=b*l ----------------
def _eigs(w, b, n_modes):
    F = lambda l: np.cos(l * w) - b * l * np.sin(l * w)
    roots, k = [], 0
    while len(roots) < n_modes:
        lo = (k * np.pi + 1e-6) / w
        hi = ((k + 1) * np.pi - 1e-6) / w
        if F(lo) * F(hi) < 0:
            roots.append(brentq(F, lo, hi, xtol=1e-14, rtol=1e-13))
        k += 1
        if k > n_modes + 50:
            break
    return np.array(roots)

def _coeffs(lam, w):
    """norm N and constant-projection coeff for each eigenvalue."""
    N = w / 2 + np.sin(2 * lam * w) / (4 * lam)
    cst = np.sin(lam * w) / (lam * N)
    return N, cst

def _sinh_ratio(u, wv):
    return np.exp(u - wv) * (1 - np.exp(-2 * u)) / (1 - np.exp(-2 * wv))

# ---------------- analytic 3D series ----------------
def phi_series(X, Y, Z, my=40, mz=40):
    mu = _eigs(a, by, my); _, cy = _coeffs(mu, a)
    nu = _eigs(c, bz, mz); _, cz = _coeffs(nu, c)
    out = np.zeros(X.shape)
    for i, m in enumerate(mu):
        cosy = np.cos(m * Y)
        for j, n in enumerate(nu):
            kap = np.hypot(m, n)
            d   = cy[i] * cz[j]
            g   = f * d / kap**2
            Xmn = (g
                   + _sinh_ratio(kap * (L - X), kap * L) * (-g)
                   + _sinh_ratio(kap * X, kap * L) * (V * d - g))
            out += Xmn * cosy * np.cos(n * Z)
    return out

# ---------------- 3D finite-difference reference ----------------
def phi_fd(nx=49, ny=25, nz=25):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    x = np.linspace(0, L, nx); y = np.linspace(0, a, ny); z = np.linspace(0, c, nz)
    dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
    idx = lambda i, j, k: (i*ny + j)*nz + k
    N = nx*ny*nz
    A = sp.lil_matrix((N, N)); rhs = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = idx(i, j, k)
                if i == 0:
                    A[p, p] = 1; rhs[p] = 0.0; continue
                if i == nx-1:
                    A[p, p] = 1; rhs[p] = V; continue
                A[p, idx(i-1,j,k)] += 1/dx**2
                A[p, idx(i+1,j,k)] += 1/dx**2
                diag = -2/dx**2
                rhs[p] = -f
                # y direction
                if j == 0:
                    A[p, idx(i,1,k)] += 2/dy**2; diag += -2/dy**2
                elif j == ny-1:
                    A[p, p] += (1/dy**2)/(1 + dy/by); A[p, idx(i,j-1,k)] += 1/dy**2
                    diag += -2/dy**2
                else:
                    A[p, idx(i,j-1,k)] += 1/dy**2; A[p, idx(i,j+1,k)] += 1/dy**2
                    diag += -2/dy**2
                # z direction
                if k == 0:
                    A[p, idx(i,j,1)] += 2/dz**2; diag += -2/dz**2
                elif k == nz-1:
                    A[p, p] += (1/dz**2)/(1 + dz/bz); A[p, idx(i,j,k-1)] += 1/dz**2
                    diag += -2/dz**2
                else:
                    A[p, idx(i,j,k-1)] += 1/dz**2; A[p, idx(i,j,k+1)] += 1/dz**2
                    diag += -2/dz**2
                A[p, p] += diag
    sol = spla.spsolve(A.tocsr(), rhs).reshape(nx, ny, nz)
    return x, y, z, sol

# ---------------- run + verify ----------------
if __name__ == "__main__":
    print(f"f={f:.3e} V/m^2   b_y={by:.3e} m   b_z={bz:.3e} m")
    print("mu_m a =", np.round(_eigs(a, by, 4)*a, 4))
    print("nu_n c =", np.round(_eigs(c, bz, 4)*c, 4))

    x, y, z, sol = phi_fd()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ser = phi_series(X, Y, Z, my=50, mz=50)
    print(f"max |series - FD| = {np.max(np.abs(ser-sol)):.3e} V  "
          f"(max |phi| ~ {np.max(np.abs(sol)):.3f} V)")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dense grid from the analytic series for smooth figures
    xf = np.linspace(0, L, 120); yf = np.linspace(0, a, 60); zf = np.linspace(0, c, 60)
    Xf, Yf, Zf = np.meshgrid(xf, yf, zf, indexing="ij")
    P = phi_series(Xf, Yf, Zf, my=50, mz=50)
    kz0 = 0          # z=0 symmetry plane
    ky0 = 0          # y=0 symmetry plane

    # ---------- figure 1: 2D slices ----------
    fig1, ax = plt.subplots(1, 3, figsize=(14, 4))
    c0 = ax[0].contourf(Xf[:,:,kz0]*1e6, Yf[:,:,kz0]*1e6, P[:,:,kz0], 30, cmap="viridis")
    ax[0].set_title("phi on z=0 plane"); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    fig1.colorbar(c0, ax=ax[0])
    c1 = ax[1].contourf(Xf[:,ky0,:]*1e6, Zf[:,ky0,:]*1e6, P[:,ky0,:], 30, cmap="viridis")
    ax[1].set_title("phi on y=0 plane"); ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("z (um)")
    fig1.colorbar(c1, ax=ax[1])
    kx = len(xf)//2
    c2 = ax[2].contourf(Yf[kx,:,:]*1e6, Zf[kx,:,:]*1e6, P[kx,:,:], 30, cmap="viridis")
    ax[2].set_title(f"phi on x={xf[kx]*1e6:.2f} um slice")
    ax[2].set_xlabel("y (um)"); ax[2].set_ylabel("z (um)")
    fig1.colorbar(c2, ax=ax[2])
    fig1.tight_layout(); fig1.savefig("green3d_slices.png", dpi=130)
    print("saved green3d_slices.png")

    # ---------- figure 2: 3D rendering (slice planes on the cube faces) ----------
    from matplotlib import cm
    norm = plt.Normalize(P.min(), P.max())
    fig2 = plt.figure(figsize=(8, 7)); ax3 = fig2.add_subplot(111, projection="3d")
    # three visible faces colored by phi
    ax3.plot_surface(Xf[:,:,kz0]*1e6, Yf[:,:,kz0]*1e6, np.zeros_like(P[:,:,kz0]),
                     facecolors=cm.viridis(norm(P[:,:,kz0])), shade=False, rstride=2, cstride=2)
    ax3.plot_surface(Xf[:,ky0,:]*1e6, np.zeros_like(P[:,ky0,:]), Zf[:,ky0,:]*1e6,
                     facecolors=cm.viridis(norm(P[:,ky0,:])), shade=False, rstride=2, cstride=2)
    ax3.plot_surface(Xf[-1,:,:]*1e6, Yf[-1,:,:]*1e6, Zf[-1,:,:]*1e6,
                     facecolors=cm.viridis(norm(P[-1,:,:])), shade=False, rstride=2, cstride=2)
    ax3.set_xlabel("x (um)"); ax3.set_ylabel("y (um)"); ax3.set_zlabel("z (um)")
    ax3.set_title("phi(x,y,z) on cell faces")
    m = cm.ScalarMappable(cmap="viridis", norm=norm); m.set_array(P)
    fig2.colorbar(m, ax=ax3, shrink=0.6, label="phi (V)")
    ax3.view_init(elev=22, azim=-60)
    fig2.tight_layout(); fig2.savefig("green3d_cube.png", dpi=130)
    print("saved green3d_cube.png")
