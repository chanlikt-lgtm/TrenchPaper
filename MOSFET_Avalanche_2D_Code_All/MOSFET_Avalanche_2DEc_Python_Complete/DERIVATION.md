# Mathematical derivation of the slight pre-peak increase and two breakdown criteria

## A. Green-function centerline field

Rectangular fully depleted silicon, source at x=0, drain at x=L, symmetry at y=0, local oxide Robin BC at y=a:

\[
-\nabla^2\phi=\frac{qN_D}{\varepsilon_s},\qquad
\phi(0,y)=0,\ \phi(L,y)=V_D,\ \phi_y(x,0)=0,\ \phi_y(x,a)+\frac{\phi(x,a)}b=0.
\]

The source report derives the unique solution \(\phi=V_D f+k h\), where \(k=qN_D/\varepsilon_s\). At y=0, define \(A(x)=f_x(x,0)\), \(B(x)=h_x(x,0)\). The report calls the positive longitudinal field \(F=\phi_x\); the physical electric field vector is \(\mathbf E=-\nabla\phi\). Therefore

\[
F(x)=V_D A(x)+K n B(x),\quad n=N_D/(10^{16}\,\mathrm{cm}^{-3}),\quad K=q10^{16}/\varepsilon_s.
\]

The code consistently uses x in micrometers and converts ionization integration variables to centimeters.


For reproducibility, the implemented equivalent longitudinal series is

\[
\begin{aligned}
P(x)&=V_D\frac{x}{L}+\frac{k}{2}x(L-x),\\
\mu_m&=\frac{m\pi}{L},\\
v_m&=\frac{2(-1)^{m+1}}{m\pi},\\
w_m&=\frac{2L^2[1-(-1)^m]}{(m\pi)^3},\\
R_m(y)&=\frac{\cosh(\mu_my)}{\cosh(\mu_ma)+b\mu_m\sinh(\mu_ma)},\\
\phi(x,y)&=P(x)-\sum_{m=1}^{\infty}(V_Dv_m+kw_m)R_m(y)\sin(\mu_mx).
\end{aligned}
\]

The centerline functions used in code are

\[
\begin{aligned}
A(x)&=\frac1L-\sum_{m\ge1}v_m\mu_m R_m(0)\cos(\mu_mx),\\
B(x)&=\frac L2-x-\sum_{m\ge1}w_m\mu_m R_m(0)\cos(\mu_mx).
\end{aligned}
\]

The rapid-decay exponential version of \(R_m\) in the source code avoids overflow in high hyperbolic modes.

## B. Avalanche condition and exact first variation

The report's 300 K van Overstraeten-de Man coefficients are \(\alpha_n(F)\) and \(\alpha_p(F)\), which are specified empirical/material coefficients. Let

\[
C(x)=\int_0^x[\alpha_p(F(t))-\alpha_n(F(t))]dt,
\quad I(V_D,n)=\int_0^L\alpha_n(F(x))\,e^{C(x)}\,dx.
\]

**Avalanche breakdown:** \(I(V_{\rm BD}(n),n)=1\), subject to the report's branch-admissibility conditions.

A variation \(\delta F\) changes the inner exponent by

\[
\delta C(x)=\int_0^x[\alpha_p'(F(s))-\alpha_n'(F(s))]\delta F(s)ds.
\]

Differentiate \(I\) and swap the order of integration:

\[
\delta I=\int_0^L W(s)\delta F(s)ds,
\]

\[
\boxed{W(s)=e^{C(s)}\alpha_n'(F(s))
+ [\alpha_p'(F(s))-\alpha_n'(F(s))]
\int_s^L\alpha_n(F(t))e^{C(t)}dt.}
\]

The code computes derivatives in units of (V/micrometer)^-1, accounting for the conversion from V/cm in the ionization parameters. The piecewise hole law has a coefficient switch at 400 kV/cm, so numerical derivatives at a point exactly on that switch warrant care.

Let \(J(s)=\int_s^L\alpha_n e^Cdt\). At threshold \(I=1\), the identity
\(e^{C(s)}-J(s)=\int_0^s \alpha_p e^Cdt\)
rewrites the weight as

\[
W(s)=\alpha_n'(F(s))\int_0^s\alpha_p e^Cdt
+\alpha_p'(F(s))\int_s^L\alpha_n e^Cdt\ge0.
\]

## C. No-fitting quantitative slope

Because \(\partial F/\partial V_D=A\) and \(\partial F/\partial n=KB\),

\[
\frac{\partial I}{\partial V_D}=\int_0^L W(x)A(x)dx,\qquad
\frac{\partial I}{\partial n}=K\int_0^L W(x)B(x)dx.
\]

Differentiate \(I(V_{\mathrm{BD}}(n),n)=1\):

\[
\boxed{\frac{dV_{\mathrm{BD}}}{dn}=-K\frac{\int_0^L W B dx}{\int_0^L W A dx}.}
\]

In the computed geometries, the denominator is positive. The source half contributes positively to the numerator and the drain half negatively. Before the maximum, the negative drain contribution wins, so dVBD/dn>0. At the maximum the two contributions sum to zero. After the maximum the source contribution dominates and dVBD/dn<0. All six illustrative values are tabulated in `data/Sensitivity_Analytical_vs_Numerical.csv` and checked against finite differences.

This is a *smooth integrated-ionization balance*, not necessarily a global 2D electric-field hotspot switch.

## D. Different 2D peak-field Ec criterion

At a point \(P=(x,y)\), the Green field has form \(\mathbf E(P)=V_D\mathbf d(P)+\mathbf E_0(P)\). A **different breakdown definition** is

\[
\max_{P\in\Omega_{\rm eval}}|\mathbf E(P)|=E_c.
\]

Define \(A_2=|\mathbf d|^2\), \(B_2=\mathbf d\cdot\mathbf E_0\), \(C_2=|\mathbf E_0|^2\). When \(A_2>0\) and the zero-bias field is below \(E_c\), the positive local root is

\[
V_c(P)=\frac{-B_2+\sqrt{B_2^2+A_2(E_c^2-C_2)}}{A_2},\qquad
V_{\rm BD}^{2D}=\min_{P\in\Omega_{\rm eval}} V_c(P).
\]

The 2D peak-field condition uses **one local vector magnitude**, whereas the centerline avalanche uses **ionization accumulated along the entire specified path**. They can yield distinct VBD magnitudes, slopes, and hotspot behavior—even using the same electrostatic potential. Do not compare the supplied numerical examples directly: the archived Ec example uses **different geometry** and an artificial corner cutoff.

## E. Physical limitations

Both models assume complete depletion, a rectangular silicon half-cell, and a local 1D oxide approximation. The ideal oxide/Dirichlet corners can produce singular fields, so naive global maxima on the unrounded rectangle are not physically meaningful. A device-level BVdss prediction requires resolved corners/fringing, self-consistent depletion, and appropriately 2D carrier multiplication.
