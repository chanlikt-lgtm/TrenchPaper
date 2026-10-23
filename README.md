# TrenchPaper — Green-function model of trench-MOSFET breakdown

Analytical/numerical modeling suite for the reduced trench-cell electrostatics and
avalanche breakdown, built around a Green-function solution of the half-cell Poisson
problem with a Robin (oxide/field-plate) boundary.

## Physics chain

1. **Green-function electrostatics** — half-cell Poisson problem
   `-∇²φ = qN_D/ε_s` with Dirichlet top/drain, symmetry (Neumann), and Si/oxide
   Robin boundary `φ + b φ_y = 0`, `b = (ε_s/ε_o)·t_ox`.
2. **Centerline avalanche** — rapidly convergent longitudinal sine expansion,
   admissibility screening, and van Overstraeten–de Man ionization integral `I_n = 1`.
3. **Moving depletion boundary** — partial-depletion obstacle problem `u = V - φ ≥ 0`
   solved by a finite-volume primal-dual active-set scheme.

## Scripts

| file | what it does |
|------|--------------|
| `green_halfcell.py` | 2D Green's series + finite-difference cross-check |
| `green_halfcell_3d.py` | 3D trench cell (two oxide faces), 2D/3D field plots |
| `vbd_vdm.py` | centerline avalanche V_BD (reproduces the 34.27 V experimental benchmark) |
| `vbd_vs_nd.py`, `vbd_zoom.py`, `map_tox.py` | V_BD vs N_D sweeps, RESURF optimum, design map |
| `freeboundary.py` | moving depletion-boundary obstacle solver (reproduces A/B/C) |
| `fig3.py` | B-to-C moving-boundary continuation |
| `fair_compare.py` | ionization integral vs fixed-E_c, identical electrostatics/admissibility |
| `biassweep.py` | depletion geometry vs drain bias |
| `fp_sweep.py` | centerline depletion depth vs N_D for field-plate potential V_FP = -60..+60 V |
| `breakdown_vs_depletion.py` | V_BD vs full-depletion threshold + depletion state at breakdown |
| `fp_sweep.py` | centerline depletion depth vs N_D for field-plate potential V_FP = -60..+60 V |
| `interface_charge.py` | V_BD vs N_D with signed Si/oxide interface charge |
| `use_predictor.py` | fast V_BD estimates via the surrogate in `vbd_regression_package/` |

## vbd_regression_package/

A standalone 4-input (`Nd`, `tox`, `VFP`, `Nox`) polynomial+thin-plate-spline
**surrogate** for V_BD on the fixed Family-1 geometry (L=1.6 um, a=0.5 um), built
on the same Green-function centerline-avalanche physics. Cross-checked against the
independent `vbd_vdm.py` solver (RMSE ~ 0.013 V; reference 47.604 V at Nd=3e16,
tox=0.25 reproduced exactly). Fully-depleted model only; returns null outside the
admissible branch. Run `python vbd_4input_predict.py --nd 3e16 --tox 0.25 --vfp 0 --nox 0`.

Generated figures (`*.png`) are included. Requires Python with numpy, scipy, matplotlib.

Reference: *Green function derivation of trench MOSFET breakdown* (Revised analytical report).
