FOUR-INPUT BREAKDOWN-VOLTAGE REGRESSION (OCTOBER 23, 2026)
==========================================================

Model inputs
------------
  Nd       : ionized donor concentration [cm^-3]
  tox      : silicon-oxide thickness [um]
  VFP      : field-plate bias relative to the grounded source [V]
  Nox      : signed fixed oxide sheet charge [cm^-2]

Model limits (no extrapolation):
  1.0e15 <= Nd <= 1.6e17 cm^-3
  0.05 <= tox <= 0.75 um
  -15 <= VFP <= +15 V
  -5e11 <= Nox <= +5e11 cm^-2
  Additionally, the Green2D centerline-admissibility test must pass.

Fixed geometry/materials:
  Drift length L=1.6 um; silicon HALF-width a=0.5 um; 300 K.
  Oxide treated as an effective Robin boundary.

Physics and regression
----------------------
  -Delta phi = q*Nd/eps_si, on 0<x<L, 0<y<a.
  phi(0,y)=0, phi(L,y)=VD, dphi/dy(x,0)=0.
  phi(x,a)+b*dphi/dy(x,a)=VFP+q*Nox*tox/eps_ox.
  b=(eps_si/eps_ox)*tox.

  Effective bias: Veff=VFP+4.63977645634*tox_um*(Nox/1e11).
  Electrostatic solution is the 2D Green-function sine/hyperbolic series.
  The impact-ionization integral is evaluated along the centerline y=0,
  using 300-K original van Overstraeten-de Man electron/hole coefficients.
  Root criterion I_n=1 with centerline monotonic-field/potential checks.
  The reference Fig. 2 calculation is reproduced at Nd=3e16, tox=.25:
  VBD=47.6041694 V with high-resolution quadrature.

  Reduced feature vector:
     u=(tox_um-.4)/.35
     z=Veff/33
     d=(Nd/1e16-4)/4

  VBD_fit=poly8(u,z,d) + local_TPS_residual(u,z,d).
  The polynomial contains 165 coefficients. Residual interpolation uses a
  thin-plate-spline kernel with 120 neighbors and smoothing=0.0001.

  Four named physical inputs are accepted, but the boundary makes VFP and
  Nox degenerate at fixed tox: they cannot be separately identified from VBD
  in this model without additional physics or measurements.

Accuracy against the underlying approximate model
-------------------------------------------------
  Training: 50,589 combinations; 33,774 admissible roots.
  Four-input off-grid holdout: 900 cases, 612 valid, 288 invalid.
  R^2=0.9999912894; RMSE=0.0133206 V; MAE=0.0027361 V.
  95th percentile abs error=0.0072959 V; max error=0.3123546 V.
  Independent effective-bias stress test: 700 cases, 470 valid.
  Stress RMSE=0.0276160 V; maximum error=0.4632442 V.

  Errors increase near certain limits of the valid branch; the returned
  estimate is NOT an experimental accuracy guarantee.

Usage
-----
  Requires Python 3, numpy, scipy; fitting also requires sklearn, matplotlib.
  Run the regression in the directory containing the model files:

  python vbd_4input_predict.py --nd 1e16 --tox 0.25 --vfp 0 --nox 0
  python vbd_4input_predict.py --nd 3e16 --tox 0.25 --vfp 0 --nox 0 --exact

  --exact also computes the Green-function ionization root to compare with
  the regression. Invalid branches return null VBD, NOT 0 or infinity.

Files
-----
  vbd_4input_physics.py      Model's Green-function/ionization equations
  vbd_4input_fit.py          Full regression/data/figure reproduction
  vbd_4input_finalize.py     Fast rebuild of plots/model metrics from data
  vbd_4input_predict.py      Prediction + domain validity checks
  vbd_4input_model.npz       Poly coefficients and local residual data
  vbd_4input_model.json      Human-readable metadata and coefficients
  vbd_4input_training.csv    Training data, valid and invalid branch cases
  vbd_4input_holdout.csv     Two off-grid validation sets with residuals
  vbd_4input_metrics.json    Accuracy and fit metadata
  vbd_4input_regression_sweeps.png
  vbd_4input_regression_validation.png
  vbd_4input_regression_report.pdf (two pages)

Limitations
-----------
  This is a mathematically well-specified 2D electrostatic benchmark but
  assumes FULL depletion. It is NOT self-consistent drift-diffusion/avalanche
  TCAD, does not solve a depletion free boundary, does not model rounded
  field-plate tips, and uses only a centerline ionization path. Especially at
  high doping or strong interface bias, no admissible root is not evidence of
  physical infinite breakdown. Validate against detailed geometry or TCAD
  before using for device engineering decisions.
