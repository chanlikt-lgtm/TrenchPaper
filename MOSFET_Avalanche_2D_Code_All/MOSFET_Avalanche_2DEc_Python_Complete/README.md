# Green-function trench MOSFET: avalanche sensitivity + 2D Ec comparison

**Self-contained Python sources, numerical data, full Figure 2 sweeps, quantitative proof, and reference documents.** No coefficients were fitted to the Figure 2 curves. The ionization coefficients are specified material-model parameters, not predictions of the electrostatic Green function.

## First steps

From the directory containing this README:

```bash
python -m pip install -r requirements.txt
python run_sensitivity_analysis.py
```

This regenerates `data/Sensitivity_Analytical_vs_Numerical.csv`, `data/Sensitivity_Profiles.csv`, and the two `figures/Avalanche_Field_Redistribution_and_Sensitivity.*` plots. It also verifies the analytic breakdown-voltage slopes against finite differences of the full ionization integral.

For the **full, longer Figure 2 sweep**, run:

```bash
python reproduce_source_fig2.py
```

That script regenerates the 40-V and 60-V geometry families (five silicon half-widths each), the source/drain electric-field vectors evaluated at corresponding avalanche voltages, and CSV files. It may be substantially slower than the sensitivity calculation.

The **separate, earlier 2D critical-field reference model** is provided under `comparison_2d_Ec/`:

```bash
python comparison_2d_Ec/run_2d_vbd.py --output comparison_2d_Ec/recomputed --points 40
```

**Do not compare the resulting VBD numbers directly** without matching geometry and boundary forcing. This separate code uses L=1.25 µm, a=0.425 µm, `Ec=30 V/µm`, and a numerical corner exclusion, whereas Figure 2 uses L=1.6 or 3.0 µm and a centerline ionization integral. Neither independently validates physical trench-corner breakdown.

## Reproduced centerline-avalanche benchmarks

Silicon half-width a=0.5 µm in both families; oxide thicknesses W_I = 0.25 and 0.40 µm:

| Family | n = ND / 1e16 cm^-3 | Voltage VBD (V) | dVBD/dn (V) | dI/dn source | dI/dn drain |
|---|---:|---:|---:|---:|---:|
| 40-V, before | 1.000000 | 45.549021 | +4.67115 | +0.017671 | -0.189077 |
| 40-V, maximum | 2.249265 | 48.834114 | 0 | +0.135169 | -0.135169 |
| 40-V, after | 3.400000 | 46.041568 | -4.83714 | +0.252468 | -0.087503 |
| 60-V, before | 1.500000 | 63.174525 | +10.20225 | +0.014447 | -0.343002 |
| 60-V, maximum | 2.652933 | 70.676793 | 0 | +0.246710 | -0.246710 |
| 60-V, after | 3.400000 | 65.404904 | -19.40488 | +0.451378 | -0.130127 |

The peak positions above solve the derivative condition continuously. The source report lists slightly different **sampled** maxima from its 251-point density grid.

## Contents

- `reproduce_source_fig2.py` — original Green-function series, ionization integral, Figure 2 geometry sweeps, and illustrative field-vector samples.
- `avalanche_sensitivity.py` — analytic derivatives of the avalanche integral with respect to applied voltage and doping, including source/drain signed contributions.
- `run_sensitivity_analysis.py` — computes continuous peaks, checks finite differences, and writes detailed CSV/plots.
- `data/` — complete Figure 2 curves, monitor-point field vectors, six-case sensitivity results and longitudinal profiles.
- `figures/` — Figure 2 curves, vector panels, zoom of the small pre-peak rise, and field/ionization-sensitivity profiles in PNG/PDF.
- `reference/` — user-supplied `Green_Function_Breakdown_Report (8).pdf` and the corresponding one-page report (PDF/PPTX).
- `comparison_2d_Ec/` — **separately labelled** earlier 2D peak-field model, with its own reference derivation PDF.
- `DERIVATION.md` — full derivative formulas and caveats.

## Important limitations

- The centerline criterion is **integrated avalanche multiplication along an assumed 1D path**, using a 2D Green-function electrostatic potential. It is not global 2D impact-ionization prediction.
- The second model instead declares threshold when the sampled 2D peak field first reaches a prescribed critical field `Ec`; it uses artificial corner exclusion and a different geometry.
- Both require full depletion, local Robin oxide approximation, fixed geometry, and the stated boundary conditions. The `W_s=0.5 µm` family results have not been calibrated to an actual MOSFET.
- A source-side/oxide-side vector sample in the Figure 2 visualization is **not established as the global physical avalanche hotspot**. Avoid interpreting the sharp 2D Ec hotspot switch as a property of the Figure 2 centerline avalanche calculation.
