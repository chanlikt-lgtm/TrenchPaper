# VBD vs oxide charge using the 2D Ec criterion

This package contains the requested sweep of breakdown voltage versus **uniform oxide/interface charge** using the full **2D critical-field** criterion.

## Fixed assumptions
- Donor concentration: Nd = 1.00e+16 cm^-3
- Critical field: Ec = 0.30 MV/cm
- Corner exclusion: 100 nm
- Geometry: L = 1.25 um, a = 0.425 um
- Permittivities: eps_si = 11.7 eps0, eps_ox = 3.9 eps0

## Sweeps
- Oxide charge Nox: -1.50e+12 to 1.50e+12 cm^-2
- Oxide thickness tox: 0.05, 0.10, 0.20 um
- Field-plate voltages VFP: -10 V and +10 V

## Files
- `plots/Vbd_vs_oxide_charge_2D_Ec.png` and `.pdf`: combined two-panel plot.
- `plots/Vbd_vs_oxide_charge_2D_Ec_VFP_-10V.*`, `plots/Vbd_vs_oxide_charge_2D_Ec_VFP_+10V.*`: separate panels.
- `data/Vbd_vs_oxide_charge_2D_Ec.csv`: numerical data.
- `data/assumptions.json`: assumptions and summary.
- `calculate_vbd_vs_oxide_charge.py`: reproducible script.

## Note
NaN values mean the zero-drain field already reaches or exceeds Ec somewhere in the evaluated 2D silicon domain.
