import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the validated 2D-Ec model implementation
import importlib.util
spec = importlib.util.spec_from_file_location('run2d', '/mnt/data/TrenchMOSFET_2D_Vbd_Package/run_2d_vbd.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def chart_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':13,
        'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'legend.fontsize':9,'figure.dpi':140,'savefig.dpi':220,
        'axes.grid':True,'grid.alpha':0.18,'lines.linewidth':2.2
    })

out = Path('/mnt/data/TrenchMOSFET_2D_Vbd_vs_OxideCharge')
plots = out/'plots'
data = out/'data'
plots.mkdir(parents=True, exist_ok=True)
data.mkdir(parents=True, exist_ok=True)
chart_style()

# Baseline assumptions (explicitly stated since user did not specify these)
Nd_cm3 = 1.0e16
Ec_MV_cm = 0.30  # inherited from source model script
cutoff_um = 0.10
nox_values = np.linspace(-1.5e12, 1.5e12, 121)  # cm^-2
vfps = [-10.0, 10.0]
tox_values = [0.05, 0.10, 0.20]
colors = ['#176B9A','#C86D20','#288C7D']
labels = ['0.05 µm','0.10 µm','0.20 µm']

# Precompute geometries by tox
geoms = {tox: mod.geometry(tox, nx=111, ny=47, modes=420) for tox in tox_values}

rows = []
summary = {}
for vfp in vfps:
    summary[str(vfp)] = {}
    for tox in tox_values:
        ys = []
        xcrit = []
        ycrit = []
        precrit = []
        for nox in nox_values:
            r = mod.compute_threshold(geoms[tox], Nd_cm3, fp_volts=vfp, nox_cm2=nox, cutoff_um=cutoff_um)
            ys.append(r['Vbd'])
            xcrit.append(r['xcrit'])
            ycrit.append(r['ycrit'])
            precrit.append(int(r['already_critical']))
            rows.append({
                'Nd_cm3': Nd_cm3,
                'Ec_MV_cm': Ec_MV_cm,
                'corner_exclusion_nm': int(round(cutoff_um*1000)),
                'tox_um': tox,
                'VFP_V': vfp,
                'Nox_cm2': float(nox),
                'Vbd_V': r['Vbd'],
                'xcrit_um': r['xcrit'],
                'ycrit_um': r['ycrit'],
                'precritical_at_zero_VD': int(r['already_critical']),
                'Veff_V': r['Veff'],
                'k_V_um2': r['k_V_um2'],
                'zero_bias_max_V_um': r['zero_bias_max_V_um'],
            })
        summary[str(vfp)][str(tox)] = {
            'min_Vbd': float(np.nanmin(np.array(ys, dtype=float))) if np.any(np.isfinite(ys)) else None,
            'max_Vbd': float(np.nanmax(np.array(ys, dtype=float))) if np.any(np.isfinite(ys)) else None,
            'precritical_count': int(np.sum(np.array(precrit, dtype=int))),
        }

# Save CSV
csv_path = data/'Vbd_vs_oxide_charge_2D_Ec.csv'
with csv_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Save assumptions
assumptions = {
    'description':'2D critical-field study of VBD versus uniform oxide/interface charge using the previously validated Green-function model',
    'fixed_parameters': {
        'Nd_cm3': Nd_cm3,
        'Ec_MV_cm': Ec_MV_cm,
        'corner_exclusion_nm': int(round(cutoff_um*1000)),
        'L_um': mod.L_UM,
        'a_um': mod.A_UM,
        'eps_si_rel': 11.7,
        'eps_ox_rel': 3.9,
        'mode_count': 420,
        'grid_nx': 111,
        'grid_ny': 47,
    },
    'swept_parameters': {
        'Nox_cm2_min': float(nox_values.min()),
        'Nox_cm2_max': float(nox_values.max()),
        'num_points': int(len(nox_values)),
        'tox_um': tox_values,
        'VFP_V': vfps,
    },
    'notes': [
        'The domain uses the full 2D critical-field criterion, not the centerline-only criterion.',
        'A 100 nm corner exclusion is used to avoid singular fields at ideal rectangular mixed-boundary corners.',
        'NaN Vbd means the zero-drain field already reaches or exceeds Ec in the evaluated domain.',
        'No fitting parameters are used.'
    ],
    'summary': summary,
}
(data/'assumptions.json').write_text(json.dumps(assumptions, indent=2))

# Plot: two panels, one per VFP, with tox families
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, constrained_layout=True)
for ax, vfp in zip(axs, vfps):
    for color, tox, label in zip(colors, tox_values, labels):
        sel = [r for r in rows if math.isclose(r['VFP_V'], vfp) and math.isclose(r['tox_um'], tox)]
        x = np.array([r['Nox_cm2']/1e12 for r in sel], dtype=float)
        y = np.array([r['Vbd_V'] for r in sel], dtype=float)
        ax.plot(x, y, color=color, label=f'tox = {label}')
    ax.set_title(f'VFP = {vfp:+.0f} V', loc='left', fontweight='bold')
    ax.set_xlabel('Uniform oxide charge Nox (×10¹² cm⁻²)')
    ax.set_xlim(float(nox_values.min()/1e12), float(nox_values.max()/1e12))
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', framealpha=0.92)
    ax.text(0.02, 0.97, f'Nd = 1.0×10¹⁶ cm⁻³\nEc = 0.30 MV/cm\ncorner exclusion = 100 nm',
            transform=ax.transAxes, va='top', fontsize=8.7, color='#526473')
axs[0].set_ylabel('2D critical-field threshold VBD (V)')
fig.suptitle('Trench MOSFET: VBD vs oxide charge using the 2D Ec criterion', fontsize=14, fontweight='bold')
fig.savefig(plots/'Vbd_vs_oxide_charge_2D_Ec.png', bbox_inches='tight')
fig.savefig(plots/'Vbd_vs_oxide_charge_2D_Ec.pdf', bbox_inches='tight')
plt.close(fig)

# Also create separate plots per VFP for easier reading
for vfp in vfps:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for color, tox, label in zip(colors, tox_values, labels):
        sel = [r for r in rows if math.isclose(r['VFP_V'], vfp) and math.isclose(r['tox_um'], tox)]
        x = np.array([r['Nox_cm2']/1e12 for r in sel], dtype=float)
        y = np.array([r['Vbd_V'] for r in sel], dtype=float)
        ax.plot(x, y, color=color, label=f'tox = {label}')
    ax.set_title(f'2D Ec criterion: VBD vs oxide charge  |  VFP = {vfp:+.0f} V', fontweight='bold')
    ax.set_xlabel('Uniform oxide charge Nox (×10¹² cm⁻²)')
    ax.set_ylabel('2D critical-field threshold VBD (V)')
    ax.set_xlim(float(nox_values.min()/1e12), float(nox_values.max()/1e12))
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', framealpha=0.92)
    ax.text(0.02, 0.97, f'Nd = 1.0×10¹⁶ cm⁻³\nEc = 0.30 MV/cm\ncorner exclusion = 100 nm',
            transform=ax.transAxes, va='top', fontsize=8.7, color='#526473')
    fig.savefig(plots/f'Vbd_vs_oxide_charge_2D_Ec_VFP_{int(vfp):+d}V.png', bbox_inches='tight')
    fig.savefig(plots/f'Vbd_vs_oxide_charge_2D_Ec_VFP_{int(vfp):+d}V.pdf', bbox_inches='tight')
    plt.close(fig)

# Simple markdown summary
readme = f"""# VBD vs oxide charge using the 2D Ec criterion

This package contains the requested sweep of breakdown voltage versus **uniform oxide/interface charge** using the full **2D critical-field** criterion.

## Fixed assumptions
- Donor concentration: Nd = {Nd_cm3:.2e} cm^-3
- Critical field: Ec = {Ec_MV_cm:.2f} MV/cm
- Corner exclusion: {int(round(cutoff_um*1000))} nm
- Geometry: L = {mod.L_UM} um, a = {mod.A_UM} um
- Permittivities: eps_si = 11.7 eps0, eps_ox = 3.9 eps0

## Sweeps
- Oxide charge Nox: {nox_values.min():.2e} to {nox_values.max():.2e} cm^-2
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
"""
(out/'README.md').write_text(readme)

# Copy calculation script into package
script_copy = out/'calculate_vbd_vs_oxide_charge.py'
script_copy.write_text(Path(__file__).read_text())

print('Created package at', out)
print('Main plot:', plots/'Vbd_vs_oxide_charge_2D_Ec.png')
