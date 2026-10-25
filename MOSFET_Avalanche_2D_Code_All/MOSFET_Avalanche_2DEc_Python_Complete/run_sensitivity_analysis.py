#!/usr/bin/env python3
"""Reproduce the quantitative explanation for Figure 2's pre-peak VBD rise.

Run: python run_sensitivity_analysis.py
Outputs: data/Sensitivity_* CSV and figures/Avalanche_* PNG/PDF.
This is a centerline avalanche calculation, NOT a 2D peak-Ec criterion.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reproduce_source_fig2 import GreenAvalanche
from avalanche_sensitivity import sensitivity, check_derivatives, ND_UNIT_CM3

ROOT = Path(__file__).resolve().parent

FAMILIES = [
    dict(name='40-V family', oxide_um=0.25, L_um=1.6, a_um=0.5,
         low_n=1.0, peak_bracket=(2.10, 2.45), high_n=3.4),
    dict(name='60-V family', oxide_um=0.40, L_um=3.0, a_um=0.5,
         low_n=1.5, peak_bracket=(2.45, 2.85), high_n=3.4)
]


def peak_by_zero_derivative(model, bracket):
    """Solve dI/dn|V=Vbd(n) = 0. It is NOT a fitted polynomial peak."""
    def derivative_at_threshold(n):
        vbd = model.threshold(n * ND_UNIT_CM3)
        if not np.isfinite(vbd):
            raise RuntimeError(f'Non-admissible n in peak root: {n}')
        return sensitivity(model, n * ND_UNIT_CM3, vbd)['dIdn']
    return brentq(derivative_at_threshold, *bracket, xtol=1e-10)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    (ROOT/'figures').mkdir(parents=True, exist_ok=True)
    (ROOT/'data').mkdir(parents=True, exist_ok=True)
    summary_rows, profile_rows = [], []
    colors = ['#276DB3', '#E07B39', '#B64C62']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), constrained_layout=True)
    for r, p in enumerate(FAMILIES):
        model = GreenAvalanche(p['oxide_um'], p['L_um'], p['a_um'], modes=240, points=4001)
        n_star = peak_by_zero_derivative(model, p['peak_bracket'])
        states = [('Before peak',p['low_n']), ('Exact maximum',n_star), ('After peak',p['high_n'])]
        for c,(label,n) in enumerate(states):
            nd = n * ND_UNIT_CM3
            vbd = model.threshold(nd)
            z = sensitivity(model, nd, vbd)
            derivative_check = check_derivatives(model, nd, vbd)
            fd_slope = (model.threshold((n+0.003)*ND_UNIT_CM3) - model.threshold((n-0.003)*ND_UNIT_CM3))/0.006
            if abs(z['I']-1) > 1e-5:
                raise AssertionError('Ionization condition I=1 failed')
            if abs(z['dIdn']-z['dIdn_source']-z['dIdn_drain']) > 2e-5:
                raise AssertionError('Source + drain weighting check failed')
            if abs(z['slope_V_per_1e16_cm3']-fd_slope) > 0.04:
                raise AssertionError('Analytical slope vs finite difference discrepancy')
            for field in ('dIdV','dIdn'):
                ana=derivative_check[f'analytical_{field}']
                fd=derivative_check[f'finite_difference_{field}']
                if abs(ana-fd) > 0.003 + 0.004*abs(ana):
                    raise AssertionError(f'Integral derivative check failed for {field}: {ana} / {fd}')
            summary_rows.append({
                'family': p['name'], 'state': label, 'a_um': p['a_um'],
                'oxide_um': p['oxide_um'], 'L_um': p['L_um'],
                'Nd_cm3': nd, 'n_Nd_per_1e16': n, 'Vbd_V': vbd,
                'avalanche_Integral_In': z['I'],
                'dIdV_per_V': z['dIdV'], 'dIdn_per_n': z['dIdn'],
                'dIdn_source_half': z['dIdn_source'],
                'dIdn_drain_half': z['dIdn_drain'],
                'dVbd_dn_analytic_V': z['slope_V_per_1e16_cm3'],
                'dVbd_dn_finite_diff_V': fd_slope,
                'F_source_V_um': z['F_source_V_um'],
                'F_drain_V_um': z['F_drain_V_um']
            })
            for i, x in enumerate(z['x_um']):
                profile_rows.append({
                    'family': p['name'], 'state': label, 'n': n,
                    'x_um': float(x), 'F_centerline_V_um': float(z['F_V_um'][i]),
                    'dF_dn_V_um': float(z['donor_response_V_um_per_n'][i]),
                    'dF_dV_V_um_per_V': float(z['drain_response_V_um_per_V'][i]),
                    'avalanche_weight_W': float(z['W'][i]),
                    'dIdn_density_per_um': float(z['dIdn_density_per_um'][i])
                })
            axes[r,0].plot(z['x_um'], z['F_V_um'], lw=2.5, color=colors[c],
                           label=f'{label}, n={n:.3f}, VBD={vbd:.2f} V')
            axes[r,1].plot(z['x_um'], z['dIdn_density_per_um'], lw=2.5, color=colors[c], label=label)
            print(f"{p['name']}: {label:13s} n={n:.6f} VBD={vbd:.6f} V "
                  f"source={z['dIdn_source']:+.6f} drain={z['dIdn_drain']:+.6f} "
                  f"dVbd/dn={z['slope_V_per_1e16_cm3']:+.5f} V")
        for ax in axes[r]:
            ax.axvline(p['L_um']/2, color='#64748B', ls=':', lw=1.3)
            ax.grid(alpha=0.16)
            ax.set_xlabel('Depth x, source to drain (micrometers)')
            ax.set_xlim(0, p['L_um'])
        axes[r,0].set_ylabel('Positive centerline field F (V/micrometer)')
        axes[r,1].set_ylabel('Signed dI/dn density (per micrometer)')
        axes[r,0].set_title(f"{p['name']}: field redistribution", fontweight='bold',loc='left')
        axes[r,1].set_title(f"{p['name']}: avalanche sensitivity", fontweight='bold',loc='left')
        axes[r,0].legend(fontsize=8, loc='best')
        axes[r,1].legend(fontsize=8, loc='best')
        axes[r,1].axhline(0, lw=0.7, color='#475569')
    fig.suptitle('Why VBD initially rises: donor field shifts avalanche weighting from drain toward source',fontsize=14,fontweight='bold')
    fig.savefig(ROOT/'figures'/'Avalanche_Field_Redistribution_and_Sensitivity.png',dpi=240,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/'Avalanche_Field_Redistribution_and_Sensitivity.pdf',bbox_inches='tight')
    plt.close(fig)
    write_csv(ROOT/'data'/'Sensitivity_Analytical_vs_Numerical.csv', summary_rows)
    write_csv(ROOT/'data'/'Sensitivity_Profiles.csv', profile_rows)
    return summary_rows


if __name__ == '__main__': main()
