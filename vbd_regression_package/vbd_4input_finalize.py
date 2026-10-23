#!/usr/bin/env python3
"""Rebuild final model exports and plots from the large precomputed training and off-grid CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vbd_4input_physics import ND_BOUNDS,TOX_BOUNDS,VFP_BOUNDS,NOX_BOUNDS,KCHARGE,Green2D
from vbd_4input_fit import build_figures, err_metrics, coords

OUT=Path(__file__).resolve().parent
N_LOCAL=120
KERNEL='thin_plate_spline'
SMOOTHING=1e-4

def main():
    old=np.load(OUT/'vbd_4input_model.npz')
    X=old['train_X']; r=old['train_residual_V']; c=old['poly_coef_V']; powers=old['poly_powers']
    poly=PolynomialFeatures(8)
    poly.fit(X)
    assert np.array_equal(poly.powers_,powers)
    rbf=RBFInterpolator(X,r,neighbors=N_LOCAL,kernel=KERNEL,degree=-1,smoothing=SMOOTHING)
    model={'poly':poly,'coef':c,'rbf':rbf}
    np.savez_compressed(OUT/'vbd_4input_model.npz',train_X=X,train_residual_V=r,
         poly_coef_V=c,poly_powers=powers,
         bounds=np.array([ND_BOUNDS,TOX_BOUNDS,VFP_BOUNDS,NOX_BOUNDS],dtype=float),
         poly_degree=8,rbf_neighbors=N_LOCAL,rbf_smoothing=SMOOTHING)

    df=pd.read_csv(OUT/'vbd_4input_holdout.csv')
    valid=df['status']=='valid'
    input_rows=df.loc[valid,['tox_um','Veff_V','Nd_cm3','Green2D_VBD_V']].to_numpy(float)
    pred=poly.transform(coords(input_rows))@c+rbf(coords(input_rows))
    df.loc[valid,'regression_VBD_V']=pred
    df.loc[valid,'regression_minus_Green2D_V']=pred-df.loc[valid,'Green2D_VBD_V'].to_numpy(float)
    df.to_csv(OUT/'vbd_4input_holdout.csv',index=False)
    pieces={k:df[df['split']==k].to_numpy(dtype=object) for k in ['independent4d','veff_stress']}
    mt1=err_metrics(pieces['independent4d']);mt2=err_metrics(pieces['veff_stress'])
    bench=Green2D(.25,240,2001).solve(3e16,0,0)[0]
    metrics={
        'model':'degree 8 polynomial + local thin-plate-spline residual',
        'reference_fig2_at_3e16_V':bench,
        'training_count':len(pd.read_csv(OUT/'vbd_4input_training.csv')),
        'training_valid_count':len(X),
        'polynomial_degree':8,'polynomial_terms':len(c),
        'rbf_neighbors':N_LOCAL,'rbf_kernel':KERNEL,'rbf_smoothing':SMOOTHING,
        'four_input_holdout':mt1,'effective_bias_stress_holdout':mt2,
        'fixed_a_um':0.5,'fixed_L_um':1.6,
        'caveat':'2D Green electrostatics + centerline avalanche; uniform full depletion and effective oxide boundary assumed.'}
    print(json.dumps(metrics,indent=2),flush=True)
    (OUT/'vbd_4input_metrics.json').write_text(json.dumps(metrics,indent=2))
    model_json={
      'name':'2D Green-function centerline avalanche four-input VBD surrogate',
      'input_units':{'Nd_cm3':'cm^-3','tox_um':'um','VFP_V':'V','Nox_cm2':'signed cm^-2'},
      'input_bounds':{'Nd_cm3':list(ND_BOUNDS),'tox_um':list(TOX_BOUNDS),'VFP_V':list(VFP_BOUNDS),'Nox_cm2':list(NOX_BOUNDS)},
      'effective_bias':'Veff_V=VFP_V+KCHARGE*tox_um*Nox_cm2/1e11',
      'KCHARGE_V_per_um_per_1e11_cm2':KCHARGE,
      'feature_order':['(tox_um-0.4)/0.35','Veff_V/33','(Nd_cm3/1e16-4)/4'],
      'fit':'8th-degree polynomial + local thin-plate spline residual, 120 neighbors, smoothing=0.0001',
      'poly_powers':powers.tolist(),'poly_coefficients_V':c.tolist(),
      'residual_model':'vbd_4input_model.npz',
      'validity':'Green2D admissibility: I(Vmin)<1; otherwise no result',
      'metrics':metrics,
      'limitations':'Fully depleted rectangle; 2D Green electrostatics, but ionization integral evaluated along 1D centerline. No 2D avalanche transport, no neutral-region free boundary, no realistic field-plate tips.'}
    (OUT/'vbd_4input_model.json').write_text(json.dumps(model_json,indent=2))
    print('making figures',flush=True)
    fig,fig2=build_figures(model,pieces['independent4d'],pieces['veff_stress'],metrics)
    fig.savefig(OUT/'vbd_4input_regression_sweeps.png',dpi=190)
    fig2.savefig(OUT/'vbd_4input_regression_validation.png',dpi=190)
    with PdfPages(OUT/'vbd_4input_regression_report.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    plt.close(fig);plt.close(fig2)
    print('DONE',flush=True)

if __name__=='__main__':main()
