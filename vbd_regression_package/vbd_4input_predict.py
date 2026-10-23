#!/usr/bin/env python3
"""Four-input VBD regression predictor; non-admissible branches return no result.

Examples:
 python vbd_4input_predict.py --nd 1e16 --tox 0.25 --vfp 0 --nox 0
 python vbd_4input_predict.py --nd 3e16 --tox 0.25 --vfp 15 --nox 5e11 --exact
"""
import argparse, functools
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from vbd_4input_physics import Green2D,KCHARGE,check_inputs

HERE=Path(__file__).resolve().parent

@functools.lru_cache(maxsize=1)
def load():
    data=np.load(HERE/'vbd_4input_model.npz')
    features=data['train_X']
    residual=data['train_residual_V']
    coef=data['poly_coef_V']
    powers=data['poly_powers']
    rbf=RBFInterpolator(features,residual,neighbors=int(data['rbf_neighbors']),kernel='thin_plate_spline',degree=-1,smoothing=float(data['rbf_smoothing']))
    return rbf,coef,powers


def predict(nd_cm3,tox_um,vfp_V,nox_cm2,exact=False):
    """Returns dict with prediction and Green centerline-root admissibility.

    No fake 0, infinity or out-of-training prediction is returned. The lower
    bound and ionization check is analytical-numerical but is NOT a physical
    proof of global depletion or a 2D avalanche TCAD result.
    """
    check_inputs(nd_cm3,tox_um,vfp_V,nox_cm2)
    veff=vfp_V+KCHARGE*tox_um*(nox_cm2/1e11)
    green=Green2D(tox_um,nmodes=160,nx=1501)
    vmin,imin=green.admissibility(nd_cm3,veff)
    if imin>=1.:
        return dict(status='no_admissible_centerline_root',VBD_V=None,
                    Veff_V=veff,VD_min_V=vmin,I_at_VD_min=imin)
    z=np.array([(tox_um-.4)/.35,veff/33,(nd_cm3/1e16-4)/4])
    rbf,coef,powers=load()
    poly=np.sum(coef*np.prod(z[None,:]**powers,axis=1))
    correction=float(rbf(z.reshape(1,-1))[0])
    est=float(poly+correction)
    result=dict(status='valid_centerline',VBD_V=est,
                Veff_V=veff,VD_min_V=vmin,I_at_VD_min=imin,
                note='Regression approximates the specified fully depleted Green2D centerline ionization model, not physical global device breakdown.')
    if imin>.98 or est<vmin:
        result['warning']='Very close to the model-admissibility limit; surrogate may be less accurate; use exact=True.'
    if exact:
        v,status,_,_=green.solve_veff(nd_cm3,veff)
        result['Green2D_exact_VBD_V']=float(v) if np.isfinite(v) else None
        result['surrogate_minus_Green2D_V']=float(est-v) if np.isfinite(v) else None
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Four-input VBD regression for the specified Green2D benchmark')
    p.add_argument('--nd',type=float,required=True,help='donor density cm^-3')
    p.add_argument('--tox',type=float,required=True,help='oxide thickness um')
    p.add_argument('--vfp',type=float,required=True,help='field-plate bias V, relative to source')
    p.add_argument('--nox',type=float,required=True,help='signed oxide sheet-charge density cm^-2')
    p.add_argument('--exact',action='store_true',help='compare against the direct Green2D root')
    args=p.parse_args()
    import json
    print(json.dumps(predict(args.nd,args.tox,args.vfp,args.nox,args.exact),indent=2))
