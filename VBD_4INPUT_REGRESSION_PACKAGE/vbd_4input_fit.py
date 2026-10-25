#!/usr/bin/env python3
"""Train and verify a four-input VBD regression on fresh Green2D calculations.

1) Generate an efficient 3D design in tox, Veff, ND (four physical inputs
   map to three independent electrostatic variables).
2) Fit an eighth-degree polynomial; correct its residual locally using
   a cubic radial-basis interpolation with 80 nearest training samples.
3) Evaluate on TWO independent off-grid sets: random four-parameter
   Latin hypercube, and expanded Veff-range stress test.
4) Export full data, model, plots, and verification metrics.

Run: OPENBLAS_NUM_THREADS=1 python vbd_4input_fit.py
Requires numpy, scipy, scikit-learn, matplotlib.
"""
from pathlib import Path
import json,csv,time,os
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from vbd_4input_physics import Green2D,KCHARGE,ND_BOUNDS,TOX_BOUNDS,VFP_BOUNDS,NOX_BOUNDS

OUT=Path(__file__).resolve().parent
S=21428
NM_TRAIN=120
NX_TRAIN=1001
NM_HOLD=140
NX_HOLD=1201
DEGREE=8
N_LOCAL=120


def coords(a):
    # Matrix columns: tox_um, Veff_V, ND_cm^-3, VBD_V, etc.
    return np.column_stack([(a[:,0]-.4)/.35,a[:,1]/33.,(a[:,2]/1e16-4.)/4.])


def revert_effective(tox,veff):
    # One legal representative from a continuum of VFP/Nox combinations.
    vfp=float(np.clip(veff,-15,15))
    nox=(veff-vfp)/(KCHARGE*tox)*1e11
    return vfp,nox


def all_records():
    tlevels=np.linspace(*TOX_BOUNDS,21)
    gleves=np.linspace(-1,1,33)
    ndlevels=np.geomspace(*ND_BOUNDS,73)
    records=[]
    for i,t in enumerate(tlevels):
        green=Green2D(t,NM_TRAIN,NX_TRAIN)
        gmax=15+5*KCHARGE*t
        for gfrac in gleves:
            veff=float(gfrac*gmax)
            vfp,nox=revert_effective(t,veff)
            for nd in ndlevels:
                vbd,status,vmin,imin=green.solve_veff(nd,veff)
                records.append([t,veff,nd,vbd,vfp,nox,status,vmin,imin,'train'])
        if i in [0,5,10,15,20]:print(f'generated {i+1}/{len(tlevels)} tox slices',flush=True)
    return records


def hold_records(n=900,kind='independent4d',seed=S):
    d=4 if kind=='independent4d' else 3
    sample=qmc.LatinHypercube(d=d,seed=seed).random(n)
    records=[]
    for s in sample:
        t=.05+.7*s[0]
        nd=10.**(15.+np.log10(1.6e17/1e15)*s[-1])
        if kind=='independent4d':
            vfp=-15.+30*s[1]
            nox=-5e11+1e12*s[2]
            veff=vfp+KCHARGE*t*(nox/1e11)
        else:
            veff=(2*s[1]-1)*(15+5*KCHARGE*t)
            vfp,nox=revert_effective(t,veff)
        green=Green2D(t,NM_HOLD,NX_HOLD)
        vbd,status,vmin,imin=green.solve_veff(nd,veff)
        records.append([t,veff,nd,vbd,vfp,nox,status,vmin,imin,kind])
        if len(records)%200==0:print(f'{kind} {len(records)}/{n}',flush=True)
    return records


def export_csv(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def model_predict(model,a):
    poly=model['poly']
    XC=coords(a)
    coarse=poly.transform(XC)@model['coef']
    correction=model['rbf'](XC)
    return coarse+correction


def err_metrics(df):
    ok=np.isfinite(df[:,3].astype(float))
    v=df[ok,3].astype(float)
    pred=df[ok,10].astype(float)
    err=pred-v
    return dict(all_count=int(len(df)), valid_count=int(ok.sum()),
                invalid_count=int((~ok).sum()),
                r2=float(r2_score(v,pred)),rmse_V=float(np.sqrt(np.mean(err**2))),
                mae_V=float(np.mean(np.abs(err))),p95_abs_error_V=float(np.quantile(np.abs(err),.95)),
                max_abs_error_V=float(np.max(np.abs(err))))


def add_prediction(rows,model):
    arr=np.array(rows,dtype=object)
    data=np.array([[float(r[0]),float(r[1]),float(r[2]),float(r[3])] for r in rows],float)
    pred=np.full(len(rows),np.nan)
    ok=np.isfinite(data[:,3])
    pred[ok]=model_predict(model,data[ok])
    extended=[]
    for i,r in enumerate(rows):
        extended.append(list(r)+[float(pred[i]) if ok[i] else np.nan,
                                 (float(pred[i])-float(r[3])) if ok[i] else np.nan])
    return np.array(extended,dtype=object)


def curve_data(t,fp,nox,model,nd_axis):
    green=Green2D(t,200,2001)
    actual=[]; predicted=[]
    for nd in nd_axis:
        val,status,vmin,imin=green.solve(nd,fp,nox)
        actual.append(val)
        if np.isfinite(val):
            g=green.effective_voltage(fp,nox)
            row=np.array([[t,g,nd,val]])
            predicted.append(float(model_predict(model,row)[0]))
        else:predicted.append(np.nan)
    return np.array(actual),np.array(predicted)


def build_figures(model,test1,test2,metrics):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.3,
                         'axes.titlesize':12.2,'axes.labelsize':10.7,'pdf.fonttype':42})
    nd_axis=np.geomspace(*ND_BOUNDS,125)
    fig,axes=plt.subplots(2,2,figsize=(14.4,9.1))
    colors=['#1971A6','#D27B24','#13866F']
    configs=[
        ('A   Oxide thickness',[(.05,0.,0.,'0.05 um'),(.25,0.,0.,'0.25 um'),(.75,0.,0.,'0.75 um')]),
        ('B   Field-plate bias',[(.25,-15.,0.,'-15 V'),(.25,0.,0.,'0 V'),(.25,15.,0.,'+15 V')]),
        ('C   Fixed oxide sheet charge',[(.25,0.,-5e11,'-5e11 cm$^{-2}$'),(.25,0.,0.,'0'),(.25,0.,5e11,'+5e11 cm$^{-2}$')])]
    for ax,(title,items) in zip(axes.flat[:3],configs):
        for i,(t,fp,nox,label) in enumerate(items):
            actual,predict=curve_data(t,fp,nox,model,nd_axis)
            ax.plot(nd_axis/1e16,predict,color=colors[i],lw=2.3,label=label)
            good=np.where(np.isfinite(actual))[0]
            plotids=good[::max(1,len(good)//11)]
            ax.scatter(nd_axis[plotids]/1e16,actual[plotids],color=colors[i],s=18,marker='o',facecolor='white',zorder=5)
            if len(good):
                ax.plot(nd_axis[good[-1]]/1e16, actual[good[-1]],marker='x',markersize=7,markeredgewidth=1.3,color=colors[i])
        ax.set_xscale('log');ax.set_xlim(.1,16)
        ax.set_xlabel(r'$N_D$ ($10^{16}$ cm$^{-3}$)')
        ax.set_ylabel(r'$V_{BD}$ (V)')
        ax.set_title(title,loc='left',weight='bold')
        ax.grid(alpha=.23,which='both');ax.legend(fontsize=9,loc='best',framealpha=.96)
        ax.spines[['top','right']].set_visible(False)
    ax=axes[1,1]
    for df,color,label in [(test1,'#1971A6','4-input holdout'),(test2,'#E18A2A','Veff stress holdout')]:
        ok=np.isfinite(df[:,3].astype(float))
        ax.scatter(df[ok,3].astype(float),df[ok,10].astype(float),s=8,color=color,alpha=.3,label=label)
    ax.plot([20,55],[20,55],ls='--',color='#444B53',lw=1.3,label='Exact agreement')
    ax.set_xlim(20,55);ax.set_ylim(20,55)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Green2D VBD (V)');ax.set_ylabel('Regression VBD (V)')
    ax.set_title('D   Independent validation',loc='left',weight='bold')
    ax.grid(alpha=.23);ax.legend(fontsize=9,loc='upper left')
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Four-input breakdown regression | 2D Green-function reference',x=.075,y=.977,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.941,'Inputs: $N_D$, $t_{ox}$, $V_{FP}$ and $N_{ox}$  |  Fixed: silicon half-width a = 0.5 um; drift length L = 1.6 um; 300 K',color='#475569',fontsize=10.5)
    fig.text(.075,.072,'Lines: polynomial + local RBF residual regression  |  Open circles: Green2D ionization roots  |  x: last admissible computed point',fontsize=9.4,color='#475569')
    fig.text(.075,.047,'No invalid-domain extrapolation. Model is centerline avalanche in a fully depleted rectangular benchmark; not calibrated TCAD.',fontsize=9.2,color='#475569')
    fig.subplots_adjust(left=.075,right=.973,top=.881,bottom=.165,wspace=.25,hspace=.35)

    fig2,axes=plt.subplots(2,2,figsize=(14.4,9.1))
    q=test1[np.isfinite(test1[:,3].astype(float))]
    e=q[:,11].astype(float)
    sc=axes[0,0].scatter(q[:,2].astype(float),e,c=q[:,0].astype(float),s=16,alpha=.55,cmap='viridis')
    axes[0,0].axhline(0,lw=1,color='#344055')
    axes[0,0].set_xscale('log');axes[0,0].set_title('A   Holdout residual vs. doping',loc='left',weight='bold')
    axes[0,0].set_xlabel(r'$N_D$ (cm$^{-3}$)');axes[0,0].set_ylabel('Regression - Green2D (V)')
    c=fig2.colorbar(sc,ax=axes[0,0],fraction=.045,pad=.03);c.set_label('Oxide thickness (um)')
    axes[0,0].grid(alpha=.23)
    axes[0,1].hist(np.abs(e),bins=50,color='#3281AD',edgecolor='none')
    axes[0,1].set_title('B   Absolute prediction error',loc='left',weight='bold')
    axes[0,1].set_xlabel('Absolute holdout error (V)');axes[0,1].set_ylabel('Count')
    axes[0,1].axvline(metrics['four_input_holdout']['p95_abs_error_V'],color='#CC7733',ls='--',lw=1.5,label='95th percentile')
    axes[0,1].legend()
    # physical admissibility map at baseline tox .25, g=0
    model_s=Green2D(.25,160,1401)
    nds=np.geomspace(*ND_BOUNDS,120)
    veffs=np.linspace(-24,24,111)
    status=np.zeros((len(veffs),len(nds)),dtype=float)
    for gi,g in enumerate(veffs):
        for ni,nd in enumerate(nds):
            vmin,imin=model_s.admissibility(nd,g)
            status[gi,ni]=int(imin<1.)
    mesh=axes[1,0].pcolormesh(nds,veffs,status,cmap=matplotlib.colors.ListedColormap(['#D7DFE8','#D1EBE3']),shading='auto',vmin=0,vmax=1)
    axes[1,0].set_xscale('log')
    axes[1,0].set_title('C   Model admissibility | tox=0.25 um',loc='left',weight='bold')
    axes[1,0].set_ylabel('Effective field-plate bias (V)');axes[1,0].set_xlabel(r'$N_D$ (cm$^{-3}$)')
    axes[1,0].text(.04,.05,'Green: valid centerline root\nGray: no valid centerline root',transform=axes[1,0].transAxes,fontsize=9,va='bottom',bbox=dict(boxstyle='round,pad=.4',fc='white',ec='none',alpha=.95))
    axes[1,1].axis('off')
    met=metrics['four_input_holdout'];met2=metrics['effective_bias_stress_holdout']
    lines=[
        'HOLDOUT VALIDATION  |  Independent 4-input test',
        f"Valid {met['valid_count']} of {met['all_count']} independent samples",
        f"R-squared: {met['r2']:.7f}",
        f"RMSE: {met['rmse_V']:.4f} V",
        f"MAE: {met['mae_V']:.4f} V",
        f"95th percentile absolute error: {met['p95_abs_error_V']:.4f} V",
        f"Maximum absolute error: {met['max_abs_error_V']:.4f} V",
        '',
        'ADDITIONAL EFFECTIVE-BIAS STRESS TEST',
        f"Valid {met2['valid_count']} of {met2['all_count']} samples",
        f"RMSE: {met2['rmse_V']:.4f} V",
        f"Maximum error: {met2['max_abs_error_V']:.4f} V",
        '',
        'NB: Invalid domain is checked using the Green model,',
        'not assigned infinite or extrapolated VBD.'
    ]
    axes[1,1].text(.04,.93,'\n'.join(lines),ha='left',va='top',fontfamily='DejaVu Sans Mono',fontsize=11,linespacing=1.55,
                   bbox=dict(boxstyle='round,pad=.8',fc='#F5F7FA',ec='#DCE4EB'))
    fig2.suptitle('Validation and admissibility of the four-input regression',x=.075,y=.977,ha='left',fontsize=18,fontweight='bold')
    fig2.text(.075,.94,'Test sets are off-grid. Invalid inputs are never used as numerical zeroes or infinite breakdown voltages.',color='#475569',fontsize=10.5)
    fig2.subplots_adjust(left=.075,right=.96,top=.88,bottom=.085,wspace=.27,hspace=.33)
    return fig,fig2


def main():
    clock=time.perf_counter()
    check=Green2D(.25,240,2001).solve(3e16,0.,0.)[0]
    assert abs(check-47.604169)<.00005, f'Fig2 benchmark mismatch: {check}'
    print('Reference Fig2 benchmark:',check,flush=True)
    train=all_records()
    train_data=np.asarray([[float(a),float(b),float(c),float(d)] for a,b,c,d,*_ in train])
    valid=np.isfinite(train_data[:,3])
    fit=train_data[valid]
    P=PolynomialFeatures(degree=DEGREE,include_bias=True)
    X=P.fit_transform(coords(fit))
    coeff=np.linalg.lstsq(X,fit[:,3],rcond=1e-10)[0]
    residual=fit[:,3]-X@coeff
    rbf=RBFInterpolator(coords(fit),residual,neighbors=N_LOCAL,kernel='thin_plate_spline',degree=-1,smoothing=0.0001)
    model={'poly':P,'coef':coeff,'rbf':rbf}
    print(f'Fit {len(fit)} valid samples; polynomial terms {len(coeff)}; time {time.perf_counter()-clock:.2f}s',flush=True)
    hold1=hold_records(n=900,kind='independent4d',seed=S)
    hold2=hold_records(n=700,kind='veff_stress',seed=S+1)
    print('holdouts generated',flush=True)
    pred1=add_prediction(hold1,model)
    print('holdout1 predicted',flush=True)
    pred2=add_prediction(hold2,model)
    print('holdout2 predicted',flush=True)
    mt1=err_metrics(pred1)
    mt2=err_metrics(pred2)
    metrics={'reference_fig2_at_3e16_V':check,
             'training_count':len(train),'training_valid_count':int(valid.sum()),
             'polynomial_degree':DEGREE,'polynomial_terms':len(coeff),
             'rbf_neighbors':N_LOCAL,'rbf_kernel':'thin_plate_spline',
             'four_input_holdout':mt1,'effective_bias_stress_holdout':mt2,
             'fixed_a_um':.5,'fixed_L_um':1.6,
             'caveat':'2D Green electrostatics + centerline avalanche; uniform full depletion and effective oxide boundary assumed.'}
    print(json.dumps(metrics,indent=2),flush=True)
    # train file has no explicit surrogate in every row; full training data provided
    export_csv(OUT/'vbd_4input_training.csv',
               ['tox_um','Veff_V','Nd_cm3','Green2D_VBD_V','VFP_V','Nox_cm2','status','min_admissible_VD_V','ionization_at_min_VD','split'],train)
    allhold=np.vstack([pred1,pred2])
    export_csv(OUT/'vbd_4input_holdout.csv',
               ['tox_um','Veff_V','Nd_cm3','Green2D_VBD_V','VFP_V','Nox_cm2','status','min_admissible_VD_V','ionization_at_min_VD','split','regression_VBD_V','regression_minus_Green2D_V'],allhold.tolist())
    np.savez_compressed(OUT/'vbd_4input_model.npz',
        train_X=coords(fit),train_residual_V=residual,
        poly_coef_V=coeff,poly_powers=P.powers_,
        bounds=np.array([[*ND_BOUNDS],[*TOX_BOUNDS],[*VFP_BOUNDS],[*NOX_BOUNDS]]),
        poly_degree=DEGREE,rbf_neighbors=N_LOCAL,rbf_smoothing=0.0001)
    short={
        'name':'2D Green-function centerline avalanche VBD four-input surrogate',
        'input_units':{'Nd':'cm^-3','tox':'um','VFP':'V','Nox':'signed cm^-2'},
        'input_bounds':{'Nd':list(ND_BOUNDS),'tox':list(TOX_BOUNDS),'VFP':list(VFP_BOUNDS),'Nox':list(NOX_BOUNDS)},
        'effective_boundary_voltage':'Veff_V = VFP_V + KCHARGE*tox_um*(Nox_cm-2/1e11)',
        'KCHARGE_V_per_um_per_1e11_cm2':KCHARGE,
        'reduced_feature_order':['(tox_um-0.4)/0.35','Veff_V/33','(Nd_cm-3/1e16-4)/4'],
        'regression':'degree-8 multivariate polynomial + local thin-plate-spline residual correction',
        'model_npz':'vbd_4input_model.npz',
        'not_an_extrapolator':True,
        'validity':'check Green2D lower_bound and ionization_at_lower_bound < 1 before predicting',
        'polynomial_powers':P.powers_.tolist(),
        'polynomial_coefficients_V':[float(a) for a in coeff],
        'metrics':metrics}
    (OUT/'vbd_4input_model.json').write_text(json.dumps(short,indent=2))
    # save printable report
    fig,fig2=build_figures(model,pred1,pred2,metrics)
    fig.savefig(OUT/'vbd_4input_regression_sweeps.png',dpi=190)
    fig2.savefig(OUT/'vbd_4input_regression_validation.png',dpi=190)
    with PdfPages(OUT/'vbd_4input_regression_report.pdf') as pdf:
        pdf.savefig(fig);pdf.savefig(fig2)
    plt.close(fig);plt.close(fig2)
    print('DONE',time.perf_counter()-clock,'seconds',flush=True)

if __name__=='__main__':main()
