#!/usr/bin/env python3
"""Reproduce corner-excluded 2D Ec-criterion curves from a Green-function MOSFET model.

Model is linear Poisson with constant fully depleted donor density and a local
Robin oxide condition. The corner-exclusion radii are numerical conventions,
not physical trench corner rounding. No fitting or calibration is performed.

Dependencies: numpy, matplotlib. Python 3.9+.
Usage: python run_2d_vbd.py --output results
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = 1.602176634e-19   # C
EPS0_CM = 8.8541878128e-14  # F/cm
EPS_SI = 11.7 * EPS0_CM
EPS_OX = 3.9 * EPS0_CM
L_UM = 1.25
A_UM = 0.425
EC_V_UM = 30.0  # 0.30 MV/cm


def geometry(tox_um, nx=111, ny=47, modes=420):
    """Compute analytic unit-voltage and unit-donor field gradients on grid.

    Each quantity corresponds to the gradient of potential (not minus E),
    so vector field magnitude remains identical. Coordinates in micrometers.
    """
    b_um = (EPS_SI/EPS_OX) * tox_um
    xs = np.linspace(0, L_UM, nx)
    ys = np.linspace(0, A_UM, ny)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    dx = np.full((nx, ny), 1.0 / L_UM, dtype=np.float64)
    dy = np.zeros((nx, ny), dtype=np.float64)
    px = np.zeros((nx, ny), dtype=np.float64)
    py = np.zeros((nx, ny), dtype=np.float64)
    hx = L_UM/2 - xx
    hy = np.zeros((nx, ny), dtype=np.float64)
    for n in range(1, modes+1):
        alpha = n * np.pi / L_UM
        sn = 4.0/(n*np.pi) if n % 2 else 0.0
        rn = 2.0 * ((-1)**n) / (n*np.pi)
        c = np.cos(alpha * xs)[:, None]
        s = np.sin(alpha * xs)[:, None]
        ea = np.exp(-alpha * (A_UM - ys))
        eb = np.exp(-alpha * (A_UM + ys))
        denominator = (1.0+np.exp(-2*alpha*A_UM))*(1+b_um*alpha*np.tanh(alpha*A_UM))
        # cosh(alpha*y)/(b*Delta), evaluated without overflow:
        v = ((ea + eb)/denominator)[None,:]
        v_y = (alpha*(ea-eb)/denominator)[None,:]
        dx += rn*alpha*c*v
        dy += rn*s*v_y
        if sn:
            px += sn*alpha*c*v
            py += sn*s*v_y
            hx -= (sn/alpha)*c*v
            hy -= (sn/(alpha**2))*s*v_y
    return dict(xs=xs,ys=ys,xx=xx,yy=yy,b_um=b_um,dx=dx,dy=dy,px=px,py=py,hx=hx,hy=hy,
                modes=modes,nx=nx,ny=ny,tox_um=tox_um)


def compute_threshold(g, nd_cm3, fp_volts=0.0, nox_cm2=0.0, cutoff_um=0.1, centerline_only=False,
                      return_map=False):
    # Poisson source q Nd/epsilon is V/cm^2; convert to V/um^2.
    k = Q*nd_cm3/EPS_SI/1.0e8
    # sheet-charge boundary forcing in V/um: sigma/epsilon / (um per cm).
    sheet_drive = (Q*nox_cm2/EPS_SI)/1.0e4
    v_eff = fp_volts + g['b_um']*sheet_drive
    x,y = g['xx'],g['yy']
    valid = ((x*x+(A_UM-y)**2 >= cutoff_um**2) &
             ((L_UM-x)**2+(A_UM-y)**2>=cutoff_um**2))
    if centerline_only:
        valid = valid & np.isclose(y, 0.0)
    ax,ay = g['dx'],g['dy']
    ex0 = v_eff*g['px'] + k*g['hx']
    ey0 = v_eff*g['py'] + k*g['hy']
    C = ex0**2 + ey0**2
    existing = (C >= EC_V_UM**2) & valid
    if np.any(existing):
        peak = np.where(valid,np.sqrt(C),-np.inf)
        ij=np.unravel_index(np.argmax(peak),peak.shape)
        return {'Vbd':float('nan'),'xcrit':float('nan'),'ycrit':float('nan'),
                'already_critical':True,'zero_bias_max_V_um':float(peak[ij]),
                'Veff':v_eff, 'k_V_um2':k}
    A = ax*ax+ay*ay
    B = ax*ex0+ay*ey0
    discriminant = np.maximum(0.0,B*B+A*(EC_V_UM**2-C))
    with np.errstate(invalid='ignore',divide='ignore'):
        roots = np.where(valid & (A>1.0e-18),(-B+np.sqrt(discriminant))/A,np.inf)
    roots = np.where(np.isfinite(roots)&(roots>=0),roots,np.inf)
    ij=np.unravel_index(np.argmin(roots),roots.shape)
    vd=float(roots[ij])
    if not np.isfinite(vd):
        return {'Vbd':float('nan'),'xcrit':float('nan'),'ycrit':float('nan'),
                'already_critical':False,'zero_bias_max_V_um':float(np.nan),
                'Veff':v_eff,'k_V_um2':k}
    result={'Vbd':vd,'xcrit':float(x[ij]),'ycrit':float(y[ij]),
            'already_critical':False,'zero_bias_max_V_um':float(np.max(np.sqrt(C[valid]))),
            'Veff':v_eff,'k_V_um2':k}
    if return_map:
        e_x=vd*ax+ex0
        e_y=vd*ay+ey0
        result['field_V_um']=np.sqrt(e_x*e_x+e_y*e_y)
        result['mask']=valid
    return result


def all_cases():
    return [
        ('tox_0.05um',0.05,0,0),('tox_0.10um',0.10,0,0),('tox_0.20um',0.20,0,0),
        ('Nox_negative_1e12',0.10,0,-1e12),('Nox_zero',0.10,0,0),('Nox_positive_1e12',0.10,0,1e12),
        ('VFP_negative_10V',0.10,-10,0),('VFP_zero',0.10,0,0),('VFP_positive_10V',0.10,10,0)]


def study(nd_values=None, cutoffs=(0.05,0.1,0.15), nx=111,ny=47,modes=420):
    if nd_values is None:
        nd_values = np.linspace(1e15,8e16,80)
    cases=all_cases()
    geoms={tox:geometry(tox,nx,ny,modes) for tox in (0.05,0.1,0.2)}
    rows=[]
    for cutoff in cutoffs:
        for nd in nd_values:
            row={'cutoff_um':float(cutoff),'Nd_cm3':float(nd)}
            for name,tox,vfp,nox in cases:
                v=compute_threshold(geoms[tox],nd,vfp,nox,cutoff)
                row[name+'_V']=v['Vbd']
                row[name+'_xcrit_um']=v['xcrit']
                row[name+'_ycrit_um']=v['ycrit']
                row[name+'_precritical_at_zero_VD']=int(v['already_critical'])
            rows.append(row)
    return rows,geoms


def chart_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':13,
                         'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
                         'legend.fontsize':9,'figure.dpi':140,'savefig.dpi':220,
                         'axes.grid':True,'grid.alpha':0.18,'lines.linewidth':2.2})


def plot_groups(rows, cutoff, outdir):
    sel=[r for r in rows if math.isclose(r['cutoff_um'],cutoff)]
    x=np.array([r['Nd_cm3']/1e16 for r in sel])
    spec=[
        ('Group_1_Oxide_Thickness', '1. Oxide thickness', [('tox_0.05um_V','0.05 µm'),('tox_0.10um_V','0.10 µm'),('tox_0.20um_V','0.20 µm')], 'Baseline VFP = 0 V; Nox = 0'),
        ('Group_2_Interface_Charge', '2. Uniform interface charge', [('Nox_negative_1e12_V','−10¹² cm⁻²'),('Nox_zero_V','0'),('Nox_positive_1e12_V','+10¹² cm⁻²')], 'Baseline tox = 0.10 µm; VFP = 0 V'),
        ('Group_3_Field_Plate', '3. Field-plate voltage', [('VFP_negative_10V_V','−10 V'),('VFP_zero_V','0 V'),('VFP_positive_10V_V','+10 V')], 'Baseline tox = 0.10 µm; Nox = 0')
    ]
    palette=['#176B9A','#C86D20','#288C7D']
    fig,axs=plt.subplots(1,3,figsize=(17.5,5.2),sharey=True,constrained_layout=True)
    for ax,(basename,title,fields,subtitle) in zip(axs,spec):
        for idx,(key,label) in enumerate(fields):
            z=np.array([r[key] for r in sel],dtype=float)
            ax.plot(x,z,label=label,color=palette[idx],lw=2.1)
        ax.set_title(title,fontweight='bold',loc='left')
        ax.set_xlabel('Donor concentration Nd (×10¹⁶ cm⁻³)')
        ax.set_xlim(float(x.min()),float(x.max()))
        ax.set_ylim(bottom=0)
        ax.text(0.02,0.96,subtitle,transform=ax.transAxes,fontsize=8.2,va='top',color='#526473')
        ax.legend(loc='upper right',frameon=True,facecolor='white',framealpha=.92)
        f2,a2=plt.subplots(figsize=(8.0,5.1),constrained_layout=True)
        for idx,(key,label) in enumerate(fields):
            z=np.array([r[key] for r in sel],dtype=float)
            a2.plot(x,z,label=label,color=palette[idx],lw=2.4)
        a2.set(title=title,xlabel='Donor concentration Nd (×10¹⁶ cm⁻³)',ylabel='2D critical-field threshold VBD (V)')
        a2.set_xlim(float(x.min()),float(x.max())); a2.set_ylim(bottom=0)
        a2.legend(loc='best',framealpha=.92)
        f2.suptitle('2D Ec = 0.30 MV/cm  |  corner exclusion = '+str(round(cutoff*1000))+' nm',fontsize=11)
        f2.savefig(outdir/(basename+'.png'),bbox_inches='tight')
        f2.savefig(outdir/(basename+'.pdf'),bbox_inches='tight')
        plt.close(f2)
    axs[0].set_ylabel('2D critical-field threshold VBD (V)')
    fig.suptitle('Trench MOSFET: 2D Ec criterion  |  corner exclusion = '+str(round(cutoff*1000))+' nm',fontsize=14,fontweight='bold')
    fig.savefig(outdir/'Three_Groups_2D.png',bbox_inches='tight')
    fig.savefig(outdir/'Three_Groups_2D.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity(rows,outdir):
    fig,axs=plt.subplots(1,3,figsize=(16.5,4.8),constrained_layout=True,sharey=True)
    cases=[('tox_0.10um_V','Baseline oxide thickness'),('Nox_positive_1e12_V','Positive interface charge'),('VFP_positive_10V_V','Positive field plate')]
    colors=['#176B9A','#C86D20','#288C7D']
    for ax,(key,title) in zip(axs,cases):
        for i,cut in enumerate((0.05,0.1,0.15)):
            sel=[r for r in rows if math.isclose(r['cutoff_um'],cut)]
            x=np.array([r['Nd_cm3']/1e16 for r in sel])
            y=np.array([r[key] for r in sel],dtype=float)
            ax.plot(x,y,label=f'{round(cut*1000)} nm',color=colors[i])
        ax.set_title(title,loc='left',fontweight='bold')
        ax.set_xlabel('Nd (×10¹⁶ cm⁻³)')
        ax.set_xlim(x.min(),x.max());ax.set_ylim(bottom=0)
        ax.legend(title='Exclusion radius',loc='upper right')
    axs[0].set_ylabel('2D critical-field threshold VBD (V)')
    fig.suptitle('Sensitivity to artificial corner exclusion (not a physical corner radius)',fontweight='bold',fontsize=14)
    fig.savefig(outdir/'Corner_Exclusion_Sensitivity.png',bbox_inches='tight')
    fig.savefig(outdir/'Corner_Exclusion_Sensitivity.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_field_map(g,cut,nd,outdir):
    r=compute_threshold(g,nd,0,0,cut,return_map=True)
    field=r['field_V_um'].copy()
    field[~r['mask']]=np.nan
    fig,ax=plt.subplots(figsize=(10.2,4.1),constrained_layout=True)
    # x horizontal = silicon depth x, y vertical = transverse position y.
    m=ax.pcolormesh(g['xx'],g['yy'],field/10,shading='auto',cmap='viridis')
    fig.colorbar(m,ax=ax,label='|E| (MV/cm)')
    ax.scatter([r['xcrit']],[r['ycrit']],s=90,facecolor='none',edgecolor='red',linewidth=2.5,
               label=f'Critical position ({r["xcrit"]:.3f}, {r["ycrit"]:.3f}) µm')
    ax.set(xlabel='x: source → drain (µm)',ylabel='y: symmetry → oxide (µm)',title=f'2D field magnitude at VBD = {r["Vbd"]:.3f} V, Nd = 10¹⁶ cm⁻³')
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='lower left')
    fig.savefig(outdir/'Baseline_2D_Field_Map.png',bbox_inches='tight')
    fig.savefig(outdir/'Baseline_2D_Field_Map.pdf',bbox_inches='tight')
    plt.close(fig)
    return {k:v for k,v in r.items() if k not in ('field_V_um','mask')}


def save_csvs(rows,folder):
    cols=list(rows[0].keys())
    for cut in (0.05,0.10,0.15):
        sel=[r for r in rows if math.isclose(r['cutoff_um'],cut)]
        f=folder/f'Vbd_2D_rc_{int(round(cut*1000)):03d}nm.csv'
        with f.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=cols)
            w.writeheader()
            w.writerows(sel)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='recomputed_2d')
    parser.add_argument('--nx',type=int,default=111)
    parser.add_argument('--ny',type=int,default=47)
    parser.add_argument('--modes',type=int,default=420)
    parser.add_argument('--points',type=int,default=80)
    a=parser.parse_args()
    output=Path(a.output); plots=output/'plots'; data=output/'data'
    plots.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    chart_style()
    rows, geoms = study(np.linspace(1e15,8e16,a.points),nx=a.nx,ny=a.ny,modes=a.modes)
    save_csvs(rows,data)
    plot_groups(rows,0.10,plots)
    plot_sensitivity(rows,plots)
    base=plot_field_map(geoms[0.10],0.10,1e16,plots)
    center=compute_threshold(geoms[0.10],1e16,0,0,0.10,True)
    metadata={'model':'fully depleted rectangular silicon + local Robin oxide; 2D Ec criterion',
              'geometry':{'L_um':L_UM,'a_um':A_UM,'eps_si_rel':11.7,'eps_ox_rel':3.9},
              'assumptions':{'Ec_MV_cm':EC_V_UM/100,'baseline_tox_um':0.10,'baseline_Vfp_V':0,
                             'baseline_Nox_cm2':0,'corner_cutoffs_um':[0.05,0.1,0.15],
                             'mode_count':a.modes,'grid_nx':a.nx,'grid_ny':a.ny,
                             'no_fitting':True,'not_physical_rounded_corner':True},
              'baseline_Nd_1e16':{'2D':base,'centerline':center},
              'units':'all fields internally V/um; 1 V/um=0.01 MV/cm',
              'missing_values':'NaN means model is at/above Ec at zero drain voltage'}
    (data/'assumptions_and_baseline.json').write_text(json.dumps(metadata,indent=2,default=lambda x:float(x)))
    print(json.dumps({'2D_VBD_baseline':base['Vbd'],'xcrit':base['xcrit'],'ycrit':base['ycrit'],
                      'centerline_VBD':center['Vbd'],'num_rows':len(rows)},indent=2))

if __name__=='__main__':
    main()
