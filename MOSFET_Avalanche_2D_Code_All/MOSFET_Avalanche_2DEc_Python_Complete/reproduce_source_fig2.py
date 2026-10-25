#!/usr/bin/env python3
"""Reproduce Figure 2 of user-supplied Green_Function_Breakdown_Report (8).pdf.
The potential is its full-depletion rectangular Green-function series; breakdown
is the one-path centerline van Overstraeten-de Man avalanche criterion at 300 K.
This is NOT the earlier global 2D |E|=Ec criterion; no corner hotspot is inferred.
"""
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid, trapezoid

Q=1.602176634e-19
EPS0=8.8541878128e-14   # F/cm
EPS_S=11.7*EPS0
EPS_OX=3.9*EPS0

class GreenAvalanche:
    def __init__(self,wi_um,L_um,a_um, modes=240,points=2001):
        self.wi=wi_um;self.L=L_um;self.a=a_um
        self.b=wi_um*EPS_S/EPS_OX
        self.modes=modes;self.points=points
        self.x=np.linspace(0,L_um,points)
        n=np.arange(1,modes+1)
        self.mu=n*np.pi/L_um
        self.vn=2*(-1.0)**(n+1)/(n*np.pi)
        self.wn=2*L_um**2*(1-(-1.0)**n)/(n*np.pi)**3
        r0=self.transfer(0)
        c=np.cos(np.outer(self.mu,self.x))
        s=np.sin(np.outer(self.mu,self.x))
        self.fx=1/L_um-(self.vn*self.mu*r0)@c
        self.hx=L_um/2-self.x-(self.wn*self.mu*r0)@c
        self.f=self.x/L_um-(self.vn*r0)@s
        self.h=self.x*(L_um-self.x)/2-(self.wn*r0)@s

    def transfer(self,y):
        m=self.mu
        e1=np.exp(-m*(self.a-y));e2=np.exp(-m*(self.a+y))
        den=(1+self.b*m)+(1-self.b*m)*np.exp(-2*m*self.a)
        return (e1+e2)/den

    def transfer_derivative(self,y):
        m=self.mu
        e1=np.exp(-m*(self.a-y));e2=np.exp(-m*(self.a+y))
        den=(1+self.b*m)+(1-self.b*m)*np.exp(-2*m*self.a)
        return m*(e1-e2)/den

    def k(self,nd):
        return Q*nd/EPS_S/1e8

    def minimum_allowed_voltage(self,k):
        mask=(1-self.f)>1e-10
        vmin=max(0.,float(np.max(k*self.h[mask]/(1-self.f[mask]))))
        positive=self.fx>1e-10
        if not np.all(positive):
            # This geometry ordinarily has positive drain response along centerline.
            pass
        vmin=max(vmin,float(np.max(-k*self.hx[positive]/self.fx[positive])))
        return vmin

    def avalanche(self,v,k):
        # V/um -> V/cm
        field=np.maximum((v*self.fx+k*self.hx)*1e4,0.)
        f=np.maximum(field,1e-99)
        an=7.03e5*np.exp(-1.231e6/f)
        ap=np.where(field<4e5,1.582e6*np.exp(-2.036e6/f),6.71e5*np.exp(-1.693e6/f))
        c=cumulative_trapezoid(ap-an,self.x*1e-4,initial=0.)
        return trapezoid(an*np.exp(np.clip(c,-650,650)),self.x*1e-4)

    def threshold(self,nd):
        k=self.k(nd)
        lower=self.minimum_allowed_voltage(k)
        if self.avalanche(lower,k)>=1:
            return float('nan')
        upper=max(40,2*lower)
        for i in range(34):
            if self.avalanche(upper,k)>1:
                break
            upper=upper*1.5+5
        else:
            return float('nan')
        v=brentq(lambda vv:self.avalanche(vv,k)-1,lower,upper,xtol=1e-9)
        if not np.all(v*self.fx +k*self.hx>=-1e-8):return float('nan')
        if not np.all(v*(1-self.f)-k*self.h>=-1e-8):return float('nan')
        return float(v)

    def E(self,nd,v,x,y):
        k=self.k(nd)
        pn=v*self.vn+k*self.wn
        r=self.transfer(y)
        ry=self.transfer_derivative(y)
        px=v/self.L+k*(self.L/2-x)-np.sum(pn*self.mu*r*np.cos(self.mu*x))
        py=-np.sum(pn*ry*np.sin(self.mu*x))
        return np.array([-px,-py],dtype=float)


def main():
    basedir=Path(__file__).resolve().parent
    figs=basedir/'figures';dat=basedir/'data'
    figs.mkdir(exist_ok=True);dat.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':10,
         'axes.labelsize':11,'axes.titlesize':12.5,'legend.fontsize':9,'savefig.dpi':270,
         'axes.spines.right':False,'axes.spines.top':False,'axes.grid':True,'grid.alpha':.21})
    nds=np.logspace(15,np.log10(1.6e17),251)
    a_values=[.2,.3,.5,.7,.9]
    families=[('40-V geometry',.25,1.6),('60-V geometry',.40,3.0)]
    colors={.2:'#5A3480',.3:'#4163A0',.5:'#178F9A',.7:'#20AA69',.9:'#BDCF00'}
    rows=[];case_table={};curves={}
    for family,wi,L in families:
        for a in a_values:
            print('Computing',family,'a',a,flush=True)
            c=GreenAvalanche(wi,L,a)
            vals=[]
            for nd in nds:
                v=c.threshold(nd)
                vals.append(v)
                rows.append(dict(family=family,wi_um=wi,L_um=L,a_um=a,nd_cm3=nd, Vbd_V=v))
            case_table[(family,a)]=c
            curves[(family,a)]=np.array(vals)
            i=np.nanargmax(vals)
            print('peak Nd V',nds[i],vals[i],flush=True)
    with (dat/'Figure2_recomputed_all_widths.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

    # Prepare source-equivalent figure; a=0.5 highlighted and with state circles.
    states={
      '40-V geometry':[(1e16,'Low'),(2.233e16,'Peak'),(3.4e16,'High')],
      '60-V geometry':[(1.5e16,'Low'),(2.627e16,'Peak'),(3.4e16,'High')]
    }
    state_rows=[]
    for family,wi,L in families:
        c=case_table[(family,.5)]
        for nd,phase in states[family]:
            v=c.threshold(nd)
            e_s=c.E(nd,v,0.,0.)
            e_d=c.E(nd,v,L-.10,.5)
            state_rows.append(dict(family=family,phase=phase,a_um=.5,wi_um=wi,L_um=L,
               Nd_cm3=nd,Vbd_V=v,source_Ex_Vum=e_s[0],source_Ey_Vum=e_s[1],
               drain_Ex_Vum=e_d[0],drain_Ey_Vum=e_d[1],
               source_field_magnitude_Vum=np.linalg.norm(e_s),
               drain_field_magnitude_Vum=np.linalg.norm(e_d)))
    with (dat/'Figure2_corresponding_2D_field_vectors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=state_rows[0].keys());w.writeheader();w.writerows(state_rows)

    # First combined plot with 5 half-width series + markers
    fig,axs=plt.subplots(1,2,figsize=(15.8,4.45),sharey=False,constrained_layout=True)
    for ax,(family,wi,L) in zip(axs,families):
        for a in a_values:
            y=curves[(family,a)]
            ax.plot(nds/1e16,y,label=f'$W_s={a:.1f}\,\\mu$m',color=colors[a],lw=3.0 if a==.5 else 1.75,
                    alpha=1 if a==.5 else .9)
            finite=np.where(np.isfinite(y))[0]
            if len(finite):
                j=finite[-1]
                ax.plot(nds[j]/1e16,y[j],marker='x',ms=7, mew=1.6,color=colors[a])
        c=case_table[(family,.5)]
        for nd,phase in states[family]:
            v=c.threshold(nd)
            ax.scatter([nd/1e16],[v],s=55,edgecolor='white',linewidth=1,zorder=5,color=colors[.5])
            if phase=='Low':
                ax.annotate('Before peak',xy=(nd/1e16,v),xytext=(nd/1e16*.80,v-9),fontsize=9,
                    arrowprops=dict(arrowstyle='->',lw=1.2,color='#374151'),color='#374151')
            elif phase=='Peak':
                ax.annotate('Peak',xy=(nd/1e16,v),xytext=(nd/1e16*.77,v+12),fontsize=10,
                    arrowprops=dict(arrowstyle='->',lw=1.2,color='#374151'),color='#374151')
            else:
                ax.annotate('After peak',xy=(nd/1e16,v),xytext=(nd/1e16*.90,v-11),fontsize=9,
                    arrowprops=dict(arrowstyle='->',lw=1.2,color='#374151'),color='#374151')
        ax.set_xscale('log')
        ax.set_xlim(.1,16)
        ax.set_ylim(0,100 if family=='60-V geometry' else 77)
        ax.set_xlabel(r'$N_D$  ($10^{16}$ cm$^{-3}$)')
        ax.set_ylabel('Centerline avalanche $V_{BD}$ (V)')
        ax.set_title(f'{family}   $W_I={wi:.2f}\\,\\mu$m, $L={L:.1f}\\,\\mu$m',fontweight='bold',pad=8)
        ax.legend(ncol=1,loc='lower left',framealpha=.92,fontsize=8)
        ax.grid(which='both',alpha=.16)
    fig.suptitle('Reproduced Figure 2: small pre-peak rise, maximum, then decline',fontsize=15,fontweight='bold')
    fig.savefig(figs/'Figure2_SourceAvalanche_Recomputed.png',bbox_inches='tight')
    fig.savefig(figs/'Figure2_SourceAvalanche_Recomputed.pdf',bbox_inches='tight')
    plt.close(fig)

    # Zoom onto representative curves; this makes slope before max obvious.
    fig,axs=plt.subplots(1,2,figsize=(13.5,3.0),constrained_layout=True)
    for ax,(family,wi,L) in zip(axs,families):
        arr=curves[(family,.5)]
        subset=(nds/1e16>.75)&(nds/1e16<4.2)&np.isfinite(arr)
        ax.plot(nds[subset]/1e16,arr[subset],color=colors[.5],lw=3)
        for nd,phase in states[family]:
            v=case_table[(family,.5)].threshold(nd)
            ax.scatter([nd/1e16],[v],s=50,color=colors[.5],zorder=5)
            ax.annotate(f'{phase}: {v:.2f} V',xy=(nd/1e16,v),xytext=(3,5),textcoords='offset points',fontsize=8.5)
        ax.set_title(f'{family}; half-width $W_s=0.5\\,\\mu$m')
        ax.set_xlabel(r'$N_D$ ($10^{16}$ cm$^{-3}$)')
        ax.set_ylabel('$V_{BD}$ (V)')
        ax.grid(alpha=.18)
    fig.savefig(figs/'PrePeak_Slight_Rise_Zoom.png',bbox_inches='tight')
    plt.close(fig)

    # Six Ex Ey vector panels. NO critical field circle: original VBD is avalanche integral, not peak Ec.
    fig, axs=plt.subplots(2,3,figsize=(15.5,6.25),constrained_layout=True)
    e_s_col='#2563EB';e_d_col='#EA6C27'
    for rr,(family,wi,L) in enumerate(families):
        for cc,(nd,phase) in enumerate(states[family]):
            row=next(p for p in state_rows if p['family']==family and p['phase']==phase)
            ax=axs[rr,cc]
            Es=np.array([row['source_Ex_Vum'],row['source_Ey_Vum']])
            Ed=np.array([row['drain_Ex_Vum'],row['drain_Ey_Vum']])
            ax.annotate('',xy=Es,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=3.3,color=e_s_col,mutation_scale=16))
            ax.annotate('',xy=Ed,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=3.3,color=e_d_col,mutation_scale=16))
            ax.scatter([Es[0],Ed[0]],[Es[1],Ed[1]],color=[e_s_col,e_d_col],s=25,zorder=5)
            ax.axhline(0,color='#8190A0',lw=1)
            ax.axvline(0,color='#8190A0',lw=1)
            ax.set_xlim(-83,8);ax.set_ylim(-9,65)
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'{family}  | {phase.upper()}\n$N_D={nd/1e16:.3f}\\times10^{{16}}$   $V_{{BD}}={row["Vbd_V"]:.2f}$ V',fontsize=10.3)
            ax.set_xlabel('$E_x$ (V/$\\mu$m)');ax.set_ylabel('$E_y$ (V/$\\mu$m)')
            ax.text(.02,.02,f'S ({Es[0]:+.1f},{Es[1]:+.1f})   D ({Ed[0]:+.1f},{Ed[1]:+.1f})',
                transform=ax.transAxes,fontsize=7.8,color='#344054',va='bottom',
                bbox=dict(boxstyle='round,pad=.22',facecolor='white',alpha=.86,edgecolor='none'))
    fig.suptitle('Field-vector evolution at the same avalanche-defined VBD: source vs drain-wall monitor',fontsize=14,fontweight='bold')
    fig.savefig(figs/'Figure2_Corresponding_Ex_Ey_SixPanels.png',bbox_inches='tight')
    fig.savefig(figs/'Figure2_Corresponding_Ex_Ey_SixPanels.pdf',bbox_inches='tight')
    plt.close(fig)

    summary=[]
    for family,wi,L in families:
        r=[p for p in state_rows if p['family']==family]
        summary.append(dict(family=family,low_Nd=r[0]['Nd_cm3'],low_V=r[0]['Vbd_V'],
         peak_Nd=r[1]['Nd_cm3'],peak_V=r[1]['Vbd_V'],
         high_Nd=r[2]['Nd_cm3'],high_V=r[2]['Vbd_V']))
    print('SAMPLE SUMMARY:',summary)
    (basedir/'MODEL_NOTES.md').write_text('''# Source Figure 2 replacement: same mathematical model\n\nBased on `Green_Function_Breakdown_Report (8).pdf`, page 7, Figure 2 and its equations on pages 3-6.\n\n**Breakdown criterion:** centerline avalanche integral `I_n=1` using 300 K van Overstraeten-de Man rates. This is NOT the 2D global `max |E| = Ec` criterion.\n\n**Electrostatics:** source 0, drain V, symmetry `phi_y(x,0)=0`, sidewall Robin `phi_y(x,a)+phi(x,a)/b=0`, and uniform depleted donor concentration. Source field is plotted at `(x=0,y=0)`. Drain-side *monitor* is `(x=L-0.1 um,y=a)`, safely away from the ideal corner. These are monitoring locations, **not verified physical breakdown hotspots**.\n\nThe supplied Figure 2's small rise before the maxima is retained. The curves are recalculated from the source report's Green-function electrostatics and avalanche integral, not a fitted smooth curve. At a=0.5 um, the source tables specify approximate peaks 48.83 V / 2.233e16 for the 40-V family and 70.67 V / 2.627e16 for the 60-V family; reproduced values are in the CSV. All plotted high-doping terminations follow the model's centerline admissibility conditions. No global corner breakdown, partial depletion, or 2D avalanche path is solved.\n\nReference PDF: user-provided `Green_Function_Breakdown_Report (8).pdf`.\n''')

if __name__=='__main__':main()
