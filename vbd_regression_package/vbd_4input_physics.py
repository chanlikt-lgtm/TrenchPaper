"""Fully depleted 2D Green half-cell + centerline avalanche benchmark.

Model geometry: L=1.6 um, silicon half-width a=0.5 um, 300 K.
Boundary conditions: phi(0,y)=0, phi(L,y)=VD, dphi/dy(x,0)=0,
phi(x,a)+b*dphi/dy(x,a)=VFP+q*Nox*tox/eps_ox, b=eps_si*tox/eps_ox.
Valid centerline branches are tested for phi<=VD and dphi/dx>=0.
This is NOT a self-consistent 2D avalanche drift-diffusion simulation.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid, trapezoid

EPS0=8.8541878128e-14
EPS_S=11.7*EPS0
EPS_OX=3.9*EPS0
Q=1.602176634e-19
L=1.6e-4
A=0.5e-4
TOX_BOUNDS=(0.05,0.75)
VFP_BOUNDS=(-15.,15.)
NOX_BOUNDS=(-5e11,5e11)
ND_BOUNDS=(1e15,1.6e17)
KCHARGE=Q*1e11*1e-4/EPS_OX  # 4.639776... V per (um * 1e11 cm^-2)

class Green2D:
    def __init__(self,tox_um,nmodes=120,nx=1001):
        self.tox_um=float(tox_um)
        self.tox=self.tox_um*1e-4
        self.b=EPS_S*self.tox/EPS_OX
        self.x=np.linspace(0.,L,nx)
        n=np.arange(1,nmodes+1)
        mu=n*np.pi/L
        vn=2*(-1.)**(n+1)/(n*np.pi)
        wn=2*L**2*(1-(-1.)**n)/(n*np.pi)**3
        sn=2*(1-(-1.)**n)/(n*np.pi)
        decay=np.exp(-mu*A)
        R0=2*decay/(1+self.b*mu+(1-self.b*mu)*decay**2)
        sine=np.sin(np.outer(mu,self.x))
        cosine=np.cos(np.outer(mu,self.x))
        self.f=self.x/L-np.dot(vn*R0,sine)
        self.h=.5*self.x*(L-self.x)-np.dot(wn*R0,sine)
        self.j=np.dot(sn*R0,sine)
        self.fx=np.full_like(self.x,1/L)-np.dot(vn*mu*R0,cosine)
        self.hx=L/2-self.x-np.dot(wn*mu*R0,cosine)
        self.jx=np.dot(sn*mu*R0,cosine)
        if np.min(self.fx)<=0:
            raise ArithmeticError('Invalid Green-function longitudinal basis')

    def effective_voltage(self,vfp,nox_cm2):
        return float(vfp)+KCHARGE*self.tox_um*(float(nox_cm2)/1e11)

    def field(self,vd,nd,veff):
        return vd*self.fx+(Q*nd/EPS_S)*self.hx+veff*self.jx

    def phi(self,vd,nd,veff):
        return vd*self.f+(Q*nd/EPS_S)*self.h+veff*self.j

    def lower_bound(self,nd,veff):
        k=Q*nd/EPS_S
        bphi=k*self.h+veff*self.j
        bE=k*self.hx+veff*self.jx
        vfield=float(np.max(-bE/self.fx))
        ratio=bphi[:-1]/(1-self.f[:-1])
        vover=float(np.max(ratio))
        vlim=float(-bE[-1]/self.fx[-1])
        return max(0.,vfield,vover,vlim)

    def ion(self,e):
        F=np.maximum(e,1e-30)
        an=7.03e5*np.exp(-1.231e6/F)
        ap=np.where(F<4e5,1.582e6*np.exp(-2.036e6/F),6.71e5*np.exp(-1.693e6/F))
        C=cumulative_trapezoid(ap-an,self.x,initial=0.)
        return float(trapezoid(an*np.exp(np.clip(C,-700,700)),self.x))

    def admissibility(self,nd,veff):
        vmin=self.lower_bound(nd,veff)
        imin=self.ion(self.field(vmin,nd,veff))
        return vmin,imin

    def solve_veff(self,nd,veff):
        vmin,imin=self.admissibility(nd,veff)
        if imin>=1.:
            return np.nan,'no_centerline_root',vmin,imin
        high=max(40.,vmin+20.)
        while self.ion(self.field(high,nd,veff))<1.:
            high=1.4*high+10.
            if high>1e4:
                return np.nan,'unbracketed_root',vmin,imin
        root=brentq(lambda voltage:self.ion(self.field(voltage,nd,veff))-1.,vmin,high,xtol=1e-9,rtol=1e-12)
        return root,'valid',vmin,imin

    def solve(self,nd,vfp=0.,nox_cm2=0.):
        return self.solve_veff(nd,self.effective_voltage(vfp,nox_cm2))


def check_inputs(nd,tox_um,vfp,nox_cm2):
    for name,val,(lo,hi) in (
        ('nd_cm3',nd,ND_BOUNDS),('tox_um',tox_um,TOX_BOUNDS),
        ('vfp_V',vfp,VFP_BOUNDS),('nox_cm2',nox_cm2,NOX_BOUNDS)):
        if not lo <= float(val) <= hi:
            raise ValueError(f'{name} must be in [{lo:g},{hi:g}] (got {val:g})')

