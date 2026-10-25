#!/usr/bin/env python3
"""Exact first derivatives of the chosen centerline avalanche functional.

Uses the Green-function electrostatic solution in reproduce_source_fig2.py.
No curve fitting: differentiates the van Overstraeten-de Man electron
avalanche integral with respect to V_D and n=N_D/(10^16 cm^-3).

At avalanche threshold I_n=1, the voltage slope is
  dVbd/dn = - (dI_n/dn) / (dI_n/dV).

Note: F is positive phi_x(x,0) in V/um. The physical vector field E=-grad(phi).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from reproduce_source_fig2 import Q, EPS_S, GreenAvalanche

ND_UNIT_CM3 = 1.0e16
K_PER_N = Q * ND_UNIT_CM3 / EPS_S / 1.0e8  # V/um^2 per unit n


def sensitivity(model: GreenAvalanche, nd_cm3: float, vd_volts: float):
    """Compute avalanche I, analytic first derivatives and full integrands.

    x_um: depth coordinate from source to drain in micrometers.
    dI/dn: ionization response per n=N_D/(10^16 cm^-3).
    dI/dV: ionization response per drain volt.
    W(x): avalanche functional derivative with respect to F(x) in V/um.
    """
    x_um = model.x
    x_cm = x_um * 1.0e-4
    kval = model.k(nd_cm3)            # V/um^2
    F_um = vd_volts * model.fx + kval * model.hx
    if np.min(F_um) < -1.e-8:
        raise ValueError('Negative centerline field: outside the selected admissible branch')
    F_cm = np.maximum(F_um * 1.0e4, 1.0e-8)

    an = 7.03e5 * np.exp(-1.231e6 / F_cm)
    mask_low = F_cm < 4.0e5
    ap_pref = np.where(mask_low, 1.582e6, 6.71e5)
    ap_B = np.where(mask_low, 2.036e6, 1.693e6)
    ap = ap_pref * np.exp(-ap_B / F_cm)

    # Derivatives with respect to F_um, not F_cm:
    dan_dFum = an * 1.231e6 / F_cm**2 * 1.0e4
    dap_dFum = ap * ap_B / F_cm**2 * 1.0e4
    C = cumulative_trapezoid(ap - an, x_cm, initial=0.)
    expC = np.exp(np.clip(C, -650, 650))
    density = an * expC
    I = trapezoid(density, x_cm)

    # J(x) = integral_x^L an[F(t)] exp(C(t)) dt
    J = -cumulative_trapezoid(density[::-1], x_cm[::-1], initial=0.)[::-1]
    W = expC * dan_dFum + J * (dap_dFum - dan_dFum)

    # At threshold I=1 there is a manifestly non-negative equivalent
    # W(x)=an'(F)*int_0^x ap exp(C) dt + ap'(F)*int_x^L an exp(C) dt.
    dF_dV = model.fx
    dF_dn = K_PER_N * model.hx
    dIdV_density_um = W * dF_dV * 1.0e-4  # density per um depth
    dIdn_density_um = W * dF_dn * 1.0e-4
    dIdV = trapezoid(dIdV_density_um, x_um)
    dIdn = trapezoid(dIdn_density_um, x_um)
    mid = model.L / 2
    on_source = x_um <= mid
    on_drain = x_um >= mid
    dIdn_source = trapezoid(dIdn_density_um[on_source], x_um[on_source])
    dIdn_drain = trapezoid(dIdn_density_um[on_drain], x_um[on_drain])

    return dict(
        I=float(I), dIdV=float(dIdV), dIdn=float(dIdn),
        slope_V_per_1e16_cm3=float(-dIdn / dIdV),
        dIdn_source=float(dIdn_source), dIdn_drain=float(dIdn_drain),
        F_source_V_um=float(F_um[0]), F_drain_V_um=float(F_um[-1]),
        x_um=x_um, F_V_um=F_um,
        W=W, donor_response_V_um_per_n=dF_dn,
        drain_response_V_um_per_V=dF_dV,
        dIdn_density_per_um=dIdn_density_um,
        dIdV_density_per_um=dIdV_density_um,
    )


def check_derivatives(model, nd_cm3, vd_volts, delta_n=0.002, delta_v=0.02):
    """Independent central-difference check at a fixed operating point."""
    s = sensitivity(model, nd_cm3, vd_volts)
    k = model.k(nd_cm3)
    i_plus = model.avalanche(vd_volts + delta_v, k)
    i_minus = model.avalanche(vd_volts - delta_v, k)
    dIdV_fd = (i_plus - i_minus) / (2 * delta_v)
    i_plus = model.avalanche(vd_volts, model.k(nd_cm3 + delta_n * ND_UNIT_CM3))
    i_minus = model.avalanche(vd_volts, model.k(nd_cm3 - delta_n * ND_UNIT_CM3))
    dIdn_fd = (i_plus - i_minus) / (2 * delta_n)
    return {"analytical_dIdV": s['dIdV'], "finite_difference_dIdV": dIdV_fd,
            "analytical_dIdn": s['dIdn'], "finite_difference_dIdn": dIdn_fd}
