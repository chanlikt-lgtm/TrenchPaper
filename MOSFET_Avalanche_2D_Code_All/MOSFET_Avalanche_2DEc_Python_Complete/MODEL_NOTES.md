# Source Figure 2 replacement: same mathematical model

Based on `Green_Function_Breakdown_Report (8).pdf`, page 7, Figure 2 and its equations on pages 3-6.

**Breakdown criterion:** centerline avalanche integral `I_n=1` using 300 K van Overstraeten-de Man rates. This is NOT the 2D global `max |E| = Ec` criterion.

**Electrostatics:** source 0, drain V, symmetry `phi_y(x,0)=0`, sidewall Robin `phi_y(x,a)+phi(x,a)/b=0`, and uniform depleted donor concentration. Source field is plotted at `(x=0,y=0)`. Drain-side *monitor* is `(x=L-0.1 um,y=a)`, safely away from the ideal corner. These are monitoring locations, **not verified physical breakdown hotspots**.

The supplied Figure 2's small rise before the maxima is retained. The curves are recalculated from the source report's Green-function electrostatics and avalanche integral, not a fitted smooth curve. At a=0.5 um, the source tables specify approximate peaks 48.83 V / 2.233e16 for the 40-V family and 70.67 V / 2.627e16 for the 60-V family; reproduced values are in the CSV. All plotted high-doping terminations follow the model's centerline admissibility conditions. No global corner breakdown, partial depletion, or 2D avalanche path is solved.

Reference PDF: user-provided `Green_Function_Breakdown_Report (8).pdf`.
