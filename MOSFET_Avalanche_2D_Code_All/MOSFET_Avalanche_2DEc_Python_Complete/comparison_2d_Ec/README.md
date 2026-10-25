# Separate 2D Ec model (different geometry; not direct Figure 2 comparator)

Usage:

```
python run_2d_vbd.py --output recalculated --points 40
```

This is the earlier sample calculation: rectangular silicon L=1.25 um, a=0.425 um, fixed Ec=30 V/um, oxide thickness/charge/field-plate sweeps, and artificial corner exclusions at 50/100/150 nm. The critical voltage is the first positive V_D making any sampled *nonexcluded* point reach the chosen Ec. It is not the centerline avalanche voltage defined by the uploaded report and uses a different physical geometry. No fitting is used.
