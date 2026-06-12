Ferrari:
Fast ECAL Rough Reconstruction and Analysis from Raw Inputs

See Wiki!

This branch is tuned to work on ch185 (eta/phi 18/6) for energy reconstruction, using the "Multifit" (no pileup) for energy and timing
It shows 1.6% resolution at 30GeV

```
tree->Draw("Sum$(ecal_lsfit_amp* (abs(ecal_ieta_within_5x5) < 2  &&  abs(ecal_iphi_within_5x5) < 2 ))>>h(1000, -10, 990)", "abs(ecal_ieta_centroid-18) < 0.2 && abs(ecal_iphi_centroid-6) < 0.2 && ecal_charge_seed/ecal_charge_sum_5x5 > 0.7")
```
