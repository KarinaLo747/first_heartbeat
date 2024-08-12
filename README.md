# first_heartbeat

Analysing fluorescent data obtained from live imaging of 8.0 dpc GCaMP-positive mouse embryo hearts.
1. 320 frames (~11 fps) were taken per embryo
1. On ImageJ, 6 ROIs (RL, RI, RM, LM, LI, LL) were drawn on the cardiac crescent.
1. Mean fluorescence was obtained for each ROI per frame over the imaging duration.
1. $t_{1/2}$ was determined for each ROI by finding the midpoint between the peak and left base (see Methods Section 2.4 for details).
1. Direction: determined by the combination of $\Delta{t_{1/2}} (L)$ ($t_{1/2}(LM) - t_{1/2}(LL)$) and $\Delta{t_{1/2}} (R)$ ($t_{1/2}(RM) - t_{1/2}(LRL)$) values.
1. Frequency: Equation used ($1/t_{1/2}$), with $t_{1/2}$ values of RI and LI. The pre- and post-bisection frequency and frequency range values were compared to look for any change in rhythmicity between both the left and right.

Run `<exp_num>-manual_peak_pick.ipynb` to manually define `prominence` and `rel_height` arguements when finding peaks and left bases for calculating $t_{1/2}$. One example for embryo 4 (`E4`) has been shown.

Run `load-peak_pick_params.ipynb` to load the manually defined `prominence` and `rel_height` arguements and to run the full analysis. The results CSV file is saved under `data/processed/dataset/`.