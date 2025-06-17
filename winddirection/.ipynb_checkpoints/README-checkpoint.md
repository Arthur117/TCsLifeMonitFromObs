# winddirection

- 01_getSARasymmetries
Computes asymmetry parameters on a filtered dataset of (~100) SAR images
Takes ~8 hours to run

- 02_plotKepertAndSARasymmetries
Builds on the previous dataset to plot the SAR asymmetries
Also computes and plots the asymmetry metrics for the Kepert model
Takes ~8 hours to run

- 03: TODO: Fit the K values from Kepert to the SAR data once I have identified the cases (or the case!?) on which I will do it 
i.e look at VWS before;

- bebinca, bheki1, bheki2, leslie, usagi: tests with Kepert and IWRAP;

- larry_z: most advanced notebook containing the up-do-date (non-vectorized) Kepert 2001 functions, and some tests with IWRAP and EarthCARE;

- test_asymmetry_windshear: asymmetry diagnostics from SAR only;

- test_asymmetry_kepert: asymmetry diagnostics from Kepert 2001 only;

- apply_K25: notebook containing the non-vectorized Kepert 2025 functions and the up-to-date dropsondes functions;

- lee_K25: mot up-to-date notebook containing the vectorized Kepert 2025 functions and the up-to-date dropsondes functions; Also contains the fitting procedure (which I probably won't use). Also contains the way to estimate an average Cd independently (from the wind profile). Function get_wn1() was later updated in notebook 00_FIT_K25.ipynb



