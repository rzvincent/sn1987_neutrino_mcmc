# Pending
* **Click [Here](https://github.com/rzvincent/sn1987_neutrino_mcmc/releases/download/data_v1.0/Bin.zip)**: to download the overall collection of our mcmc results, the 2d simulation spectrum from [David Vartanyan](https://dvartany.github.io/data/), as well as other effective plotting data.

Below is the description of each file:

Python:
* Data.py: The events from Kamiokande-II, IMB and Baksan
* definition.py: The globally adopted properties like CvB temperature T_nu, as well as some useful customized functions
* Errors.py: The Gaussian error of Energy
* HK_sm.py: Calculating 95% C.L of mock Hyper-K sample according to profile-likelihood-ratio test
* mcmc_2d.py and mcmc_2d_osc.py: Running mcmc chains with and without oscillation, based on the 2D Vartanyan23 model
* mcmc_8p.py and mcmc_8p_osc.py: As above, based on the analytic Pagliaroli09 model.
* neutrino_IS.py: Calculating neutrino scattering rate in an Inverse Seesaw
* neutrino_TPI.py: As above, in a Type-I seesaw
* SignalRate_...: Calculating flux and signal rate on detectors, based on different models and different cases

Notebook:
* 2d_sim_all.ipynb: Generating all 22 contour plots of Vantanyan23 in one figure (Fig.9-Fig.12)
* contrast_plot.ipynb: Generating the $m_\phi$-$\lambda_(\phi\nu)$ parameter space and plotting our results on it (Fig.5)
* corner_plot.ipynb: Ploting 8p contour plots (Fig.6-Fig.8)
* flux_osc_IS.ipynb and flux_osc_TPI.ipynb: Plotting integrated flux curve in an Inverse Seesaw or a Type-I seesaw. Before running the two files, make sure to check the value of "Seesaw_Inverse" in definition.py to be True or False.
* HK_sm_IS.ipynb and HK_sm_TPI.ipynb: Simulating events from Hyper-K and calculating 95% bounds afterwards
* scat_rate_IS.ipynb and scat_rate_TPI.ipynb: Calculating scattering rate with different $m_\phi$ and $m_\nu$.
* stacked...ipynb: ploting 2-pamameter contours, the ones constrained from Vartanyan23 is stacked with 22 independent constraints

Data:
