# SN1987A Neutrino MCMC Analysis

**📦 [Download the Complete Dataset](https://github.com/rzvincent/sn1987_neutrino_mcmc/releases/download/data_v1.0/Bin.zip)** 
Click the link above to download the full archive of our MCMC results, the 2D simulation spectra from [David Vartanyan](https://dvartany.github.io/data/), and additional data required for plotting.

---

## File Descriptions

### Python Scripts (`.py`)
* **`Data.py`**: Event data from the Kamiokande-II, IMB, and Baksan detectors.
* **`definition.py`**: Globally adopted physical properties (e.g., $C\nu B$ temperature $T_\nu$) and useful customized functions. This includes a Gauss–Legendre integration implementation, which significantly speeds up the MCMC process without sacrificing precision. Setting precision=64 is sufficient for most calculations, though this value can be increased if higher accuracy is needed.
* **`Errors.py`**: Gaussian error models for energy.
* **`HK_sm.py`**: Calculates the 95% C.L. of the mock Hyper-Kamiokande sample using a profile-likelihood-ratio test.
* **`mcmc_2d.py` & `mcmc_2d_osc.py`**: Runs MCMC chains without and with neutrino oscillations, respectively, based on the 2D Vartanyan23 model.
* **`mcmc_8p.py` & `mcmc_8p_osc.py`**: Runs MCMC chains without and with neutrino oscillations, respectively, based on the analytic Pagliaroli09 model.
* **`neutrino_IS.py`**: Calculates the neutrino scattering rate in an Inverse Seesaw model.
* **`neutrino_TPI.py`**: Calculates the neutrino scattering rate in a Type-I Seesaw model.
* **`SignalRate_*.py`**: Calculates neutrino fluxes and signal rates on detectors across various models and scenarios.
* **`Simulation_Spectrum.py`**: Interpolates the 2D neutrino flux based on the simulated supernova spectra.

### Jupyter Notebooks (`.ipynb`)
* **`2d_sim_all.ipynb`**: Generates the 22 contour plots for the Vartanyan23 model in a single figure (Figs. 9–12).
* **`contrast_plot.ipynb`**: Maps the $m_\phi$ - $\lambda_{\phi\nu}$ parameter space and plots our constraints (Fig. 5).
* **`corner_plot.ipynb`**: Generates the 8-parameter MCMC contour plots (Figs. 6–8).
* **`flux_osc_IS.ipynb` & `flux_osc_TPI.ipynb`**: Plots integrated flux curves for the Inverse Seesaw and Type-I Seesaw models.(Fig. 3)  *(Note: Verify that `Seesaw_Inverse` is set to `True` or `False` in `definition.py` before running).*
* **`HK_sm_IS.ipynb` & `HK_sm_TPI.ipynb`**: Simulates mock events for Hyper-Kamiokande and calculates the subsequent 95% bounds.
* **`scat_rate_IS.ipynb` & `scat_rate_TPI.ipynb`**: Calculates scattering rates across a grid of different $m_\phi$ and $m_\nu$ values. (Fig. 2)
* **`stacked_*.ipynb`**: Plots 2-parameter contours (Fig. 4), stacking the constraints from the Vartanyan23 model with 22 independent mass spectra constraints.

### Data Files
* **`m_lambda_space_2d_*.npy`**: $\log(m_\phi)$ vs. $\log(\lambda_{\phi\nu})$ data points calculated from 95% C.L. bounds based on the mock Hyper-Kamiokande sample.
* **`samples_HK_*.npy`**: The generated Hyper-Kamiokande mock sample.
* **`scat_rate_*`**: Scattering rates calculated over a fixed $m_\phi$ grid, used for generating Fig. 2.
* **`smellycat_*`**: Raw MCMC chain output files.

---

If you have any questions regarding the code, the datasets, or the underlying physics, or if you have suggestions for improvement, please feel free to reach out!
