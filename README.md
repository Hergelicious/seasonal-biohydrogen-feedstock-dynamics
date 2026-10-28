Lagged Seasonal Deficit of Biodegradable Organics in a Hot-Climate Sewer
This repository contains the data and analysis code used to investigate seasonal variations in biodegradable organic matter in a municipal wastewater system and to evaluate temperature-driven sewer processes in a hot-climate environment. The workflow reproduces the analytical results reported in the associated study, including seasonal diagnostics, temperature-response analyses, process-model evaluation, and wastewater design-factor calculations.

Repository Structure
project/
│
├── data/
│   ├── ismailia.csv
│   ├── era5_openmeteo.csv
│   └── hannovil.xlsx
│
├── analysis.py
├── era5_check.py
├── sewer_model.py
├── supp_checks.py
├── design_factors.py
│
└── run_all.py
Requirements
The analysis has been tested with:

Python 3.13
NumPy 2.5
pandas 3.0
SciPy 1.18
statsmodels 0.15
Matplotlib 3.11
Install dependencies using:

pip install -r requirements.txt
Running the Analysis
Execute the complete workflow with:

python run_all.py
The full analysis typically completes in approximately one minute on a standard laptop.

Analysis Workflow
The scripts are executed in the following order by run_all.py.

Script	Description
analysis.py	Calculates daily load series, seasonal climatology, seasonal phase relationships, lag analyses, season-adjusted temperature tests, dissolved oxygen and solids diagnostics, Ramadan event analyses, robustness assessments, and Hannovil plant statistics.
era5_check.py	Evaluates consistency between plant temperature observations and ERA5 air temperature data, performs ERA5-based lag analyses, and conducts season-adjusted temperature tests.
sewer_model.py	Implements sewer-temperature transfer modelling, kinetics-only and biomass-memory process models, parameter estimation, sensitivity analyses, out-of-sample validation, and per-capita consistency checks.
supp_checks.py	Performs additional diagnostic analyses and validation checks supporting the interpretation of dissolved oxygen behaviour and model timing characteristics.
design_factors.py	Computes monthly wastewater feedstock design factors and associated uncertainty estimates.
Data Sources
ismailia.csv
Operational monitoring data from the Ismailia Sewage Treatment Plant covering the period 2000–2008.

Citation

Farghaly, A. (2026). Ismailia Sewage Treatment Plant operational monitoring dataset, 2000–2008. Mendeley Data, Version 1.

https://doi.org/10.17632/bzppxc3ddx.1

Licence: CC BY 4.0

era5_openmeteo.csv
Daily mean ERA5 2 m air temperature data for the study area (30.50°N, 32.25°E), obtained through the Open-Meteo Historical Weather API.

Sources

Hersbach et al. (2020), ERA5 Reanalysis
Open-Meteo Historical Weather API
Licence: CC BY 4.0

hannovil.xlsx
Monthly laboratory monitoring data from the Hannovil (Alagamy) Wastewater Treatment Plant (2008–2009), used for independent comparative analyses.

Please ensure permission has been obtained for public distribution and include the appropriate citation before public release.

Notes
A sensor fault affects the plant temperature record during 2004, followed by a measurement shift. Consequently, ERA5-derived temperature data are used for inferential temperature analyses.
Ramadan dates used in the event-study analyses are specified directly within the analysis scripts.
Population-based calculations in the sewer model assume service to Ismailia City, using census-based population estimates with interpolation between census years.
Reproducibility
All analyses are fully deterministic. Random seeds are fixed wherever permutation testing or Monte Carlo procedures are used, ensuring exact reproducibility of all reported statistical results.
