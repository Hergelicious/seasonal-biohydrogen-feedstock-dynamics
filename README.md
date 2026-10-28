# Lagged Seasonal Deficit of Biodegradable Organics in a Hot-Climate Sewer

## Overview

This repository contains the analysis code and supporting data for the study *Lagged Seasonal Deficit of Biodegradable Organics in a Hot-Climate Sewer*. The study investigates seasonal variations in biodegradable organic matter in a municipal wastewater system, examines the temporal relationships between temperature and wastewater characteristics, and evaluates potential temperature-driven processes in hot-climate sewers.

The computational workflow reproduces the study's reported analyses, including seasonal diagnostics, lagged relationships, temperature-response assessments, sewer-process modelling, robustness checks, and monthly wastewater design-factor calculations.

## Repository Structure

```text
project/
├── data/
│   ├── ismailia.csv
│   ├── era5_openmeteo.csv
│   └── hannovil.xlsx
├── analysis.py
├── era5_check.py
├── sewer_model.py
├── supp_checks.py
├── design_factors.py
├── run_all.py
└── requirements.txt
```

## Data Sources

### 1. Ismailia wastewater monitoring data

**File:** `data/ismailia.csv`

Operational monitoring data from the Ismailia Sewage Treatment Plant covering 2000–2008. These observations provide the primary dataset for investigating seasonal wastewater characteristics and their relationships with temperature.

**Citation**

Farghaly, A. (2026). *Ismailia Sewage Treatment Plant operational monitoring dataset, 2000–2008*. Mendeley Data, Version 1.

https://doi.org/10.17632/bzppxc3ddx.1

**Licence:** CC BY 4.0, as specified by the source dataset.

### 2. ERA5 air temperature data

**File:** `data/era5_openmeteo.csv`

Daily mean 2-m air temperature data for the study area (30.50° N, 32.25° E), obtained through the Open-Meteo Historical Weather API using ERA5 reanalysis data.

**Sources**

- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. https://doi.org/10.1002/qj.3803
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

**Data provenance and licensing:** The file contains temperature data retrieved through Open-Meteo from ERA5. Refer to the applicable dataset and provider terms when redistributing the data. The original ERA5 attribution and the applicable Open-Meteo licence or terms should be retained with the distributed data.

### 3. Hannovil wastewater monitoring data

**File:** `data/hannovil.xlsx`

Monthly laboratory monitoring data from the Hannovil (Alagamy) Wastewater Treatment Plant covering 2008–2009. These data are used for independent comparative analyses.

**Data provenance and permissions:** The dataset originates from laboratory measurements associated with a master's thesis. Before redistributing this file, verify the relevant permissions, ownership, and any applicable thesis or dataset restrictions. Include the appropriate source citation and acknowledgement. If permission for public redistribution cannot be established, exclude the file from the public repository and document the procedure for obtaining access separately.

## Requirements

The analysis has been developed and tested using the following Python environment:

- Python 3.13
- NumPy 2.5
- pandas 3.0
- SciPy 1.18
- statsmodels 0.15
- Matplotlib 3.11

Install the required dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The versions listed above describe the reported environment. For strict reproducibility, use the dependency versions recorded in `requirements.txt`.

## Running the Analysis

Run the complete analysis pipeline from the repository root:

```bash
python run_all.py
```

The workflow typically completes in approximately one minute on a standard laptop, depending on the hardware and computational workload.

The `run_all.py` script coordinates the individual analysis modules. Run it from the repository root so that the scripts can locate the input files using their expected paths.

## Analysis Workflow

The workflow comprises the following modules.

| Script | Description |
|---|---|
| `run_all.py` | Coordinates the analysis pipeline and executes the component scripts in the prescribed order. |
| `analysis.py` | Calculates daily load series, seasonal climatology, seasonal phase relationships, lag analyses, season-adjusted temperature tests, dissolved oxygen and solids diagnostics, Ramadan event analyses, robustness assessments, and Hannovil comparative statistics. |
| `era5_check.py` | Evaluates consistency between plant temperature observations and ERA5 air temperature data, performs ERA5-based lag analyses, and conducts season-adjusted temperature tests. |
| `sewer_model.py` | Implements sewer-temperature transfer modelling, kinetics-only and biomass-memory process models, parameter estimation, sensitivity analyses, out-of-sample validation, and per-capita consistency checks. |
| `supp_checks.py` | Performs additional diagnostic analyses and validation checks supporting the interpretation of dissolved oxygen behaviour and model timing characteristics. |
| `design_factors.py` | Calculates monthly wastewater feedstock design factors and associated uncertainty estimates. |

## Methodological Notes

### Temperature data quality

A sensor fault affects the plant temperature record during 2004, followed by a measurement shift. ERA5-derived air temperature is therefore used for the inferential temperature analyses described in the study. The comparison between plant measurements and ERA5 data is implemented in `era5_check.py`.

### Seasonal and event-based analyses

The analysis includes seasonal climatology, lagged associations, and season-adjusted temperature tests. Ramadan event-study dates are specified directly in the relevant analysis scripts.

### Sewer-process modelling

The sewer model evaluates temperature transfer and alternative process representations, including kinetics-only and biomass-memory formulations. Model fitting, sensitivity analyses, and out-of-sample validation are implemented in `sewer_model.py`.

### Population-based calculations

Population-based calculations assume that the wastewater system serves Ismailia City. Census-based population estimates are used, with interpolation between census years where required.

### Comparative plant analysis

Hannovil monitoring data are used for comparative analyses. Their inclusion in the public repository is conditional on confirmation of the relevant data-sharing permissions.

## Reproducibility

The workflow is designed to reproduce the analytical results reported in the associated study.

- Random seeds are fixed for permutation tests and Monte Carlo procedures where applicable.
- Analysis steps are implemented in Python scripts rather than relying solely on manual calculations.
- Input datasets and processing steps are identified in the repository structure and analysis modules.

Exact numerical reproducibility depends on the input data, dependency versions, execution environment, and deterministic behaviour of the underlying numerical libraries.

## Citation

If you use this repository or adapt its analysis code, please cite the associated study:

> Hassan, H. H. *Lagged Seasonal Deficit of Biodegradable Organics in a Hot-Climate Sewer*. Manuscript submitted to the *Journal of Water Process Engineering*.

Replace this provisional citation with the final bibliographic information when the study is published. If the repository receives a DOI, cite the archived release as well.

## Licence


The original analysis code in this repository is released under the MIT License. See the `LICENSE` file for the full licence text.

Third-party datasets remain subject to their respective licences, terms of use, attribution requirements, and applicable permissions. The MIT License does not extend to third-party data unless explicitly authorized by the relevant rights holders.
## Contact

For questions regarding the analysis code or reproducibility of the study, please contact the corresponding author or open an issue in this repository.
