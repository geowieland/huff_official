---
title: 'huff: A Python package for Market Area Analysis'

tags:

- Python
- Market area models
- Spatial accessibility
- Location analysis
- Spatial analysis
- Economic geography
- Health geography
- GIS
- Econometrics

authors:
- name: Thomas Wieland
  orcid: 0000-0001-5168-9846
  affiliation: 1

affiliations:
- name: Freiburg, Germany
  index: 1

date: 5 October 2026
bibliography: paper.bib
---


## Summary

Market area models, such as the *Huff Model* and its extensions, are used to estimate regional market shares and customer flows of retail and service locations. In health geography, market area and accessibility models are applied for the analysis of catchment areas and spatial accessibility of healthcare locations. The `huff` Python package provides a complete workflow for market area and spatial accessibility analysis, including data import, construction of origin-destination interaction matrices, basic model analysis, parameter estimation from empirical data, calculation of distance or travel time matrices, and map visualization. The package is modular and object-oriented. It is intended for researchers in economic geography, regional economics, marketing, geoinformation science, and health geography. The software is openly available via the [Python Package Index (PyPI)](https://pypi.org/project/huff/). Its development and version history are managed in a public [GitHub Repository](https://github.com/geowieland/huff_official) and archived at [Zenodo](https://doi.org/10.5281/zenodo.18639559). A methodological handbook is also [available on GitHub Pages](https://geowieland.github.io/huff_official/).


## Statement of need

### Research context

Market area models are used in economic geography, regional economics, geoinformation science, and marketing, enabling the analysis and forecasting of market areas and customer flows for retail and service locations. The classical and most popular approach is the *Huff Model* [@huff1962; @huff1963; @huff1964] and its numerous extensions, such as the *Multiplicative Competitive Interaction (MCI) Model* [@nakanishi1974; @nakanishi1982; @cooper1983]. Typical research applications include examining the influence of marketing or location variables on consumer store choice, forecasting the revenue of new locations, or predicting the impact of new locations on existing ones [@debeule2014; @fittkau2004; @li2012; @mensing2018; @oruc2012; @suarezvega2015; @wieland2015; @wieland2018]. 

In health geography, such models are used to analyse catchment areas with respect to medical practices and hospitals [@bai2023; @fuelop2011; @jia2016; @latruwe2023; @vonrhein2025; @wieland2018]. In both cases, methods for calculating spatial accessibility are used that are structurally related to market area models, such as the *Hansen Accessibility* [@hansen1959] and the *Two-Step Floating Catchment Area (2SFCA) Analysis* [@luo2003]. The two concepts are also increasingly being linked to methods for analyzing the supply structure and accessibility of health locations [@liu2022; @luo2014; @rauch2023; @subal2021]. These models are also applied to other location-related contexts such as airports or recreation facilities [@wang2022; @wang2026]. 

### Challenges in application

There are several major challenges in model-based market area and accessibility analyses:

  - Model calibration based on observed data on consumer behavior and/or store sales is difficult because the models are nonlinear in their weighting parameters [@huff2008; @wieland2017]. For this purpose, the *MCI Model* [@nakanishi1974; @nakanishi1982; @cooper1983] has been developed as an econometric estimation technique. As this approach requires empirical market shares for fitting, it is applied in cases where customer-store interaction data was obtained by surveys or secondary data [@baviera2016; @latruwe2023; @oruc2012; @suarezvega2015; @wieland2015; @wieland2018]. When only total sales of the locations investigated are available, nonlinear iterative fitting approaches may be applied [@debeule2014; @guessefeldt2002; @haines1972; @li2012; @liang2020; @mensing2018; @orpana2003; @wieland2017]. Due to the pronounced sensitivity of market area models to weighting schemes, the availability of multiple calibration approaches is essential in market area analysis.

  - Market area and accessibility models typically include travel times instead of Euclidean distances. Calculating travel times is based on graph theory network analysis and requires real street networks. Therefore, there is a need for GIS (Geographic Information System) support and/or access to an API providing calculations based on input origins and destinations [@huff2008; @miller2015]. It is extremely helpful for researchers if they can also complete this part of the market area analysis workflow within the analysis tool.

  - Researchers must choose and compare appropriate weighting functions and parameters, which may result in substantially different results. For input variables such as travel time, several weighting functions (e.g., power, exponential, logistic) are used, and the model results are compared using goodness-of-fit metrics [@bai2023; @latruwe2023; @li2012; @orpana2003]. It is, thus, necessary that, within the market area analysis workflow, several weighting functions are available, and that researchers may compare different model specifications based on model evaluation metrics.

### Functionality of the `huff` package

The `huff` package for Python v1.9.x essentially provides the following features:

  - *Data management and preliminary analysis*: Users may load customer origins and supply locations from GeoDataFrames, point shapefiles, CSV, or XLSX files. Attributes of customer origins and supply locations (variables, weightings) may be set by the user. The next step is to create an *interaction matrix* with a built-in function, on the basis of which all implemented models can then be calculated. Within an interaction matrix, *transport costs* (distance or travel time between customer origins and supply locations) may be calculated with built-in methods.

  - *Basic Huff Model analysis*: Given an interaction matrix, users may calculate probabilities and expected customer flows with respect to customer origins, and total market areas of supply locations.

  - *Model fitting based on empirical data*: Given empirical data on customer flows, regional market shares, or total sales, users may calibrate market area models. Model parametrization may be conducted using the *MCI Model*, by Maximum Likelihood optimization, or by a local optimization algorithm [guessefeldt2002; wieland2017]. Additionally, machine learning-based market area models may be trained using empirical customer flow data. Evaluation metrics are available that are automatically calculated to assess the predictive fit of the models (e.g., *R-squared*, *RMSE*, *MAPE*).

  - *Accessibility analysis*: The package provides methods of accessibility analysis, which may be combined with market area analysis (*Hansen Accessibility*, *2SFCA Analysis*). Competitor accessibility may also be calculated directly in order to extend the *Huff Model* in terms of the *Competing Destinations Model* [@fotheringham1985].

  - *GIS tools*: The library includes auxiliary GIS functions for market area analysis (buffer, distance matrix, overlay statistics) and clients for OpenRouteService [@neis2008] and OpenStreetMap [@haklay2008] for simple maps. All of them are implemented in the modeling functions but may also be used stand-alone.


## State of the field

To the best of our knowledge, no open-source Python package currently provides market area analysis and parameter estimation for the Huff or MCI Model. In particular, there is no known open-source software package that covers the entire workflow of market area analyses, as described in the "Statement of need" section. Some but not all of the functionalities mentioned are implemented in R packages: Both the `SpatialPosition` package [@giraud2025] and the `huff-tools` package [@pavlis2014] provide basic Huff Model analyses with two parameters, calculation of air distances, and map visualization. The R package `MCI` [@wieland2017] focuses on model fitting based on empirical data, but does not provide processing of geospatial data and the calculation of distances or travel times. Accessibility analysis via 2SFCA Analysis is implemented in the R package `accessibility` [@pereira2024]. The (almost) complete workflow for market area analyses using the Huff/MCI Model is currently only implemented in proprietary GIS software, namely the *ArcGIS Business Analyst* by *ESRI* [@esri2025; @huff2008].


## Software architecture and documentation

### Design choices

The `huff` package is organized into a modular architecture that separates core modeling functionality from auxiliary helper modules. All model-related classes, methods and functions are implemented in the `models` module. Supporting functionalities are provided in separate modules, organized thematically. For example, the `ors` module provides an OpenRouteService client for retrieving travel time matrices and isochrones, which may be directly accessed from the `models` module. This design allows auxiliary functions to be used independently of the core models (stand-alone). Configurations for all models, classes, and functions are stored in a `config` module.

The `huff` library follows an object-oriented design. The class structure reflects the conceptual actors within a spatial market: Customer demand locations are represented by the `CustomerOrigins` class and supply locations by the `SupplyLocations` class. Their connection is established via an interaction matrix containing all possible origin-destination combinations and the corresponding data, such as travel times and location attributes. It is created from the location data using the built-in function `create_interaction_matrix()` from the `models` module, resulting in an instance of the `InteractionMatrix` class. All implemented model analyses can be performed from an `InteractionMatrix` object, with the individual steps of the model calculations being methods of this class, e.g., `transport_costs()` for adding distances or travel times, `probabilities()`, `flows()`, and `marketareas()` for Huff Model calculations, or `mci_fit()` for a MCI Model analysis. These model analyses return objects of specific classes for each model, e.g., `HuffModel` and `MCIModel` for Huff and MCI Models, respectively. This structure was chosen to ensure a consistent workflow and a unified data structure, regardless of which model analysis is to be performed. All mentioned classes include `summary()` and `plot()` methods. Any object contains time stamps for each change, which may be accessed with the `show_log()` method.

### Example workflow

A basic Huff Model analysis (without empirical parameter estimation) in the `huff` package consists of the following steps:
1. Load geospatial data of customer origins and supply locations
2. Define their attributes and weightings
3. Create an interaction matrix from origins and destinations, including the calculation of distances or travel times
4. Calculate utilities, regional market shares, and expected customer flows
5. Calculate total market areas of all supply locations

### Documentation

The workflow above is illustrated in the *Examples* section of the package [README](https://github.com/geowieland/huff_official/blob/main/README.md). For advanced model analyses (e.g., model calibration) and other package functionalities, the [examples](https://github.com/geowieland/huff_official/tree/main/examples) folder in the `huff` [GitHub repository](http://www.github.com/geowieland/huff_official.git) contains commented Python scripts. In addition to the API documentation and examples, `huff` provides a methodological handbook as a [GitHub Pages site](https://geowieland.github.io/huff_official/). This manual documents the mathematical formulations and theoretical foundations of the implemented models and provides guidance on their application and interpretation. This combination of software, examples, and methodological documentation is intended to support reproducible and transparent application of the package in research.


## Research impact statement

The `huff` package has already been used in a health geography project at Würzburg University Hospital to model the catchment areas for pediatric oncology care; one publication citing the package has resulted from this research project so far [@kapitza2026]. The library is actively used: since its first release in April 2025, it has been downloaded 44,915 times from the Python Package Index (source: [pepy.tech](https://pepy.tech/project/huff), accessed October 5, 2026).


## AI usage disclosure

No AI tools were used for software design, implementation, or decision-making. GitHub Copilot in Microsoft Visual Studio Code using the GPT-5 mini model (by OpenAI) was used to assist in drafting and refining docstrings for documentation. The corresponding guidelines and constraints defined by the author are documented in `AGENTS-docstrings.md` in the [public GitHub repository](https://github.com/geowieland/huff_official). The manuscript text was written without the use of AI tools.


## References