---
title: "huff: A Python package for Market Area Analysis"
author:
  - |
    \textbf{Thomas Wieland}  
    Freiburg, Germany  
    ORCID: 0000-0001-5168-9846  
    EMail: geowieland@googlemail.com
bibliography: paper.bib
date: 19 February 2026
version: 1.0.0
---


**Summary**

Market area models, such as the *Huff model* and its extensions, are widely used to estimate regional market shares and customer flows of retail and service locations. Another, now very common, area of ​​application is the analysis of catchment areas, supply structures and the accessibility of healthcare locations. The `huff` Python package provides a complete workflow for market area analysis, including data import, construction of origin-destination interaction matrices, basic model analysis, parameter estimation from empirical data, calculation of distance or travel time indicators, and map visualization. Additionally, the package provides several methods of spatial accessibility analysis. The package is modular and object-oriented. It is intended for researchers in economic geography, regional economics, spatial planning, marketing, geoinformation science, and health geography. The software is openly available via the [Python Package Index (PyPI)](https://pypi.org/project/huff/); its development and version history are managed in a public [GitHub Repository](https://github.com/geowieland/huff_official) and archived at [Zenodo](https://doi.org/10.5281/zenodo.18639559).


**Statement of need**

Market area models are used in economic geography, regional economics, spatial planning, geoinformation science, and marketing, enabling the analysis and forecasting of market areas and customer flows for retail and service locations. The classical and most popular approach is the *Huff model* [@huff1962; @huff1963; @huff1964] and its numerous derivates and extensions, such as the *Multiplicative Competitive Interaction (MCI) Model* [@nakanishi1974; @nakanishi1982]. Typical research applications include examining the influence of store attributes and transport costs on consumer store choice, forecasting the revenue of new locations, or predicting the impact of new locations on existing ones [@debeule2014; @fittkau2004; @li2012; @mensing2018; @oruc2012; @suarezvega2015; @wieland2015; @wieland2019].

In health geography, such models are used to analyse catchment areas with respect to medical practices and hospitals [@bai2023; @fuelop2011; @jia2016; @latruwe2023; @vonrhein2025; @wieland2018], and they are also increasingly being linked to methods for analyzing the supply structure and accessibility of health locations [@liu2022; @rauch2023; @subal2021]. Moreover, market area models are also applied to other location-related contexts such as airports or recreation facilities [@wang2022; @wang2026].

There are several major challenges in model-based market area analyses:

  - The calibration of the Huff model based on observed data on consumer behavior and/or store sales is difficult because the model is nonlinear in its weighting parameters [@huff2003; @wieland2017]. In this context, the MCI model [@nakanishi1974; @nakanishi1982] has been developed as an econometric estimation technique based on a linearization (*log-centering transformation*). As this approach requires empirical market shares for fitting, it is applied in cases where customer-store interaction data was obtained by surveys or secondary data [@baviera2016; @latruwe2023; @oruc2012; @suarezvega2015; @wieland2015; @wieland2019]. Several other researchers developed and used nonlinear iterative fitting approaches, especially when no empirical customer-store interactions are available, but only total sales of the locations investigated [@debeule2014; @guessefeldt2002; @haines1972; @li2012; @liang2020; @mensing2018; @orpana2003; @wieland2017]. Due to the pronounced sensitivity of market area models to weighting schemes, the availability of multiple calibration approaches is essential in market area analysis.

  - Researchers must choose and compare appropriate weighting functions, which may be chosen based on theoretical considerations and may result in substantially different results. Nowadays, for input variables such as travel time, several weighting functions (e.g., power, exponential, logistic) are used, and the model results are compared using goodness-to-fit metrics [@bai2023; @latruwe2023; @li2012; @orpana2003]. It is, thus, necessary that, within the market area analysis workflow, several weighting functions are available, and that there are options to compare different model specifications based on fit metrics.

  - Calculating travel times may be time consuming because these are based on graph theory network analysis and require real street networks [@huff2008]. Therefore, market area analysis typically requires GIS (Geographic Information System) support and/or access to an API providing calculations based on input origins and destinations. It is extremely helpful for researchers if they can also complete this part of the market area analysis workflow within the analysis tool.

The huff package for Python v1.8.x essentially provides the following features:

  - *Data management and preliminary analysis*: Users may load customer origins and supply locations from point shapefiles (or CSV, XLSX). Attributes of customer origins and supply locations (variables, weightings) may be set by the user. The next step is to create an *interaction matrix* with a built-in function, on the basis of which all implemented models can then be calculated. Within an interaction matrix, *transport costs* (distance or travel time between customer origins and supply locations) may be calculated with built-in methods.

  - *Basic Huff model analysis*: Given an interaction matrix, users may calculate probabilities and expected customer flows with respect to customer origins, and total market areas of supply locations.

  - *Parameter estimation based on empirical data*: Given empirical data on customer flows, regional market shares, or total sales, users may estimate weighting parameters of market area models. Model parametrization may be undergone using the econometric approach in the *MCI model* (if regional market shares are available) or by Maximum Likelihood optimization using regional market shares, customer flows, or total market areas. 

  - *Accessibility analysis*: The package includes methods of accessibility analysis, which may be combined with market area analysis (especially empirical estimation of weighting parameters), namely *Hansen accessibility* [@hansen1959] and *Two-step floating catchment area analysis (2SFCA)* [@luo2003].

  - *GIS tools*: The library also includes auxiliary GIS functions for market area analysis (buffer, distance matrix, overlay statistics) and clients for OpenRouteService [@neis2008] and OpenStreetMap [@haklay2008] for simple maps, with all of them being implemented in the market area analysis functions but are also able to be used stand-alone.

**State of the field**

To the best of our knowledge, no open-source Python package currently provides market area analysis and parameter estimation for the Huff or MCI model. No open-source software package currently exists that covers the entire workflow of market area analyses, as described in the "Statement of need" section. Some but not all of the functionalities mentioned are implemented in R packages: Both the `SpatialPosition` package [@giraud2025] and the `huff-tools` package [@pavlis2014] provide basic Huff Model analyses with two parameters, calculation of air distances, and map visualization. The R package `MCI` [@wieland2017] focuses on model fitting based on empirical data, but does not provide processing of geospatial data and the calculation of distances or travel times. Accessibility analysis via two-step floating catchment area analysis is implemented in the R package `accessibility` [@pereira2024]. The (almost) complete workflow for market area analyses using the Huff/MCI model is currently only implemented in proprietary GIS software, namely the *ArcGIS Business Analyst* by *ESRI* [@esri2025; @huff2008].

**Software Architecture**

The `huff` package is organized into a modular architecture that separates core modeling functionality from auxiliary helper modules. All model-related classes, methods and functions are implemented in the `models` module. Supporting functionalities are provided in separate modules, organized thematically. For example, the `ors` module provides an OpenRouteService client for retrieving travel time matrices and isochrones, which may be directly accessed from the `models` module. This design allows auxiliary functions to be used independently of the core models (stand-alone). In order to harmonize the data and outputs while processing, the `config` module includes configurations for all functions and definitions of default column names, suffixes and prefixes, and model terminology.

The `huff` library follows an object-oriented design. The class structure reflects the conceptual actors of a spatial market: Customer demand locations are represented by the `CustomerOrigins` class and supply locations by the `SupplyLocations` class. Their connection is established via an interaction matrix containing all possible origin-destination combinations and the corresponding data, such as travel times and location attributes. It is created from the location data using the built-in function `create_interaction_matrix()` from the `models` module, resulting in an instance of the `InteractionMatrix` class. All implemented model analyses may be calculated from an `InteractionMatrix` object, with the individual steps of the model calculations being methods of this class, e.g., `transport_costs()` for adding distances or travel times, `probabilities()`, `flows()`, and `marketareas()` for Huff model calculations, or `mci_fit()` for a MCI model analysis. These model analyses return objects of specific classes for each model, e.g., `HuffModel` and `MCIModel` for Huff and MCI models, respectively. All mentioned classes include `summary()`, `show_log()`, and, in relevant cases, `plot()` methods. 

This structure was chosen to ensure a consistent workflow and a unified data structure, regardless of which model analysis is to be performed. The typical workflow for a basic Huff analysis (without empirical parameter estimation) consists of the following steps: (1) Load geospatial data of customer origins and supply locations, (2) Define their attributes and weightings, (3) Create an interaction matrix from origins and destinations, including the calculation of distances or travel time, (4) Calculate regional market shares, expected customer flows, and total market areas of all supply locations (This workflow is shown in the *Examples* section of the package README.MD). Advanced model analyses (e.g., including empirical calibration) require further steps (See the examples folder in the [corresponding GitHub repository](http://www.github.com/geowieland/huff_official.git)).

**Research impact statement**

The `huff` package is currently used in a health geography project at the Wuerzburg university hospital which deals with the catchment areas of pediatric oncology care; a paper on this topic is currently in the review process. Given the rising number of scientific studies using market area models - particularly for non-retail purposes such as health geography - and the widespread use of Python as a programming language, it is to be expected that the `huff` library will see frequent adoption in related research projects.

**Software development history statement**

Due to data confidentiality requirements, the early development of the `huff` library took place in a private repository; the public repository was initialized more recently to provide open access for reproducibility and review. The `huff` Python package has been publicly developed and published via the [Python Package Index] (https://pypi.org/project/huff/) since April 2025. As of submission, it has undergone 41 releases, showing continuous improvement and feature additions. The library is actively used: since its first release (version 1.0.0) in April 2025, it has been downloaded approximately 21,900 times from the Python Package Index (source: [pepy.tech](https://pepy.tech/project/huff), accessed February 14, 2026).

**AI usage disclosure**

No AI tools were used for software design, implementation, or decision-making. The Continue agent in Microsoft Visual Studio Code (with model GPT-5 mini) was used to generate initial docstrings, which were subsequently reviewed and adapted by the author. The manuscript text was written without the use of AI tools.

**References**