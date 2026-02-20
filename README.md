# huff: Market Area Analysis in Python

![](https://raw.githubusercontent.com/geowieland/huff_official/main/images/Huff_Freiburg.png)

This Python library is designed for performing market area analyses with the *Huff Model* (Huff 1962, 1964) and/or the *Multiplicative Competitive Interaction (MCI) Model* (Nakanishi and Cooper 1974, 1982). The package is especially intended for researchers in economic geography, regional economics, spatial planning, marketing, geoinformation science, and health geography. It is designed to cover the entire workflow of a market area analysis, including model calibration and GIS-related processing. Users may load point shapefiles (or CSV, XLSX) of customer origins and supply locations and conduct a market area analysis step by step. The first step after importing is always to create an interaction matrix with a built-in function, on the basis of which all implemented models can then be calculated. The library supports parameter estimation based on empirical customer data using the MCI model or Maximum Likelihood estimation. See Huff and McCallum (2008), Orpana and Lampinen (2003) and Wieland (2017) for a description of the models, their practical application and fitting procedures. Additionally, the library includes functions for accessibility analysis, which may be combined with market area analysis, namely the *Hansen accessibility* (Hansen 1959) and the *Two-step floating catchment area analysis* (Luo and Wang 2003). The package also includes auxiliary GIS functions for market area analysis (buffer, distance matrix, overlay statistics) and clients for OpenRouteService(1) for network analysis (e.g., transport cost matrix) and OpenStreetMap(2) for simple maps. All auxiliary functions are implemented in the market area analysis functions but are also able to be used stand-alone. 


## Author

Thomas Wieland [ORCID](https://orcid.org/0000-0001-5168-9846) [EMail](mailto:geowieland@googlemail.com) 


## Availability

- 📦 PyPI: [huff](https://pypi.org/project/huff/)
- 💻 GitHub Repository: [huff_official](https://github.com/geowieland/huff_official)
- 📄 DOI (Zenodo): [10.5281/zenodo.18639559](https://doi.org/10.5281/zenodo.18639559)

A software paper describing the library is available at [arXiv](https://arxiv.org/abs/2602.17640)


## Citation

If you use this software, please cite:

Wieland, T. (2026). huff: Market Area Analysis in Python (Version 1.8.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18639559


## Installation

To install the package from the Python Package Index (PyPI), use `pip`:

```bash
pip install huff
```

To install the package from GitHub with `pip`: 

```bash
pip install git+https://github.com/geowieland/huff_official.git
```


## Features

- **Data management and preliminary analysis**: 
  - Importing tables or point geodata or interaction matrix
  - Setting attributes of customer origins and supply locations (variables, weightings)
  - Creating interaction matrix from point geodata (origins and destinations), including calculation of transport costs (distance, travel time)
  - Creating interaction matrix from survey data
- **Huff Model**:
  - Basic Huff Model analysis based on an interaction matrix
  - Different function types: power, exponential, logistic
  - Defining further attraction indicators in the utility function
  - Huff model parameter estimation via Maximum Likelihood (ML) by probalities, customer flows, and total market areas
  - Huff model market simulation
- **Multiplicative Competitive Interaction Model**: 
  - Log-centering transformation of interaction matrix
  - Fitting MCI model with >= 2 independent variables in the utility function
  - Huff-like MCI model market simulation
  - MCI model market simulation with inverse log-centering transformation
- **Hansen accessibility**:
  - Calculating basic Hansen accessibility based on an interaction matrix
  - Calculating multivariate and (empirically) weighted Hansen accessibility based on an interaction matrix
- **Two-step floating catchment area analysis**:
  - Calculating basic 2SFCA analysis based on an interaction matrix
  - Calculating multivariate and (empirically) weighted 2SFCA analysis based on an interaction matrix
- **GIS tools**:
  - OpenRouteService(1) Client (implemented in model functions, but also available stand-alone):
    - Creating transport costs matrix from origins and destinations
    - Creating isochrones from origins and destinations
  - OpenStreetMap(2) Client (implemented in model functions, but also available stand-alone):
    - Creating simple maps with OSM basemap
  - Other GIS tools (implemented in model functions, but also available stand-alone):
    - Creating buffers from geodata
    - Spatial join with with statistics
    - Creating euclidean distance matrix from origins and destinations
    - Overlay-difference analysis of polygons

(1) © openrouteservice.org by HeiGIT | Map data © OpenStreetMap contributors | https://openrouteservice.org/

(2) © OpenStreetMap contributors | available under the Open Database License | https://www.openstreetmap.org/


## Examples

```python
# Workflow for basic Huff model analysis:

from huff.data_management import load_geodata
from huff.models import create_interaction_matrix

Haslach = load_geodata(
    "data/Haslach.shp",
    location_type="origins",
    unique_id="BEZEICHN"
    )
# Loading customer origins (shapefile)

Haslach.define_marketsize("pop")
# Definition of market size variable

Haslach.define_transportcosts_weighting(
    func = "power",
    param_lambda = -2.2,    
    )
# Definition of transport costs weighting (lambda)

Haslach.summary()
# Summary after update

Haslach_supermarkets = load_geodata(
    "data/Haslach_supermarkets.shp",
    location_type="destinations",
    unique_id="LFDNR"
    )
# Loading supply locations (shapefile)

Haslach_supermarkets.define_attraction("VKF_qm")
# Defining attraction variable

Haslach_supermarkets.define_attraction_weighting(
    param_gamma=0.9
    )
# Define attraction weighting (gamma)

Haslach_supermarkets.summary()
# Summary of updated customer origins

haslach_interactionmatrix = create_interaction_matrix(
    Haslach,
    Haslach_supermarkets
    )
# Creating interaction matrix

haslach_interactionmatrix.transport_costs(
    ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
    network=True,
    )
# Obtaining transport costs (default: driving-car)
# set network = True to calculate transport costs matrix via ORS API (default)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

haslach_interactionmatrix.summary()
# Summary of interaction matrix

haslach_interactionmatrix.flows()
# Calculating spatial flows for interaction matrix

huff_model = haslach_interactionmatrix.marketareas()
# Calculating total market areas
# Result of class HuffModel

huff_model.summary()
# Summary of Huff model

haslach_interactionmatrix.plot(
    origin_point_style = {
        "name": "Districts",
        "color": "black",
        "alpha": 1,
        "size": 100,
        },
    location_point_style = {
        "name": "Supermarket chains",
        "color": {
            "Name": {
                "Aldi Süd": "blue",
                "Edeka": "yellow",
                "Lidl": "red",
                "Netto": "orange",
                "Real": "darkblue",
                "Treff 3000": "fuchsia"
                }
            },
        "alpha": 1,
        "size": 100
        },    
    )
# Plot of interaction matrix with expected customer flows
```

For detailed examples, see the /examples folder in the [public GitHub repository](https://github.com/geowieland/huff_official).


## Literature
  - Cooper LG, Nakanishi M (1983) Standardizing Variables in Multiplicative Choice Models. *Journal of Consumer Research* 10(1): 96–108. [10.1086/208948](https://doi.org/10.1086/208948)
  - De Beule M, Van den Poel D, Van de Weghe N (2014) An extended Huff-model for robustly benchmarking and predicting retail network performance. *Applied Geography* 46(1): 80–89. [10.1016/j.apgeog.2013.09.026](https://doi.org/10.1016/j.apgeog.2013.09.026)
  - Güssefeldt J (2002) Zur Modellierung von räumlichen Kaufkraftströmen in unvollkommenen Märkten. *Erdkunde* 56(4): 351–370. [10.3112/erdkunde.2002.04.02](https://doi.org/10.3112/erdkunde.2002.04.02)
  - Haines Jr GH, Simon LS, Alexis M (1972) Maximum Likelihood Estimation of Central-City Food Trading Areas. *Journal of Marketing Research* 9(2): 154-159. [10.2307/3149948](https://doi.org/10.2307/3149948)
  - Hansen WG (1959) How Accessibility Shapes Land Use. *Journal of the American Institute of Planners* 25(2): 73-76. [10.1080/01944365908978307](https://doi.org/10.1080/01944365908978307)
  - Huff DL (1962) *Determination of Intra-Urban Retail Trade Areas*. Real Estate Research Program, Graduate Schools of Business Administration, University of California.
  - Huff DL (1963) A Probabilistic Analysis of Shopping Center Trade Areas. *Land Economics* 39(1): 81-90. [10.2307/3144521](https://doi.org/10.2307/3144521)
  - Huff DL (1964) Defining and estimating a trading area. *Journal of Marketing* 28(4): 34–38. [10.2307/1249154](https://doi.org/10.2307/1249154)
  - Huff DL (2003) Parameter Estimation in the Huff Model. *ArcUser* 6(4): 34–36. https://stg.esri.com/news/arcuser/1003/files/huff.pdf
  - Huff DL, Batsell RR (1975) Conceptual and Operational Problems with Market Share Models of Consumer Spatial Behavior. *Advances in Consumer Research* 2(1): 165-172. 
  - Huff DL, McCallum BM (2008) Calibrating the Huff Model using ArcGIS Business Analyst. ESRI White Paper, September 2008. https://www.esri.com/library/whitepapers/pdfs/calibrating-huff-model.pdf.
  - Luo W, Wang F (2003) Measures of spatial accessibility to health care in a GIS environment: synthesis and a case study in the Chicago region. *Environment and Planning B: Planning and Design* 30: 865-884. [10.1068/b29120](https://doi.org/10.1068/b29120)
  - Luo J (2014) Integrating the Huff Model and Floating Catchment Area Methods to Analyze Spatial Access to Healthcare Services. *Transactions in GIS* 18(3): 436-448. [10.1111/tgis.12096](https://doi.org/10.1111/tgis.12096)
  - Nakanishi M, Cooper LG (1974) Parameter estimation for a Multiplicative Competitive Interaction Model: Least squares approach. *Journal of Marketing Research* 11(3): 303–311. [10.2307/3151146](https://doi.org/10.2307/3151146).
  - Nakanishi M, Cooper LG (1982) Technical Note — Simplified Estimation Procedures for MCI Models. *Marketing Science* 1(3): 314-322. [10.1287/mksc.1.3.314](https://doi.org/10.1287/mksc.1.3.314)
  - Orpana T, Lampinen J (2003) Building Spatial Choice Models from Aggregate Data. *Journal of Regional Science* 43(2): 319-348. [10.1111/1467-9787.00301](https://doi.org/10.1111/1467-9787.00301)
  - Rauch S, Wieland T, Rauh J (2025) Accessibility of food - A multilevel approach comparing a choice based model with perceived accessibility in Mainfranken, Germany. *Journal of Transport Geography* 128: 104367. [10.1016/j.jtrangeo.2025.104367](https://doi.org/10.1016/j.jtrangeo.2025.104367)
  - Wieland T (2015) *Räumliches Einkaufsverhalten und Standortpolitik im Einzelhandel unter Berücksichtigung von Agglomerationseffekten - Theoretische Erklärungsansätze, modellanalytische Zugänge und eine empirisch-ökonometrische Marktgebietsanalyse anhand eines Fallbeispiels aus dem ländlichen Raum Ostwestfalens/Südniedersachsens*. Mannheim: MetaGIS. https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753
  - Wieland T (2017) Market Area Analysis for Retail and Service Locations with MCI. *R Journal* 9(1): 298-323. [10.32614/RJ-2017-020](https://doi.org/10.32614/RJ-2017-020)
  - Wieland T (2018) A Hurdle Model Approach of Store Choice and Market Area Analysis in Grocery Retailing. *Papers in Applied Geography* 4(4): 370-389. [10.1080/23754931.2018.1519458](https://doi.org/10.1080/23754931.2018.1519458)
  - Wieland T (2018) Competitive locations of grocery stores in the local supply context - The case of the urban district Freiburg-Haslach. *European Journal of Geography* 9(3): 98-115. https://www.eurogeojournal.eu/index.php/egj/article/view/41


## What's new (v1.8.3)
- Bugfixes
  - Correction in goodness_of_fit.modelfit(): Length of observed and expected vectors is refreshed after removing NaN (if desired by user)
- Other
  - goodness_of_fit.modelfit() skips zero values when calculating APE and MAPE instead of returning None
  - Update of literature in README