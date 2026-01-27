# huff: Market Area Analysis in Python

![](https://raw.githubusercontent.com/geowieland/huff_official/main/images/Huff_Freiburg.png)

This Python library is designed for performing market area analyses with the Huff Model (Huff 1962, 1964) and/or the Multiplicative Competitive Interaction (MCI) Model (Nakanishi and Cooper 1974, 1982). Users may load point shapefiles (or CSV, XLSX) of customer origins and supply locations and conduct a market area analysis step by step. The library supports parameter estimation based on empirical customer data using the MCI model and Maximum Likelihood. See Huff and McCallum (2008), Orpana and Lampinen (2003) and Wieland (2017) for a description of the models, their practical application and fitting procedures. The package also includes GIS functions for market area analysis (buffer, distance matrix, overlay statistics) and clients for OpenRouteService(1) for network analysis (e.g., transport cost matrix) and OpenStreetMap(2) for simple maps. 


## Author

Thomas Wieland [ORCID](https://orcid.org/0000-0001-5168-9846) [EMail](mailto:geowieland@googlemail.com) 


## Updates v1.7.1
- Creating maps of CustomerOrigins, SupplyLocations, InteractionMatrix, and HuffModel objects via plot()
- Logging with timestamps for objects of all model-specific classes (CustomerOrigins, SupplyLocations, InteractionMatrix, ...)
- Bugfixes:
  - Checking for CRS in map_with_basemap(): Converting crs to str
  - Fixed math error in distance_matrix() when source == destination
  - Default suffix for LCT-transformed variables in the MCI model
  - Fixed bug in construction of MCIModel object in add_to_model()
  - Fixed bug in load_interaction_matrix(): geodata_gpd_original includes geometry if available
  - Fixed bug in map_with_basemap(): Harmonizing CRS now works in all cases
 

## Features

- **Data management and preliminary analysis**: 
  - Importing tables or point geodata or interaction matrix
  - Defining origins and destinations with weightings
  - Creating interaction matrix from point geodata (origins and destinations)
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
  - Calculating Hansen accessibility based on an interaction matrix
 - **GIS tools**:
  - OpenRouteService(1) Client:
    - Creating transport costs matrix from origins and destinations
    - Creating isochrones from origins and destinations
  - OpenStreetMap(2) Client:
    - Creating simple maps with OSM basemap
  - Other GIS tools:
    - Creating buffers from geodata
    - Spatial join with with statistics
    - Creating euclidean distance matrix from origins and destinations
    - Overlay-difference analysis of polygons

(1) © openrouteservice.org by HeiGIT | Map data © OpenStreetMap contributors | https://openrouteservice.org/

(2) © OpenStreetMap contributors | available under the Open Database License | https://www.openstreetmap.org/


## Examples

```python
# Dealing with customer origins (statistical districts):

Haslach = load_geodata(
    "data/Haslach.shp",
    location_type="origins",
    unique_id="BEZEICHN"
    )
# Loading customer origins (shapefile)

Haslach_buf = Haslach.buffers(
    segments_distance=[500,1000,1500],
    save_output=True,
    output_filepath="Haslach_buf.shp",
    output_crs="EPSG:31467"
    )
# Buffers for customer origins

Haslach.summary()
# Summary of customer origins

Haslach.plot(
    point_style = {
            "name": "Districts",
            "color": "black",
            "alpha": 1,
            "size": 15,
        },
    polygon_style= {
        "name": "Buffers",
        "color": {
            "buffer": {
                500: "midnightblue", 
                1000: "blue", 
                1500: "dodgerblue",
                }
            },            
        "alpha": 0.3
    },
    map_title = "Districts in Haslach with buffers",    
)
# Plot map of customer origins and assigned buffers, 
# both with user-defined layer styles

Haslach.define_marketsize("pop")
# Definition of market size variable

Haslach.define_transportcosts_weighting(
    param_lambda = -2.2,    
    # one weighting parameter for power function (default)
    # two weighting parameters for logistic function
    )
# Definition of transport costs weighting (lambda)

Haslach.summary()
# Summary after update

Haslach.show_log()
# Show log of CustomerOrigins object

# Dealing with supply locations (supermarkets):

Haslach_supermarkets = load_geodata(
    "data/Haslach_supermarkets.shp",
    location_type="destinations",
    unique_id="LFDNR"
    )
# Loading supply locations (shapefile)

Haslach_supermarkets.summary()
# Summary of supply locations

Haslach_supermarkets.define_attraction("VKF_qm")
# Defining attraction variable

Haslach_supermarkets.define_attraction_weighting(
    param_gamma=0.9
    )
# Define attraction weighting (gamma)

Haslach_supermarkets.isochrones(
    segments=[2, 4, 6],
    # minutes or kilometers
    range_type = "time",
    # "time" or "distance" (default: "time")
    profile = "foot-walking",
    save_output=True,
    ors_auth="5b3ce3597851110001cf62480a15aafdb5a64f4d91805929f8af6abd",
    output_filepath="Haslach_supermarkets_iso.shp",
    output_crs="EPSG:31467",
    delay=0.2
    )
# Obtaining isochrones for walking (5 and 10 minutes)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

Haslach_supermarkets.summary()
# Summary of updated customer origins

Haslach_supermarkets.plot(
    point_style = {
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
        "size": 30
    },
    polygon_style= {
        "name": "Isochrones",
        "color": {
            "segm_min": {
                2: "midnightblue", 
                4: "blue",
                6: "dodgerblue"
                }
            },            
        "alpha": 0.3
    },
    map_title = "Supermarkets in Haslach with isochrones"
)
# Plot map of supply locations and assigned isochrones,
# both with user-defined layer styles

Haslach_supermarkets_isochrones = Haslach_supermarkets.get_isochrones_gdf()
# Extracting isochrones as gdf


# Using customer origins and supply locations for building interaction matrix:

haslach_interactionmatrix = create_interaction_matrix(
    Haslach,
    Haslach_supermarkets
    )
# Creating interaction matrix

haslach_interactionmatrix.transport_costs(
    ors_auth="5b3ce3597851110001cf62480a15aafdb5a64f4d91805929f8af6abd",
    network=False,
    #distance_unit="meters",
    # set network = True to calculate transport costs matrix via ORS API (default)
    )
# Obtaining transport costs (default: driving-car)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

haslach_interactionmatrix.summary()
# Summary of interaction matrix

print(haslach_interactionmatrix.hansen())
# Hansen accessibility for interaction matrix

haslach_interactionmatrix.flows()
# Calculating spatial flows for interaction matrix

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

huff_model = haslach_interactionmatrix.marketareas()
# Calculating total market areas
# Result of class HuffModel

huff_model.summary()
# Summary of Huff model
```

See the /tests directory for usage examples of most of the included functions.


## Literature
  - Bai L, Tao Z, Cheng Y, Feng L, Wang S (2023) Delineating hierarchical obstetric hospital service areas using the Huff model based on medical records. *Applied Geography* 153: 102903. [10.1016/j.apgeog.2023.102903](https://doi.org/10.1016/j.apgeog.2023.102903)
  - De Beule M, Van den Poel D, Van de Weghe N (2014) An extended Huff-model for robustly benchmarking and predicting retail network performance. *Applied Geography* 46(1): 80–89. [10.1016/j.apgeog.2013.09.026](https://doi.org/10.1016/j.apgeog.2013.09.026)
  - Haines Jr GH, Simon LS, Alexis M (1972) Maximum Likelihood Estimation of Central-City Food Trading Areas. *Journal of Marketing Research* 9(2): 154-159. [10.2307/3149948](https://doi.org/10.2307/3149948)
  - Huff DL (1962) *Determination of Intra-Urban Retail Trade Areas*. Real Estate Research Program, Graduate Schools of Business Administration, University of California.
  - Huff DL (1963) A Probabilistic Analysis of Shopping Center Trade Areas. *Land Economics* 39(1): 81-90. [10.2307/3144521](https://doi.org/10.2307/3144521)
  - Huff DL (1964) Defining and estimating a trading area. *Journal of Marketing* 28(4): 34–38. [10.2307/1249154](https://doi.org/10.2307/1249154)
  - Huff DL, McCallum BM (2008) Calibrating the Huff Model using ArcGIS Business Analyst. ESRI White Paper, September 2008. https://www.esri.com/library/whitepapers/pdfs/calibrating-huff-model.pdf.
  - Jia P (2016) Developing a flow-based spatial algorithm to delineate hospital service areas. *Applied Geography* 75: 137-143. [10.1016/j.apgeog.2016.08.008](https://doi.org/10.1016/j.apgeog.2016.08.008)
  - Latruwe T, Van der Wee M, Vanleenhove P, Michielsen K, Verbrugge S, Colle D (2023) Simulation analysis of an adjusted gravity model for hospital admissions robust to incomplete data. *BMC Medical Research Methodology* 23: 215. [10.1186/s12874-023-02033-0](https://doi.org/10.1186/s12874-023-02033-0)
  - Nakanishi M, Cooper LG (1974) Parameter estimation for a Multiplicative Competitive Interaction Model: Least squares approach. *Journal of Marketing Research* 11(3): 303–311. [10.2307/3151146](https://doi.org/10.2307/3151146).
  - Nakanishi M, Cooper LG (1982) Technical Note — Simplified Estimation Procedures for MCI Models. *Marketing Science* 1(3): 314-322. [10.1287/mksc.1.3.314](https://doi.org/10.1287/mksc.1.3.314)
  - Orpana T, Lampinen J (2003) Building Spatial Choice Models from Aggregate Data. *Journal of Regional Science* 43(2): 319-348. [10.1111/1467-9787.00301](https://doi.org/10.1111/1467-9787.00301)
  - Suárez-Vega R, Gutiérrez-Acuña JL, Rodríguez-Díaz M (2015) Locating a supermarket using a locally calibrated Huff model. *International Journal of Geographical Information Science* 29(2): 217–233. [10.1080/13658816.2014.958154](https://doi.org/10.1080/13658816.2014.958154)
  - Wieland T (2015) *Nahversorgung im Kontext raumökonomischer Entwicklungen im Lebensmitteleinzelhandel: Konzeption und Durchführung einer GIS-gestützten Analyse der Strukturen des Lebensmitteleinzelhandels und der Nahversorgung in Freiburg im Breisgau*. Working paper. Göttingen. https://webdoc.sub.gwdg.de/pub/mon/2015/5-wieland.pdf.
  - Wieland T (2017) Market Area Analysis for Retail and Service Locations with MCI. *R Journal* 9(1): 298-323. [10.32614/RJ-2017-020](https://doi.org/10.32614/RJ-2017-020)
  - Wieland T (2018) A Hurdle Model Approach of Store Choice and Market Area Analysis in Grocery Retailing. *Papers in Applied Geography* 4(4): 370-389. [10.1080/23754931.2018.1519458](https://doi.org/10.1080/23754931.2018.1519458)
  - Wieland T (2023) Spatial shopping behavior during the Corona pandemic: insights from a micro-econometric store choice model for consumer electronics and furniture retailing in Germany. *Journal of Geographical Systems* 25(2): 291–326. [10.1007/s10109-023-00408-x](https://doi.org/10.1007/s10109-023-00408-x)


## Installation

To install the package from the Python Package Index (PyPI), use `pip`:

```bash
pip install huff
```

To install the package from Github with `pip`: 

```bash
pip install git+https://github.com/geowieland/huff_official.git
```