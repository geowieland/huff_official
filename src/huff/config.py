#-----------------------------------------------------------------------
# Name:        config (huff package)
# Purpose:     Huff Model package definitions and configurations
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.10
# Last update: 2026-02-20 18:40
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


PACKAGE_NAME = "huff"
PACKAGE_VERSION = "1.8.3"

# Basic config:

FLOAT_ROUND = 3
SUMMARY_WIDTH = 22
SUMMARY_NOT_DEFINED = "Not defined"

VERBOSE = False

# Permitted location types:
PERMITTED_LOCATION_TYPES = [
    "origins",
    "destinations"
]

# Market area models config:

MATRIX_OD_SEPARATOR = "_"

MATRIX_COL_SOURCE = "source"
MATRIX_COL_DESTINATION = "destination"

# Available models:
MODELS = {
    "Huff": {
        "description": "Huff Model",
        "metadata_model_type": "Huff",
        "fit_function": "huff_ml_fit"
    },
    "MCI": {
        "description": "Multiplicative Competitive Interaction Model",
        "metadata_model_type": "MCI",
        "fit_function": "mci_fit"
    },
    "Hansen": {
        "description": "Hansen Accessibility",
        "metadata_model_type": "Hansen"
    },
    "2SFCA": {
        "description": "Two-step floating catchment area analysis",
        "metadata_model_type": "2SFCA"
    },
}
MODELS_LIST = list(MODELS.keys())

# Default column names:
DEFAULT_COLNAME_ATTRAC = "A_j"
DEFAULT_COLNAME_TC = "t_ij"
DEFAULT_COLNAME_UTILITY = "U_ij"
DEFAULT_COLNAME_UTILITY_SUM = "U_i"
DEFAULT_COLNAME_MARKETSIZE = "C_i"
DEFAULT_COLNAME_PROBABILITY = "p_ij"
DEFAULT_COLNAME_FLOWS = "E_ij"
DEFAULT_COLNAME_CUSTOMER_ORIGINS = "i"
DEFAULT_COLNAME_SUPPLY_LOCATIONS = "j"
DEFAULT_COLNAME_INTERACTION = "ij"
DEFAULT_COLNAME_TOTAL_MARKETAREA = "T_j"
DEFAULT_COLNAME_UTILITY_SUPPLY = "U_ji"
DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS = "A_i"
DEFAULT_COLNAME_SD_RATIO = "R_j"

# Default column name suffixes:
DEFAULT_LCT_SUFFIX = "__LCT"
DEFAULT_WEIGHTED_SUFFIX = "_weighted"
DEFAULT_OBSERVED_SUFFIX = "_emp"
DEFAULT_TOTAL_SUFFIX = "_total"
DEFAULT_COORDS_SUFFIX = "_coords"
DEFAULT_UNIQUE_ID_SUFFIX = "_uid"
DEFAULT_ZST_SUFFIX = "__ZST"

DEFAULT_OBSERVED_COL = "observed"
DEFAULT_EXPECTED_COL = "expected"

DEFAULT_COLNAME_ATTRAC_WEIGHTED = f"{DEFAULT_COLNAME_ATTRAC}{DEFAULT_WEIGHTED_SUFFIX}"
DEFAULT_COLNAME_TC_WEIGHTED = f"{DEFAULT_COLNAME_TC}{DEFAULT_WEIGHTED_SUFFIX}"

DEFAULT_COLNAME_PROBABILITY_OBSERVED = f"{DEFAULT_COLNAME_PROBABILITY}{DEFAULT_OBSERVED_SUFFIX}"
DEFAULT_COLNAME_FLOWS_OBSERVED = f"{DEFAULT_COLNAME_FLOWS}{DEFAULT_OBSERVED_SUFFIX}"
DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED = f"{DEFAULT_COLNAME_TOTAL_MARKETAREA}{DEFAULT_OBSERVED_SUFFIX}"

# Default descriptions:
DEFAULT_NAME_ATTRAC = "Attraction"
DEFAULT_NAME_TC = "Transport costs"
DEFAULT_NAME_MARKETSIZE = "Market size"
DEFAULT_NAME_CUSTOMER_ORIGINS = "Customer origins"
DEFAULT_NAME_SUPPLY_LOCATIONS = "Supply locations"
DEFAULT_NAME_INTERACTION_MATRIX = "Interaction matrix"
DEFAULT_NAME_SD_RATIO = "Supply-to-demand ratio"

# Default weighting functions:
PERMITTED_WEIGHTING_FUNCTIONS = {
    "power": {
        "description": "Power function",
        "function": "a*(values**b)",
        "no_params": 1,
        "type": float
        },
    "exponential": {
        "description": "Exponential function",
        "function": "a*np.exp(b*values)",
        "no_params": 1,
        "type": float 
        },
    "logistic": {
        "description": "Logistic function",
        "function": "1+np.exp(b+c*values)",
        "no_params": 2,
        "type": list
        },
    "linear": {
        "description": "Linear function",
        "function": "a+(b*values)",
        "no_params": 1,
        "type": list
        }
    }
PERMITTED_WEIGHTING_FUNCTIONS_LIST = list(PERMITTED_WEIGHTING_FUNCTIONS.keys())

MCI_TRANSFORMATIONS = {
    "LCT": "Log-centering transformation",
    "ILCT": "Inverse log-centering transformation"
}
MCI_TRANSFORMATIONS_LIST = list(MCI_TRANSFORMATIONS.keys())
DEFAULT_MCI_TRANSFORMATION = MCI_TRANSFORMATIONS_LIST[0]


APE_PREFIX = "APE"
APE_MIN = 1
APE_MAX = 100

GOODNESS_OF_FIT = {
    "Sum of squared residuals": "SQR",
    "Sum of absolute residuals": "SAR",
    "R-squared": "Rsq",
    "Mean squared error": "MSE",
    "Root mean squared error": "RMSE",
    "Mean absolute error": "MAE",
    "Mean absolute percentage error": "MAPE",
    "Symmetric MAPE": "sMAPE",
    "Negative log-likelihood": "LL",
    "Deviations <  5 %": f"{APE_PREFIX}5",
    "Deviations < 10 %": f"{APE_PREFIX}10",
    #"Deviations < 15 %": f"{APE_PREFIX}15",
    #"Deviations < 20 %": f"{APE_PREFIX}20",
    #"Deviations < 25 %": f"{APE_PREFIX}25",
    #"Deviations < 30 %": f"{APE_PREFIX}30",
    #"Deviations < 35 %": f"{APE_PREFIX}35",
    #"Deviations < 40 %": f"{APE_PREFIX}40",
    #"Deviations < 45 %": f"{APE_PREFIX}45",
    #"Deviations < 50 %": f"{APE_PREFIX}50",
}

# ORS config:

ORS_SERVER = "https://api.openrouteservice.org/v2/"
ORS_URL_RESTRICTIONS = "https://openrouteservice.org/restrictions/"

ORS_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8",    
}

ORS_AUTH = "5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f"
# for TESTING

ORS_ENDPOINTS = {
    "Isochrones": {
        "endpoint": "isochrones/",
        "Restrictions": {
            "Locations": 5,
            "Intervals": 10,
            "Range distance": 120,
            "Range time (Foot profiles)": 20,
            "Range time (Cycling profiles)": 5,
            "Range time (Driving profiles)": 1
        },
        "Parameters": {
            "unit": {
                "param": "range_type",
                "type": str,
                "options": {
                    "time": {
                        "description": "Time [minutes]",
                        "param": "time"
                    },
                    "distance": {
                        "description": "Distance [meters]",
                        "param": "distance"
                    }
                },
            },
        }
    },
    "Matrix": {
        "endpoint": "matrix/",
        "Restrictions": {
            "Locations (origin x destination)": 3500            
        },
        "Parameters": {
            "unit": {
                "param": "metrics",
                "type": str,
                "options": {
                    "time": {
                        "description": "Time [minutes]",
                        "param": "duration"
                    },
                    "distance": {
                        "description": "Distance [meters]",
                        "param": "distance"
                    }
                },
            },
        },
    }
}

ORS_PROFILES = {
    "Driving car": "driving-car",
    "Driving heavy goods vehicle": "driving-hgv",
    "Walking by foot": "foot-walking",
}
ORS_PROFILES_LIST = ORS_PROFILES.keys()
ORS_PROFILES_LIST_API = list(ORS_PROFILES.values())

ORS_RANGE_TYPES = {
    "Time [minutes]": "time",
    "Distance [meters]": "distance"
    }
ORS_RANGE_TYPES_LIST = ORS_RANGE_TYPES.keys()
ORS_RANGE_TYPES_LIST_API = list(ORS_RANGE_TYPES.values())

ORS_SEGMENT_COL = "segment"


# OSM config:

OSM_TILES_SERVER = "http://a.tile.openstreetmap.org/"

DEFAULT_FILENAME_ORS_TMP = "osm_map.png"


# GIS constants and defaults:

DEFAULT_GEOMETRY_COL = "geometry"

WGS84_EPSG = 4326
WGS84_CRS = f"EPSG:{WGS84_EPSG}"
PSEUDO_MERCATOR_EPSG = 3857
PSEUDO_MERCATOR_CRS = f"EPSG:{PSEUDO_MERCATOR_EPSG}"

DISTANCE_TYPES = {
    "Euclidean distance": "euclidean",
    "Manhattan distance": "manhattan"
}
DISTANCE_TYPES_LIST = list(DISTANCE_TYPES.keys())
DISTANCE_TYPES_LIST_FUNC = list(DISTANCE_TYPES.values())

DEFAULT_SEGMENTS_COL = ORS_SEGMENT_COL
DEFAULT_SEGMENTS_COL_ABBREV = DEFAULT_SEGMENTS_COL[:4]

DEFAULT_LAYER_ALPHA = 0.6
DEFAULT_LAYER_LABEL = "Layer"
DEFAULT_LEGEND_LOC = "lower right"
DEFAULT_LEGEND_FONTSIZE = "small"
DEFAULT_LEGEND_POINT_SIZE = 7