#-----------------------------------------------------------------------
# Name:        tests_huff (huff package)
# Purpose:     Tests for Huff Model package functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.6.1
# Last update: 2026-02-13 18:37
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------

import geopandas as gp
from huff.models import create_interaction_matrix
from huff.data_management import load_geodata, load_interaction_matrix, load_marketareas
from huff.gistools import point_spatial_join, map_with_basemap, distance_matrix_from_gdf


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
    profile = "driving-car",
    save_output=True,
    ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
    # Authentification token FOR TESTING
    output_filepath="Haslach_supermarkets_iso.shp",
    output_crs="EPSG:31467",
    delay=0.2
    )
# Obtaining isochrones for walking (5 and 10 minutes)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

Haslach_supermarkets.summary()
# Summary of updated customer origins

Haslach_supermarkets.show_log()
# Log of customer origins

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
    ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
    network=False,
    #distance_unit="meters",
    # set network = True to calculate transport costs matrix via ORS API (default)
    )
# Obtaining transport costs (default: driving-car)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

haslach_interactionmatrix.summary()
# Summary of interaction matrix

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

huff_model.show_log()
# Log of Huff model

print(huff_model.get_market_areas_df())
# Showing total market areas


# Maximum Likelihood fit for Huff Model:

haslach_interactionmatrix.huff_ml_fit(
    initial_params=[1, -2],
    method="trust-constr",
    bounds = [(0.8, 0.9999),(-2.5, -1.5)]    
)
# Maximum Likelihood fit for Huff Model

haslach_interactionmatrix.summary()
# Summary of fitted ML-fitted interaction matrix (Huff model)

huff_model_fit = haslach_interactionmatrix.marketareas()
# Calculcation of total market areas
# Result of class HuffModel

bootstrap_cis = huff_model_fit.confint(repeats=10)
print(bootstrap_cis)
# Confidence intervals for estimated parameters

huff_model_fit.summary()
# Huff model summary


# Adding new supply location:

Haslach_new_supermarket = load_geodata(
    "data/Haslach_new_supermarket.shp",
    location_type="destinations",
    unique_id="LFDNR"
    )
# Loading new supply locations (shapefile)

Haslach_new_supermarket.summary()
# Summary of new supply locations data

Haslach_supermarkets.add_new_destinations(Haslach_new_supermarket)
# Adding new supermarket to existing supply locations

Haslach_supermarkets.summary()
# Summary of updated supply locations

huff_model.update()
# Update interaction matrix

huff_model.summary()
# Summary of updated interaction matrix

Haslach_supermarkets.show_log()
# Log of SupplyLocations object

huff_model.interaction_matrix.show_log()
# Log of InteractionMatrix object

huff_model.show_log()
# Log of HuffModel object

print(huff_model.get_market_areas_df())
# Showing total market areas of model with estimated parameters and new destination

print(huff_model.get_interaction_matrix_df())
# Showing df of interaction matrix

huff_model.get_interaction_matrix_df().to_excel("interaction_matrix_df.xlsx")
# Export of interaction matrix


# Multiplicative Competitive Interaction Model:

mci_fit = huff_model.mci_fit()
# Fitting via MCI

mci_fit.summary()
# Summary of MCI model

mci_fit.marketareas()
# MCI model market simulation

mci_fit.get_market_areas_df()
# MCI model market areas

mci_fit.show_log()
# Log of MCIModel object


# Loading own interaction matrix:
# Data source: Wieland 2015 (https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753)

Wieland2015_interaction_matrix = load_interaction_matrix(
    data="data/Wieland2015.xlsx",
    customer_origins_col="Quellort",
    supply_locations_col="Zielort",
    attraction_col=[
        "VF", 
        "K", 
        "K_KKr"
        ],
    market_size_col="Sum_Ek1",
    flows_col="Anb_Eink1",
    transport_costs_col="Dist_Min2",
    transport_costs_metrics="time",
    transport_costs_time_unit="minutes",
    probabilities_col="MA_Anb1",
    data_type="xlsx"
    )
# Loading interaction matrix from XLSX file

Wieland2015_interaction_matrix.summary()
# Summary of interaction matrix


# Parameter estimation via MCI model:

Wieland2015_fit = Wieland2015_interaction_matrix.mci_fit(
    cols=[
        "A_j", 
        "t_ij", 
        "K", 
        "K_KKr"
        ]
    )
# Fitting MCI model with four independent variables

Wieland2015_fit.probabilities()
# Calculating probabilities

Wieland2015_fit_interactionmatrix = Wieland2015_fit.get_interaction_matrix_df()
# Export interaction matrix

Wieland2015_fit.summary()
# MCI model summary

Wieland2015_interaction_matrix.show_log()
# Logs of InteractionMatrix object

Wieland2015_fit.plot(
    origin_point_style = {
        "name": "ZIP codes",
        "color": "black",
        "alpha": 1,
        "size": 30,
        },
    location_point_style = {
        "name": "CE stores",
        "color": {
            "Zielort": {
                "E01": "blue",
                "E02": "yellow",
                "E03": "red",
                "E04": "orange",
                "E07": "darkblue",
                }
            },
        "alpha": 1,
        "size": 50
        },    
    )
# Plot of interaction matrix with customer flows
# This does NOT work because there is no geometry


# Parameter estimation via Maximum Likelihood:

Wieland2015_interaction_matrix2 = load_interaction_matrix(
    data="data/Wieland2015.xlsx",
    customer_origins_col="Quellort",
    supply_locations_col="Zielort",
    attraction_col=[
        "VF", 
        "K", 
        "K_KKr"
        ],
    market_size_col="Sum_Ek",
    flows_col="Anb_Eink",
    transport_costs_col="Dist_Min2",
    transport_costs_metrics="time",
    transport_costs_time_unit="minutes",
    probabilities_col="MA_Anb",
    data_type="xlsx",
    xlsx_sheet="interactionmatrix",
    check_df_vars=False
    )
# Loading empirical interaction matrix again

Wieland2015_interaction_matrix2.define_weightings(
    vars_funcs = {
        0: {
            "name": "A_j",
            "func": "power"
        },
        1: {
            "name": "t_ij",
            "func": "power",
            #"func": "exponential"
            #"func": "logistic"                
        },
        2: {
            "name": "K",
            "func": "power"
        },
        3: {
            "name": "K_KKr",
            "func": "power"
        }
        }
    )
# Defining weighting functions

Wieland2015_interaction_matrix2.huff_ml_fit(
    # Power TC function
    initial_params=[0.9, -1.5, 0.5, 0.3],
    bounds=[(0.5, 1), (-2, -1), (0.2, 0.7), (0.2, 0.7)],
    # # Logistic TC function:
    # initial_params=[0.9, 10, -0.5, 0.5, 0.3],
    # bounds=[(0.5, 1), (8, 12), (-0.7, -0.2), (0.2, 0.7), (0.2, 0.7)],
    fit_by="probabilities",
    #method = "trust-constr"
)
# ML fit with power transport cost function based on probabilities
# from InteractionMatrix object

Wieland2015_interaction_matrix2.summary()
# Summary of interaction matrix

huff_model_fit2 = Wieland2015_interaction_matrix2.marketareas()
# Calculation of market areas

huff_model_fit2 = huff_model_fit2.ml_fit(
    # Power TC function
    initial_params=[0.9, -1.5, 0.5, 0.3],
    bounds=[(0.5, 1), (-2, -1), (0.2, 0.7), (0.2, 0.7)],
    # # Logistic TC function:
    # initial_params=[0.9, 10, -0.5, 0.5, 0.3],
    # bounds=[(0.5, 1), (8, 12), (-0.7, -0.2), (0.2, 0.7), (0.2, 0.7)],
    fit_by="probabilities",
    #method = "trust-constr"
    )
# ML fit with power transport cost function based on probabilities
# from HuffModel object

huff_model_fit2.summary()
# Summary of Huff model


# Loading and including total market areas

wieland2015_totalmarketareas = load_marketareas(
    data="data/Wieland2015.xlsx",
    supply_locations_col="Zielort",
    total_col="Anb_Eink",
    data_type="xlsx",
    xlsx_sheet="total_marketareas"
)
# Loading empirical total market areas

huff_model_fit2 = wieland2015_totalmarketareas.add_to_model(
    huff_model_fit2
    )
# Adding total market areas to HuffModel object

print(huff_model_fit2.get_market_areas_df())
# Showing total market areas of HuffModel object

huff_model_fit3 = huff_model_fit2.ml_fit(
    # Power TC function
    initial_params=[0.9, -1.5, 0.5, 0.3],
    bounds=[(0.5, 1), (-2, -1), (0.2, 0.7), (0.2, 0.7)],
    # # Logistic TC function:
    # initial_params=[0.9, 10, -0.5, 0.5, 0.3],
    # bounds=[(0.5, 1), (8, 12), (-0.7, -0.2), (0.2, 0.7), (0.2, 0.7)],
    fit_by="totals",
    #method = "trust-constr"
    )
# Fit Huff model by totals

huff_model_fit3.summary()
# Show summary

huff_model_fit3.show_log()
# Show log

print(huff_model_fit3.get_market_areas_df())
# Show market areas df


# Stand-alone functions:

# Buffer analysis:

Haslach_supermarkets_gdf = Haslach_supermarkets.get_geodata_gpd_original()
Haslach_buffers = Haslach_buf.get_buffers_gdf()
# Extracting points and buffer polygons

Haslach_districts_buf = point_spatial_join(
    polygon_gdf = Haslach_buffers,
    point_gdf = Haslach_supermarkets_gdf,
    polygon_ref_cols = ["BEZEICHN", "segment"],
    point_stat_col = "VKF_qm"
)
# Spatial join with buffers and points
# Statistics for supermarket selling space by buffers of statistical districts
# (How much selling space in 500, 1000, and 1500 metres?)

Haslach_districts_buf[0].to_file("Haslach_districts_buf.shp")
# Save joined points as shapefile

print(Haslach_districts_buf[1])
# Showing df with overlay statistics


# Isochrones analysis:

Haslach_districts = Haslach.get_geodata_gpd_original()

Haslach_supermarkets_iso = point_spatial_join(
    polygon_gdf = Haslach_supermarkets_isochrones,
    point_gdf = Haslach_districts,
    polygon_ref_cols = ["LFDNR", "segment"],
    point_stat_col = "pop"
)
# Spatial join with isochrones and points
# Statistics for population by isochrones of supermarkets
# (How much population in 5, 10, and 15 minutes?)

Haslach_supermarkets_iso[0].to_file("Haslach_supermarkets_iso.shp")
# Save joined points as shapefile

print(Haslach_supermarkets_iso[1])
# Showing df with overlay statistics


# Creating map (stand-alone):

Haslach_gdf = Haslach.get_geodata_gpd_original()
Haslach_supermarkets_gdf = Haslach_supermarkets.get_geodata_gpd_original()
Haslach_supermarkets_gdf_iso = Haslach_supermarkets.get_isochrones_gdf()
# Extraction geopandas.GeoDataFrames

map_with_basemap(
    layers = [
        Haslach_supermarkets_gdf_iso,
        Haslach_gdf, 
        Haslach_supermarkets_gdf
        ],
    styles={
        0: {
            "name": "Isochrones",
            "color": {
                "segm_min": {
                    5: "midnightblue", 
                    10: "blue",
                    }
                },            
            "alpha": 0.3
        },
        1: {
            "name": "Districts",
            "color": "black",
            "alpha": 1,
            "size": 15,
        },
        2: {
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
        }
        },
    output_filepath = "Haslach_map.png"
    )
# Map with three layers and OSM basemap


# Distance matrix for point gdfs (stand-alone):

Haslach_SHP = gp.read_file("data/Haslach.shp")
Haslach_supermarkets_SHP  = gp.read_file("data/Haslach_supermarkets.shp")

distance_matrix = distance_matrix_from_gdf(
    sources_points_gdf=Haslach_SHP,
    sources_uid_col="BEZEICHN",
    destinations_points_gdf=Haslach_supermarkets_SHP,
    destinations_uid_col="LFDNR",
    output_crs="EPSG:31467"
    )

print(distance_matrix)