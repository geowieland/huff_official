#-----------------------------------------------------------------------
# Name:        Example_01_Basic_Huff_model_analysis (huff package)
# Purpose:     Example 01: Basic Huff model analysis
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-13 18:00
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------


# This example shows a workflow of a basic Huff model analysis:
# 1) Import of customer origins and definition of their attributes and weightings
# 2) Import of supply locations and definition of their attributes and weightings
# 3) Creating an interaction matrix based on customer origins and supply locations, including travel times
# 4) Basic Huff model analysis: calculation of customer flows and total market areas
# 5) Plotting expected customer flows in a map, export as image file (.png)

# To run this example, switch to directory 'examples' and type:
# python Example_01_Basic_Huff_model_analysis.py


from huff.models import create_interaction_matrix
from huff.data_management import load_geodata


# 1) Import of customer origins and definition of their attributes and weightings

# Loading customer origins (districts of Freiburg-Haslach):
Haslach_customer_origins = load_geodata(
    data = "data/Haslach.shp",
    location_type="origins",
    unique_id="BEZEICHN"
    )
# Parameter 'location_type' states which type of data is imported (here: "origins")
# Parameter 'unique_id' is the column with the unique identifier (here: "BEZEICHN")
# Resulting object "Haslach_customer_origins" is of class CustomerOrigins

# Defining market size of customer origins:
Haslach_customer_origins.define_marketsize("pop")
# Column with market size (=customer/expenditure potential) in the original data (here: "pop")

# Defining transport costs weighting:
Haslach_customer_origins.define_transportcosts_weighting(
    func = "power",
    param_lambda = -2.2,    
    )
# Power function (func = "power") with a weighting exponent lambda equal to -2.2

Haslach_customer_origins.summary()
# Summary of customer origins after updates

Haslach_customer_origins.show_log()
# Show log of Haslach_customer_origins


# 2) Import of supply locations and definition of their attributes and weightings

# Loading supply locations (grocery stores in Haslach):
Haslach_supermarkets = load_geodata(
    data = "data/Haslach_supermarkets.shp",
    location_type="destinations",
    unique_id="LFDNR"
    )
# Parameter 'location_type' states which type of data is imported (here: "destinations")
# Parameter 'unique_id' is the column with the unique identifier (here: "LFDNR")

Haslach_supermarkets.define_attraction("VKF_qm")
# Column with attraction variable (such as size) in the original data (here: "VKF_qm")

# Define attraction weighting:
Haslach_supermarkets.define_attraction_weighting(
    func = "power",
    param_gamma=0.9
    )
# Power function (func = "power") with a weighting exponent gamma equal to 0.9

Haslach_supermarkets.summary()
# Summary of supply locations after updates

Haslach_supermarkets.show_log()
# Show log of Haslach_customer_origins


# 3) Creating an interaction matrix based on customer origins and supply locations, including travel times

# Using customer origins and supply locations for building interaction matrix:
haslach_interactionmatrix = create_interaction_matrix(
    Haslach_customer_origins,
    Haslach_supermarkets
    )
# Creating interaction matrix

haslach_interactionmatrix.transport_costs(
    ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
    network=True,
    profile="driving-car"    
    )
# Obtaining transport costs (default: driving-car)
# Set network = True to calculate transport costs matrix via ORS API (default)
# ORS API documentation: https://openrouteservice.org/dev/#/api-docs/v2/

haslach_interactionmatrix.summary()
# Summary of interaction matrix


# 4) Basic Huff model analysis: calculation of customer flows and total market areas

haslach_interactionmatrix.flows()
# Calculating spatial flows for interaction matrix

# Based on the defined weightings above, the Huff utility function is here:
# U_ij = A_j ^ 0.9 * t_ij ^ -2.2
# The 'flows()' function calculates utilities, probabilities, and expected customer flows

haslach_interactionmatrix.show_log()
# Show log of haslach_interactionmatrix

huff_model = haslach_interactionmatrix.marketareas()
# Calculating total market areas
# Result of class HuffModel

huff_model.summary()
# Summary of Huff model

print(huff_model.get_market_areas_df())
# Showing total market areas


# 5) Plotting expected customer flows in a map, export as image file (.png)

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
    line_size_by = "flows",   
    )
# Plot expected customer flows (line_size_by = "flows") as flow map
# Customer origins as black points, and supply locations colored by grocery chain