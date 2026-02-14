#-----------------------------------------------------------------------
# Name:        Example_05_Accessibility_analysis (huff package)
# Purpose:     Example 05: Accessibility analysis with Two-step floating catchment areas and Hansen accessibility
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-14 11:28
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

# This example shows a workflow of a Maximum Likelihood Huff model calibration:
# 1) Import city districts as customer origins
# 2) Import pediatricians as supply locations
# 3) Creating an interaction matrix based on customer origins and supply locations, including travel times
# 4) Define weightings
# 5) Accessibility analysis I: Two-step floating catchment areas
# 6) Accessibility analysis II: Hansen accessibility

# To run this example, switch to directory 'examples' and type:
# python Example_05_Accessibility_analysis.py


import pandas as pd
import geopandas as gp
from huff.data_management import load_geodata
from huff.models import create_interaction_matrix


# 1) Import city districts as customer origins

Freiburg_Stadtbezirke_SHP = gp.read_file("data/Freiburg_Stadtbezirke_Point.shp")
# Loading shapefile with districts (points)

Freiburg_Stadtbezirke_Einwohner = pd.read_excel("data/Freiburg_Stadtbezirke_Einwohner.xlsx")
Freiburg_Stadtbezirke_Einwohner["nr"] = Freiburg_Stadtbezirke_Einwohner["nr"].astype(str)
# Loading population data and preparing for merge

Freiburg_Stadtbezirke = Freiburg_Stadtbezirke_SHP.merge(
    Freiburg_Stadtbezirke_Einwohner[["nr", "EWU18"]],
    left_on="nr",
    right_on="nr"
)
# Merging with population data
# column "EWU18" = inhabitants < 18 years

Freiburg_Stadtbezirke = load_geodata(
    Freiburg_Stadtbezirke,
    location_type="origins",
    unique_id="name"
    )
# Loading city districs (shapefile)

Freiburg_Stadtbezirke.define_marketsize("EWU18")
# Defining market size variable EWU18 (inhabitants < 18 years)

Freiburg_Stadtbezirke.summary()
# Summary of customer origins

Freiburg_Stadtbezirke.show_log()
# Logs of customer origins


# 2) Import pediatricians as supply locations

Freiburg_KuJAerzte = load_geodata(
    "data/Freiburg_KuJAerzte_Point.shp",
    location_type="destinations",
    unique_id="LfdNr"
    )
# Loading pediatricians in Freiburg (shapefile)


# 3) Creating an interaction matrix based on customer origins and supply locations, including travel times

pediatricians_interactionmatrix = create_interaction_matrix(
    Freiburg_Stadtbezirke,
    Freiburg_KuJAerzte,
    verbose=True
    )
# Creating interaction matrix

pediatricians_interactionmatrix.transport_costs(
    network=True,
    ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
    verbose=True
    )
# Calculate car travel times between districts and pediatricians
# Isochrones are retrieved from OpenRouteService
# REQUIRES INTERNET ACCESS AND VALID API TOKEN


# 4) Define weightings

pediatricians_interactionmatrix.define_weightings(
    vars_funcs={
    0: {
            "name": "A_j",
            "func": "power",
            "param": 1
        },
    1: {
            "name": "t_ij",
            "func": "power",
            "param": -1
        }
    }
)
# Defining weightings: Here, we refrain from any weighting

pediatricians_interactionmatrix.set_attraction_constant()
# Set "attraction" of pediatricians constant = 1

# Based on the defined weightings above, the utility function is here:
# U_ij = A_j ^ 1 * t_ij ^ -1 = 1/t_ij

pediatricians_interactionmatrix.summary()
# Summary of interaction matrix


# 5) Accessibility analysis I: Two-step floating catchment areas

accessibility_2SFCA_calculation = pediatricians_interactionmatrix.floating_catchment(
    threshold=10,
    demand_factor=1000
)
# Two-step floating catchment areas analysis with threshold = 10 (min)
# and pediatricians per 1000 inhabitants < 18y

pediatricians_interactionmatrix.get_interaction_matrix_df().to_excel("pediatricians_interactionmatrix.xlsx")
# Save interaction matrix

pediatricians_interactionmatrix.summary()
# Summary of interaction matrix

pediatricians_interactionmatrix.show_log()
# Logs of interaction matrix

accessibility_2SFCA_calculation.summary()
# Summary of Two-step floating catchment areas analysis

print(accessibility_2SFCA_calculation.get_fca_df())
# Show accessibility values as pandas df


# 6) Accessibility analysis II: Hansen accessibility

accessibility_Hansen_calculation = pediatricians_interactionmatrix.hansen()
# Hansen accessibility analysis with default values

accessibility_Hansen_calculation.summary()
# Summary of Hansen accessibility analysis

print(accessibility_Hansen_calculation.get_hansen_df())
# Show accessibility values as pandas df