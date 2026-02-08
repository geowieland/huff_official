#-----------------------------------------------------------------------
# Name:        tests_accessibility (huff package)
# Purpose:     Tests for Huff Model package accessibility analysis
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-04 18:58
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------

import pandas as pd
import geopandas as gp
from huff.data_management import load_geodata
from huff.models import create_interaction_matrix

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

Freiburg_KuJAerzte = load_geodata(
    "data/Freiburg_KuJAerzte_Point.shp",
    location_type="destinations",
    unique_id="LfdNr"
    )
# Loading pediatricians in Freiburg (shapefile)

pediatricians_interactionmatrix = create_interaction_matrix(
    Freiburg_Stadtbezirke,
    Freiburg_KuJAerzte,
    verbose=True
    )
# Creating interaction matrix

pediatricians_interactionmatrix.transport_costs(
    network=False,
    verbose=True
    )
# Calculate distances between districts and pediatricians

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
# Defining weightings

pediatricians_interactionmatrix.set_attraction_constant()
# Set "attraction" of pediatricians constant = 1

pediatricians_interactionmatrix.summary()
# Summary of interaction matrix

accessibility_2SFCA_calculation = pediatricians_interactionmatrix.floating_catchment(
    threshold=1,
    demand_factor=1000
)
# Two-step floating catchment areas analysis with threshold = 1 (km)
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

accessibility_Hansen_calculation = pediatricians_interactionmatrix.hansen()
# Hansen accessibility analysis with default values

accessibility_Hansen_calculation.summary()
# Summary of Hansen accessibility analysis

print(accessibility_Hansen_calculation.get_hansen_df())
# Show accessibility values as pandas df