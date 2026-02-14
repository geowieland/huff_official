#-----------------------------------------------------------------------
# Name:        Example_04_Huff_model_calibration_total_market_areas (huff package)
# Purpose:     Example 04: Maximum Likelihood Huff model calibration based on observed total market areas
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-14 09:08
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

# This example shows a workflow of a Maximum Likelihood Huff model calibration:
# 1) Import an existing interaction matrix including travel times
# 2) Defining weighting functions
# 3) Import total market areas and link them to the interaction matrix
# 4) Parameter estimation via Maximum Likelihood

# To run this example, switch to directory 'examples' and type:
# python Example_04_Huff_model_calibration_total_market_areas.py


from huff.data_management import load_interaction_matrix, load_marketareas


# 1) Import an existing interaction matrix including travel times

# Loading own interaction matrix (Market areas of consumer electronics stores)
# Data source: Wieland 2015 (https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753)
interaction_matrix_CE = load_interaction_matrix(
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
# Indices of customer origins are in the column "Quellort"
# Indices of supply locations are in the column "Zielort"
# Three attraction variables: "VF", "K", and "K_KKr"
# Travel times (minutes by car) are stored in the column "Dist_Min2"
# Empirical market shares are in the column "MA_Anb1"


# 2) Defining weighting functions

interaction_matrix_CE.define_weightings(
    vars_funcs = {
        0: {
            "name": "A_j",
            "func": "power"
        },
        1: {
            "name": "t_ij",
            "func": "exponential",               
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


# 3) Import total market areas and link them to the interaction matrix

wieland2015_totalmarketareas = load_marketareas(
    data="data/Wieland2015.xlsx",
    supply_locations_col="Zielort",
    total_col="Anb_Eink",
    data_type="xlsx",
    xlsx_sheet="total_marketareas"
)
# Loading empirical total market areas

CE_huff_fit = wieland2015_totalmarketareas.add_to_model(interaction_matrix_CE)
# Adding total market areas to InteractionMatrix object
# Result is an instance of class HuffModel

print(CE_huff_fit.get_market_areas_df())
# Showing total market areas of HuffModel object

huff_model_fit = CE_huff_fit.ml_fit(
    initial_params=[0.9, -0.3, 0.5, 0.3],
    bounds=[(0.5, 1), (-0.5, -0.1), (0.2, 0.7), (0.2, 0.7)],
    fit_by="totals"
    )
# ML fit with power weighting functions for the attraction variables
# and exponential transport costs weighting based on total market areas
# from HuffModel object
# initial_params and bounds are passed to scipy.optimize.minimize
# with chosen optimization algorithm (default: method = "L-BFGS-B")

huff_model_fit.summary()
# Huff model summary including estimated coefficients and fit metrics

huff_model_fit.show_log()
# Show logs of HuffModel object