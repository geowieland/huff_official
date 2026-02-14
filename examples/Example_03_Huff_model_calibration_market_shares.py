#-----------------------------------------------------------------------
# Name:        Example_03_Huff_model_calibration_market_shares (huff package)
# Purpose:     Example 03: Maximum Likelihood Huff model calibration based on observed market shares
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-14 08:41
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------


# This example shows a workflow of a Maximum Likelihood Huff model calibration:
# 1) Import an existing interaction matrix including travel times
# 2) Defining weighting functions
# 3) Parameter estimation via Maximum Likelihood
# 4) Calculation of total market areas


# To run this example, switch to directory 'examples' and type:
# python Example_03_Huff_model_calibration_market_shares.py


from huff.data_management import load_interaction_matrix


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
            "func": "power",               
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


# 3) Parameter estimation via Maximum Likelihood

interaction_matrix_CE.huff_ml_fit(
    initial_params=[0.9, -1.5, 0.5, 0.3],
    bounds=[(0.5, 1), (-2, -1), (0.2, 0.7), (0.2, 0.7)],
    fit_by="probabilities"
)
# ML fit with power weighting function based on probabilities
# from InteractionMatrix object
# initial_params and bounds are passed to scipy.optimize.minimize
# with chosen optimization algorithm (default: method = "L-BFGS-B")


interaction_matrix_CE.summary()
# Summary of interaction matrix

interaction_matrix_CE.show_log()
# Logs of InteractionMatrix object


# 4) Calculation of total market areas

CE_huff_fit = interaction_matrix_CE.marketareas()
# Calculation of market areas

CE_huff_fit.summary()
# Huff model summary including estimated coefficients and fit metrics

CE_huff_fit.show_log()
# Show logs of HuffModel object