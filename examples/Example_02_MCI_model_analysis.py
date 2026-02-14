#-----------------------------------------------------------------------
# Name:        Example_02_MCI_model_analysis (huff package)
# Purpose:     Example 02: MCI model analysis with existing interaction matrix
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-14 08:36
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------


# This example shows a workflow of a MCI model analysis:
# 1) Import an existing interaction matrix including travel times
# 2) Parameter estimation via MCI model
# 3) Show summary and inspect model fit metrics

# To run this example, switch to directory 'examples' and type:
# python Example_02_MCI_model_analysis.py


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

interaction_matrix_CE.summary()
# Summary of interaction matrix


# 2) Parameter estimation via MCI model

CE_mci_fit = interaction_matrix_CE.mci_fit(
    cols=[
        "A_j", 
        "t_ij", 
        "K", 
        "K_KKr"
        ]
    )
# Fitting MCI model with the four independent variables


# 3) Show summary and inspect model fit metrics

CE_mci_fit.probabilities(verbose=True)
# Calculating probabilities with estimated weighting parameters

CE_mci_fit.summary()
# MCI model summary including estimated coefficients and fit metrics

interaction_matrix_CE.show_log()
# Logs of InteractionMatrix object