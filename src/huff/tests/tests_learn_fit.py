#-----------------------------------------------------------------------
# Name:        tests_learn_fit (huff package)
# Purpose:     Tests for ML models in the Huff Model package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-05-06 21:34
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

from huff.data_management import load_interaction_matrix

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

print("\n=== Summary of MCI model ===\n")
Wieland2015_fit.summary()

print("\n=== Logs of MCI model ===\n")
Wieland2015_fit.show_log()

print("\n=== Summary of interaction matrix after MCI fitting ===\n")
Wieland2015_interaction_matrix.summary()

print("\n=== Logs of interaction matrix after MCI fitting ===\n")
Wieland2015_interaction_matrix.show_log()



Wieland2015_mlfit = Wieland2015_interaction_matrix.learn_fit(
    cols=[
        "A_j", 
        "t_ij", 
        "K", 
        "K_KKr"
        ],
    lct = True,
    model_type = "mlp",
    verbose=True
    )

print("\n=== Summary of interaction matrix after ML fitting ===\n")
Wieland2015_interaction_matrix.summary()

print("\n=== Logs of interaction matrix after ML fitting ===\n")
Wieland2015_interaction_matrix.show_log()

Wieland2015_mlfit.probabilities()

print("\n=== Summary of ML model interaction matrix after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.summary()

print("\n=== Logs of ML model interaction matrix after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.show_log()

print("\n=== Summary of LM model ===\n")
Wieland2015_mlfit.summary()

print("\n=== Logs of LM model ===\n")
Wieland2015_mlfit.show_log()

print("\n=== Summary of interaction matrix - supply locations after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.supply_locations.summary()
print("\n=== Logs of interaction matrix - supply locations after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.supply_locations.show_log()

print("\n=== Summary of interaction matrix - customer origins after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.customer_origins.summary()
print("\n=== Logs of interaction matrix - customer origins after ML fitting ===\n")
Wieland2015_mlfit.interaction_matrix.customer_origins.show_log()
