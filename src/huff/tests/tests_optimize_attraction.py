#-----------------------------------------------------------------------
# Name:        tests_optimize_attraction (huff package)
# Purpose:     Tests for local optimization of attraction variable
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.1
# Last update: 2026-04-19 12:34
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

from huff.data_management import load_geodata, load_interaction_matrix, load_marketareas
from huff.helper import print_timestamp


Wieland2015_interaction_matrix3 = load_interaction_matrix(
    data="data/Wieland2015.xlsx",
    customer_origins_col="Quellort",
    supply_locations_col="Zielort",
    attraction_col=["VF"],
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

Wieland2015_interaction_matrix3.define_weightings(
    vars_funcs = {
        0: {
            "name": "A_j",
            "func": "power",
            "param": 0.9
        },
        1: {
            "name": "t_ij",
            "func": "power",
            "param": -1.9
        },
    }
)
# Defining weighting functions

huff_model3 = Wieland2015_interaction_matrix3.marketareas()
# Calculation of market areas

wieland2015_totalmarketareas = load_marketareas(
    data="data/Wieland2015.xlsx",
    supply_locations_col="Zielort",
    total_col="Anb_Eink",
    data_type="xlsx",
    xlsx_sheet="total_marketareas"
)
# Loading empirical total market areas

huff_model3 = wieland2015_totalmarketareas.add_to_model(
    huff_model3
    )
# Adding total market areas to HuffModel object

huff_model3.optimize_attraction(
    verbose=True, 
    iterations=10
    )
# Local optimization of attraction variable in 10 iterations

huff_model3.summary()
# Summary of HuffModel object after optimization

print(huff_model3.get_market_areas_df())
# Show market areas after optimization

print(huff_model3.interaction_matrix.get_interaction_matrix_df())
# Show interaction matrix after optimization

print_timestamp(huff_model3)
# Print timestamps of HuffModel object after optimization