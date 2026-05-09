#-----------------------------------------------------------------------
# Name:        tests_predictive (huff package)
# Purpose:     Tests for predictive_models module in the Huff Model package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-05-06 21:33
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

import pandas as pd
from huff.predictive_models import model_wrapper


Wieland2015_interaction_matrix = pd.read_excel("data/Wieland2015.xlsx")

y = Wieland2015_interaction_matrix["MA_Anb1"]
X = Wieland2015_interaction_matrix[
    [
        "VF", 
        "K", 
        "K_KKr",
        "Dist_Min2",        
    ]
]

xgb_result = model_wrapper(
    y,
    X,
    model_type = "xgb"
)

xgb_result.summary()

ann_result = model_wrapper(
    y,
    X,
    model_params={
        "activation": "tanh",
        "hidden_layer_sizes": (5,10),
        "solver": "adam"
    },
    model_type = "mlp"
)

ann_result.summary()

print(ann_result.data)

ols_result = model_wrapper(
    y,
    X,
    model_type = "ols"
)

ols_result.summary()