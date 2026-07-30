#-----------------------------------------------------------------------
# Name:        tests_predictive (huff package)
# Purpose:     Tests for predictive_models module in the Huff Model package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.0
# Last update: 2026-07-30 19:20
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------

import pandas as pd
from huff.predictive_models import model_wrapper, models_wrapper


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

print(ols_result.predict())

print(
    ols_result.predict(
        df=Wieland2015_interaction_matrix, 
        X_cols=[
            "VF", 
            "K", 
            "K_KKr",
            "Dist_Min2",        
        ]
        )
    )

# All three models at once:
predictive_models = models_wrapper(
    y,
    X,
    models = {
        "ols": {            
        },
        "xgb": {            
        },
        "mlp": {
            "activation": "tanh",
            "hidden_layer_sizes": (5,10),
            "solver": "adam"
        }
    },
    random_state = 71,
    verbose = True
)

print(predictive_models.y_test_models)

print(predictive_models.models_fit_metrics_df)