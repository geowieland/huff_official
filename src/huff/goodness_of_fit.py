#-----------------------------------------------------------------------
# Name:        goodness_of_fit (huff package)
# Purpose:     Functions for goodness-of-fit statistics
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.3
# Last update: 2025-12-23 14:37
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from pandas.api.types import is_numeric_dtype
from math import sqrt
import huff.config as config


def modelfit(
    observed, 
    expected,
    remove_nan: bool = True,
    perc_factor: int = 100,
    verbose: bool = False
    ):

    observed_no = len(observed)
    expected_no = len(expected)

    assert observed_no == expected_no, "Error while calculating fit metrics: Observed and expected differ in length"
    
    if not isinstance(observed, np.number): 
        if not is_numeric_dtype(observed):
            raise ValueError("Error while calculating fit metrics: Observed column is not numeric")
    if not isinstance(expected, np.number):
        if not is_numeric_dtype(expected):
            raise ValueError("Error while calculating fit metrics: Expected column is not numeric")
    
    if remove_nan:
        
        observed = observed.reset_index(drop=True)
        expected = expected.reset_index(drop=True)

        obs_exp = pd.DataFrame(
            {
                config.DEFAULT_OBSERVED_COL: observed, 
                config.DEFAULT_EXPECTED_COL: expected
                }
            )
        
        obs_exp_clean = obs_exp.dropna(subset=[config.DEFAULT_OBSERVED_COL, config.DEFAULT_EXPECTED_COL])

        if len(obs_exp_clean) < len(observed) or len(obs_exp_clean) < len(expected):
            if verbose:
                print("NOTE: Vectors 'observed' and/or 'expected' contain NaNs which are dropped.")
        
        observed = obs_exp_clean[config.DEFAULT_OBSERVED_COL].to_numpy()
        expected = obs_exp_clean[config.DEFAULT_EXPECTED_COL].to_numpy()
    
    else:
        
        if np.isnan(observed).any():
            raise ValueError("Error while calculating fit metrics: Vector with observed data contains NaNs and 'remove_nan' is False")
        if np.isnan(expected).any():
            raise ValueError("Error while calculating fit metrics: Vector with expected data contains NaNs and 'remove_nan' is False")
       
    residuals = np.array(observed)-np.array(expected)
    residuals_sq = residuals**2
    residuals_abs = abs(residuals)
 
    if any(observed == 0):
        if verbose:
            print ("Vector 'observed' contains values equal to zero. No APE/MAPE calculated.")
        APE = np.full_like(observed, np.nan)
        MAPE = None
    else:
        APE = abs(observed-expected)/observed*perc_factor
        MAPE = float(np.mean(APE))
        
    sAPE = abs(observed-expected)/((abs(observed)+abs(expected))/2)*perc_factor
    
    data_residuals = pd.DataFrame({
        config.DEFAULT_OBSERVED_COL: observed,
        config.DEFAULT_EXPECTED_COL: expected,
        "residuals": residuals,
        "residuals_sq": residuals_sq,
        "residuals_abs": residuals_abs,
        config.APE_PREFIX: APE,
        f"s{config.APE_PREFIX}": sAPE
        })

    SQR = float(np.sum(residuals_sq))
    SAR = float(np.sum(residuals_abs))    
    observed_mean = float(np.sum(observed)/observed_no)
    SQT = float(np.sum((observed-observed_mean)**2))
    Rsq = float(1-(SQR/SQT))
    MSE = float(SQR/observed_no)
    RMSE = float(sqrt(MSE))
    MAE = float(SAR/observed_no)
    LL = np.sum(np.log(residuals_sq))
    
    sMAPE = float(np.mean(sAPE))

    APEs = {}
    i = 0
   

    for APE_value in range(config.APE_MIN, config.APE_MAX + 1):
        i = i+1
        APEs[f"{config.APE_PREFIX}{APE_value}"] = float(len(data_residuals[data_residuals[config.APE_PREFIX] < APE_value])/expected_no*perc_factor)

    data_lossfunctions = {
        config.GOODNESS_OF_FIT["Sum of squared residuals"]: SQR,
        config.GOODNESS_OF_FIT["Sum of absolute residuals"]: SAR,
        config.GOODNESS_OF_FIT["R-squared"]: Rsq,
        config.GOODNESS_OF_FIT["Mean squared error"]: MSE,
        config.GOODNESS_OF_FIT["Root mean squared error"]: RMSE,
        config.GOODNESS_OF_FIT["Mean absolute error"]: MAE,
        config.GOODNESS_OF_FIT["Mean absolute percentage error"]: MAPE,
        config.GOODNESS_OF_FIT["Symmetric MAPE"]: sMAPE,
        config.GOODNESS_OF_FIT["Negative log-likelihood"]: -LL,
        **APEs          
    }    
    
    modelfit_results = [
        data_residuals,
        data_lossfunctions
    ]

    return modelfit_results

def modelfit_plot(
    observed_expected: list = [],     
    remove_nan: bool = True,
    perc_factor: int = 100,
    title: str = "Observed vs. expected",
    x_lab: str = "Observed",
    y_lab: str = "Expected",
    points_cols: list = [],
    points_alpha = 0.5,
    figsize = (8,6),
    show_diag: list = ["MAPE", "Rsq"],
    round_float: int = 1,
    label_prefixes: list = [],
    grid: bool = True,
    diagonale: bool = True,
    diagonale_col = "black",
    legend_fontsize = "small",
    save_as: str = "scatterplot.png",
    save_dpi = 300,
    show_plot: bool = False,
    verbose: bool = False
    ):
    
    modelfit_list = []
    
    for obs_exp_data in observed_expected:
        
        obs_exp_data_modelfit = modelfit(
            observed = obs_exp_data[0],
            expected = obs_exp_data[1],
            remove_nan = remove_nan,
            perc_factor = perc_factor,
            verbose = verbose
            )
         
        modelfit_list.append(obs_exp_data_modelfit)
    
    values_no = len(observed_expected[0][0])
    all_values = np.concatenate([entry[0][config.DEFAULT_OBSERVED_COL] for entry in modelfit_list] + [entry[0][config.DEFAULT_EXPECTED_COL] for entry in modelfit_list])
    min_value = np.min(all_values)
    max_value = np.max(all_values)    
    
    plt.figure(figsize=figsize)
    
    if diagonale:
        diagonal = np.linspace(
            0, 
            max_value, 
            values_no
            )
        plt.plot(
            diagonal, 
            diagonal, 
            color=diagonale_col
            )
    
    for i, entry in enumerate(modelfit_list):
        
        if len(label_prefixes) == len(modelfit_list):
            label = label_prefixes[i]
        
        for key in entry[1].keys():
            if key in show_diag:
                label = f"{label} {key}={round(entry[1][key], round_float)} "
                
        if len(points_cols) == len(modelfit_list):
            color = points_cols[i]
        else:
            color = random.choice(list(mcolors.CSS4_COLORS.keys()))
        
        plt.scatter(
            entry[0][config.DEFAULT_OBSERVED_COL],
            entry[0][config.DEFAULT_EXPECTED_COL],
            color=color,
            alpha=points_alpha,
            label=label
        )
        
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)    
    plt.legend(fontsize=legend_fontsize)
    if grid:
        plt.grid(True)
    
    if show_plot:
        plt.show()
    
    if save_as is not None:
        plt.savefig(save_as, dpi=save_dpi)
     
    return modelfit_list


def modelfit_print(modelfit_results):

    maxlen = max(len(str(key)) for key in config.GOODNESS_OF_FIT.keys())

    for gof_key, gof_value in config.GOODNESS_OF_FIT.items():
                    
        if gof_key in config.GOODNESS_OF_FIT.keys():                        
        
            if modelfit_results[1][gof_value] is not None:
                print(f"{gof_key:<{maxlen}}  {round(modelfit_results[1][gof_value], 2)}")

    return modelfit_results


def modelfit_cat(
    observed,
    expected,
    remove_nan: bool = True,
    perc_factor: int = 100,
    verbose: bool = False
    ):

    observed_no = len(observed)
    expected_no = len(expected)

    assert observed_no == expected_no, "Error while calculating fit metrics: Observed and expected differ in length"
    
    if not isinstance(observed, np.number): 
        if not is_numeric_dtype(observed):
            raise ValueError("Error while calculating fit metrics: Observed column is not numeric")
    if not isinstance(expected, np.number):
        if not is_numeric_dtype(expected):
            raise ValueError("Error while calculating fit metrics: Expected column is not numeric")
    
    if remove_nan:
        
        observed = observed.reset_index(drop=True)
        expected = expected.reset_index(drop=True)

        obs_exp = pd.DataFrame(
            {
                config.DEFAULT_OBSERVED_COL: observed, 
                config.DEFAULT_EXPECTED_COL: expected
                }
            )
        
        obs_exp_clean = obs_exp.dropna(subset=[config.DEFAULT_OBSERVED_COL, config.DEFAULT_EXPECTED_COL])

        if len(obs_exp_clean) < len(observed) or len(obs_exp_clean) < len(expected):
            if verbose:
                print("NOTE: Vectors 'observed' and/or 'expected' contain NaNs which are dropped.")
        
        observed = obs_exp_clean[config.DEFAULT_OBSERVED_COL].to_numpy()
        expected = obs_exp_clean[config.DEFAULT_EXPECTED_COL].to_numpy()
    
    else:
        
        if np.isnan(observed).any():
            raise ValueError("Error while calculating fit metrics: Vector with observed data contains NaNs and 'remove_nan' is False")
        if np.isnan(expected).any():
            raise ValueError("Error while calculating fit metrics: Vector with expected data contains NaNs and 'remove_nan' is False")
        
    data_residuals = pd.DataFrame(
        {
            config.DEFAULT_OBSERVED_COL: observed,
            config.DEFAULT_EXPECTED_COL: expected,
        }
        )
    data_residuals["fit"] = 0
    data_residuals.loc[data_residuals[config.DEFAULT_OBSERVED_COL] == data_residuals[config.DEFAULT_EXPECTED_COL], "fit"] = 1
    
    TP = np.sum((observed == 1) & (expected == 1))
    FP = np.sum((observed == 0) & (expected == 1))
    TN = np.sum((observed == 0) & (expected == 0))
    FN = np.sum((observed == 1) & (expected == 0))
    
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0 
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0 
    acc = (TP + TN) / (TP + TN + FP + FN)
    
    data_lossfunctions = {
        "sens": sens*perc_factor,
        "spec": spec*perc_factor,
        "acc": acc*perc_factor,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,        
     }

    modelfit_results = [
        data_residuals,
        data_lossfunctions
    ]

    return modelfit_results