#-----------------------------------------------------------------------
# Name:        goodness_of_fit (huff package)
# Purpose:     Functions for goodness-of-fit statistics
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.5
# Last update: 2026-02-08 12:56
# Copyright (c) 2024-2026 Thomas Wieland
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

    """
    Compute goodness-of-fit metrics for observed and expected values.

    Parameters
    ----------
    observed : array-like
        One-dimensional numeric vector containing observed (true) values.
        Supported types include NumPy arrays and pandas Series.
    expected : array-like
        One-dimensional numeric vector containing expected (predicted) values.
        Must have the same length as `observed`.
    remove_nan : bool, optional
        If True (default), rows containing NaN values in either `observed`
        or `expected` are removed prior to computation. If False, the
        presence of NaNs raises a ValueError.
    perc_factor : int, optional
        Scaling factor for percentage-based error metrics (default is 100).
    verbose : bool, optional
        If True, print informational messages during processing.

    Returns
    -------
    data_residuals : pandas.DataFrame
        DataFrame containing observed values, expected values, residuals,
        squared residuals, absolute residuals, absolute percentage error
        (APE), and symmetric APE (sAPE) for each observation.
    data_lossfunctions : dict
        Dictionary containing aggregated goodness-of-fit metrics, including
        R-squared, MAE, MSE, RMSE, MAPE, sMAPE, negative log-likelihood, and
        APE-threshold statistics.

    Raises
    ------
    ValueError
        If `perc_factor` <= 0.
    AssertionError
        If `observed` and `expected` differ in length.
    ValueError
        If input data are non-numeric or contain NaN values while
        `remove_nan` is False.

    Notes
    -----
    - Mean Absolute Percentage Error (MAPE) is not computed if any observed
      value equals zero.

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> exp = np.array([1.1, 1.9, 3.2])
    >>> residuals, metrics = modelfit(obs, exp)
    >>> metrics["Root mean squared error"]
    """
    
    if perc_factor <= 0:
        raise ValueError("Parameter 'perc_factor' must be positive. Use perc_factor = 100 to get deviation-based metrics in percent")

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
    observed_expected: list | None = None,     
    remove_nan: bool = True,
    perc_factor: int = 100,
    title: str = "Observed vs. expected",
    x_lab: str = "Observed",
    y_lab: str = "Expected",
    points_cols: list | None = None,
    points_alpha: float = 0.5,
    figsize: tuple = (8,6),
    show_diag: list | None = None,
    round_float: int = 1,
    label_prefixes: list | None = None,
    grid: bool = True,
    diagonale: bool = True,
    diagonale_col = "black",
    legend_fontsize = "small",
    save_as: str = "scatterplot.png",
    save_dpi: int = 300,
    show_plot: bool = False,
    verbose: bool = False
    ):
    
    """
    Create an observed-vs-expected scatter plot with goodness-of-fit diagnostics.

    This function computes model fit statistics for one or multiple pairs of
    observed and expected values using `modelfit` and visualizes them in a
    scatter plot. Each dataset can be displayed with individual colors and
    labels, optionally annotated with selected goodness-of-fit metrics.

    Parameters
    ----------
    observed_expected : list of tuple, optional
        List of (observed, expected) pairs. Each element must be a tuple
        containing two one-dimensional numeric vectors of equal length.
        Supported types include NumPy arrays and pandas Series.
    remove_nan : bool, optional
        If True (default), rows containing NaN values are removed prior to
        computing fit metrics. If False, NaNs raise a ValueError.
    perc_factor : int, optional
        Scaling factor for percentage-based error metrics (default is 100).
    title : str, optional
        Plot title.
    x_lab : str, optional
        Label for the x-axis (observed values).
    y_lab : str, optional
        Label for the y-axis (expected values).
    points_cols : list, optional
        List of colors for scatter points. If provided, its length must match
        the number of datasets. 
        Colors are randomly assigned if `points_cols` is not provided.
    points_alpha : float, optional
        Alpha (transparency) value for scatter points.
    figsize : tuple, optional
        Figure size passed to matplotlib (width, height).
    show_diag : list, optional
        List of goodness-of-fit metric names to include in the legend
        (e.g., ["MAPE", "Rsq"]; available metrics: see config.GOODNESS_OF_FIT.values()).
    round_float : int, optional
        Number of decimal places used when rounding diagnostic values
        displayed in the legend.
    label_prefixes : list, optional
        List of label prefixes for each dataset shown in the legend.
    grid : bool, optional
        If True, display a grid on the plot.
    diagonale : bool, optional
        If True, plot the 1:1 diagonal line.
    diagonale_col : str, optional
        Color of the diagonal reference line.
    legend_fontsize : str or int, optional
        Font size of the legend.
    save_as : str or None, optional
        File path for saving the plot. If None, the plot is not saved.
    save_dpi : int, optional
        DPI used when saving the figure.
    show_plot : bool, optional
        If True, display the plot using matplotlib.
    verbose : bool, optional
        If True, print informational messages during processing.

    Returns
    -------
    modelfit_list : list
        List of results returned by `modelfit` for each (observed, expected)
        pair. Each element contains a residuals DataFrame and a dictionary
        of goodness-of-fit metrics.

    Examples
    --------
    >>> data = [(obs1, exp1), (obs2, exp2)]
    >>> results = modelfit_plot(
    ...     observed_expected=data,
    ...     show_diag=["MAPE", "Rsq"],
    ...     show_plot=True
    ... )
    """

    if observed_expected is None:
        observed_expected = []
    if points_cols is None:
        points_cols = []
    if show_diag is None:
        show_diag = ["MAPE", "Rsq"]
    if label_prefixes is None:
        label_prefixes = []
        
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

def modelfit_cat(
    observed,
    expected,
    remove_nan: bool = True,
    perc_factor: int = 100,
    verbose: bool = False
    ):

    """
    Compute categorical goodness-of-fit statistics for observed and expected binary values.

    Parameters
    ----------
    observed : array-like
        Vector of observed binary values (0/1).
    expected : array-like
        Vector of expected binary values (0/1).
    remove_nan : bool, optional
        If True, remove pairs with NaN values before calculation (default: True).
    perc_factor : int, optional
        Factor used to scale percentage-based metrics (default: 100).
    verbose : bool, optional
        If True, print informational messages during processing.

    Returns
    -------
    list
        A list containing a residuals DataFrame and a dictionary with goodness-of-fit
        measures (sensitivity, specificity, accuracy, ni-information rate, 
        true positives, false positives, true negatives, false negatives).

    Raises
    ------
    ValueError
        If `perc_factor` <= 0.
    AssertionError
        If `observed` and `expected` differ in length.
    ValueError
        If input data are non-numeric or contain NaN values while
        `remove_nan` is False.

    Examples
    --------
    >>> observed = np.array([1, 0, 1, 1, 0])
    >>> expected = np.array([1, 0, 0, 1, 0])
    >>> results = modelfit_cat(observed, expected)
    >>> results[1]["acc"]
    """

    if perc_factor <= 0:
        raise ValueError("Parameter 'perc_factor' must be positive. Use perc_factor = 100 to get deviation-based metrics in percent")

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
    nir = TN / (TP + TN + FP + FN)
    
    data_lossfunctions = {
        "sens": sens*perc_factor,
        "spec": spec*perc_factor,
        "acc": acc*perc_factor,
        "nir": nir*perc_factor,
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