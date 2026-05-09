#-----------------------------------------------------------------------
# Name:        predictive_models (huff package)
# Purpose:     Creating ML predictive models for Market Area Analyses
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-05-02 14:20
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------

from statsmodels.formula.api import ols
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import huff.config as config
from huff.goodness_of_fit import modelfit
from huff.helper import create_timestamp, print_modelfit


class PredictiveModel:

    """
    Container for predictive model results.

    Parameters
    ----------
    y_pred : array-like or None
        Predicted target values for the test set.
    model : object
        Trained model object (e.g. scikit-learn regressor).
    fit_metrics : dict or pandas.DataFrame or None
        Goodness-of-fit metrics produced by ``modelfit``.
    params : dict
        Parameters used for modelling (e.g. ``model_type``, ``model_params``).
    data : dict
        Dictionary containing ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    runtime_error : str or None
        Runtime error message if training or prediction failed.
    analysis_description : str
        Short description of the analysis.
    timestamp : str
        Timestamp or metadata about creation.

    Returns
    -------
    None
        Class instantiation returns an instance of ``PredictiveModel``.

    Examples
    --------
    >>> import pandas as pd
    >>> from huff.predictive_models import model_wrapper
    >>> Wieland2015_interaction_matrix = pd.read_excel("data/Wieland2015.xlsx")
    >>> y = Wieland2015_interaction_matrix["MA_Anb1"]
    >>> X = Wieland2015_interaction_matrix[
    ...     [
    ...         "VF", 
    ...         "K", 
    ...         "K_KKr",
    ...         "Dist_Min2",        
    ...     ]
    ... ]
    >>> xgb_result = model_wrapper(
    ...     y,
    ...     X,
    ...     model_type = "xgb"
    ... )
    >>> xgb_result.summary()
    >>> ann_result = model_wrapper(
    ...     y,
    ...     X,
    ...     model_params={
    ...         "activation": "tanh",
    ...         "hidden_layer_sizes": (5,10),
    ...         "solver": "adam"
    ...     },
    ...     model_type = "mlp"
    ... )
    >>> ann_result.summary()
    """

    def __init__(
        self,
        y_pred,
        model,
        fit_metrics,
        params,
        data,
        runtime_error,
        analysis_description,
        timestamp
        ):
        
        """
        Initialize PredictiveModel container.

        Parameters
        ----------
        y_pred : array-like or None
            Predicted target values for the test set.
        model : object
            Trained model object (e.g. scikit-learn regressor).
        fit_metrics : dict or pandas.DataFrame or None
            Goodness-of-fit metrics produced by ``modelfit``.
        params : dict
            Parameters used for modelling (e.g. ``model_type``, ``model_params``).
        data : dict
            Dictionary containing ``X_train``, ``X_test``, ``y_train``, ``y_test``.
        runtime_error : str or None
            Runtime error message if training or prediction failed.
        analysis_description : str
            Short description of the analysis.
        timestamp : str
            Timestamp or metadata about creation.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> from huff.predictive_models import model_wrapper
        >>> Wieland2015_interaction_matrix = pd.read_excel("data/Wieland2015.xlsx")
        >>> y = Wieland2015_interaction_matrix["MA_Anb1"]
        >>> X = Wieland2015_interaction_matrix[
        ...     [
        ...         "VF", 
        ...         "K", 
        ...         "K_KKr",
        ...         "Dist_Min2",        
        ...     ]
        ... ]
        >>> xgb_result = model_wrapper(
        ...     y,
        ...     X,
        ...     model_type = "xgb"
        ... )
        >>> xgb_result.summary()
        >>> ann_result = model_wrapper(
        ...     y,
        ...     X,
        ...     model_params={
        ...         "activation": "tanh",
        ...         "hidden_layer_sizes": (5,10),
        ...         "solver": "adam"
        ...     },
        ...     model_type = "mlp"
        ... )
        >>> ann_result.summary()
        """

        self.y_pred = y_pred
        self.model = model
        self.fit_metrics = fit_metrics
        self.params = params
        self.data = data
        self.runtime_error = runtime_error
        self.analysis_description = analysis_description
        self.timestamp = timestamp
        
    def summary(self):

        """
        Print a short, formatted summary of the predictive analysis.

        The method prints the analysis description, used parameters, any
        runtime error and, if available, the goodness-of-fit metrics via
        ``print_modelfit``.

        Returns
        -------
        PredictiveModel
            The same instance (``self``) to allow method chaining.

        Examples
        --------
        >>> import pandas as pd
        >>> from huff.predictive_models import model_wrapper
        >>> Wieland2015_interaction_matrix = pd.read_excel("data/Wieland2015.xlsx")
        >>> y = Wieland2015_interaction_matrix["MA_Anb1"]
        >>> X = Wieland2015_interaction_matrix[
        ...     [
        ...         "VF", 
        ...         "K", 
        ...         "K_KKr",
        ...         "Dist_Min2",        
        ...     ]
        ... ]
        >>> xgb_result = model_wrapper(
        ...     y,
        ...     X,
        ...     model_type = "xgb"
        ... )
        >>> xgb_result.summary()
        >>> ann_result = model_wrapper(
        ...     y,
        ...     X,
        ...     model_params={
        ...         "activation": "tanh",
        ...         "hidden_layer_sizes": (5,10),
        ...         "solver": "adam"
        ...     },
        ...     model_type = "mlp"
        ... )
        >>> ann_result.summary()
        """

        analysis_description = self.analysis_description
        
        params = self.params
        model_type = params.get("model_type")
        
        fit_metrics = self.fit_metrics
        
        runtime_error = self.runtime_error
        
        print("=" * config.SUMMARY_SECTION_SEP_LINELENGTH)
        print(f"{analysis_description}")
        print("-" * config.SUMMARY_SECTION_SEP_LINELENGTH)
        
        for key, value in params.items():
            
            if key == "model_type":
                print(f"Model type: {config.MODEL_WRAPPER_AVAILABLE[model_type]}")
                print("-" * config.SUMMARY_SECTION_SEP_LINELENGTH)
                                 
            elif key == "model_params":
                print("Model parameters:")
                if len(value) == 0:
                    print("  None")
                else:
                    for mp_key, mp_value in value.items():
                        print(f"  {mp_key}: {mp_value}")
                        
            elif key == "split_params":
                print("Training/test data split parameters:")
                if len(value) == 0:
                    print("  None")
                else:
                    for sp_key, sp_value in value.items():
                        print(f"  {sp_key}: {sp_value}")
                        
            else:
                print(f"{key}: {value}")
                
        print("-" * config.SUMMARY_SECTION_SEP_LINELENGTH)
        
        if runtime_error is not None:
            print(runtime_error)
         
        if fit_metrics is not None:
            
            print (f"Goodness-of-fit for predictive model ({model_type}):")
            
            print_modelfit(fit_metrics)

        print("=" * config.SUMMARY_SECTION_SEP_LINELENGTH)
        
        return self        

def model_wrapper(
    y,
    X,
    model_type: str,
    model_params: dict = None,
    split_params: dict = None,
    X_train: list = None,
    X_test: list = None,
    y_train: list = None,
    y_test: list = None,
    random_state: int = 71,
    verbose: bool = False
    ) -> PredictiveModel:
    
    """
    Generic wrapper for training, prediction and evaluation for 
    machine learning regression models.

    Parameters
    ----------
    y : array-like
        Target variable for the full dataset.
    X : array-like or pandas.DataFrame
        Predictor variables for the full dataset.
    model_type : str
        Key of the model to use. Must be one of
        ``config.MODEL_WRAPPER_AVAILABLE_LIST`` (e.g. ``'ols'``, ``'rf'``).
    model_params : dict, optional
        Parameters forwarded to the model constructor. Default is ``None``.
    split_params : dict, optional
        Parameters forwarded to ``train_test_split``. Default is
        ``config.MODEL_WRAPPER_SPLIT_PARAMS_DEFAULT``.
    X_train : array-like, optional
        User-provided training predictors. If provided together with
        ``X_test``, ``y_train`` and ``y_test``, the automatic split is skipped.
    X_test : array-like, optional
        User-provided test predictors.
    y_train : array-like, optional
        User-provided training targets.
    y_test : array-like, optional
        User-provided test targets.
    random_state : int, optional
        Random seed for reproducibility. Default is ``71``.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    PredictiveModel
        Container with predictions, trained model, fit metrics and metadata.

    Raises
    ------
    TypeError
        If ``model_type`` is not a string.
    ValueError
        If ``model_type`` is unknown or provided train/test arrays have
        mismatched lengths.

    Examples
    --------
    >>> import pandas as pd
    >>> from huff.predictive_models import model_wrapper
    >>> Wieland2015_interaction_matrix = pd.read_excel("data/Wieland2015.xlsx")
    >>> y = Wieland2015_interaction_matrix["MA_Anb1"]
    >>> X = Wieland2015_interaction_matrix[
    ...     [
    ...         "VF", 
    ...         "K", 
    ...         "K_KKr",
    ...         "Dist_Min2",        
    ...     ]
    ... ]
    >>> xgb_result = model_wrapper(
    ...     y,
    ...     X,
    ...     model_type = "xgb"
    ... )
    >>> xgb_result.summary()
    >>> ann_result = model_wrapper(
    ...     y,
    ...     X,
    ...     model_params={
    ...         "activation": "tanh",
    ...         "hidden_layer_sizes": (5,10),
    ...         "solver": "adam"
    ...     },
    ...     model_type = "mlp"
    ... )
    >>> ann_result.summary()
    """

    if not isinstance(model_type, str):
        raise TypeError(f"Param 'model_type' must be a string with one of: {', '.join(config.MODEL_WRAPPER_AVAILABLE_LIST)}.")
    if model_type not in config.MODEL_WRAPPER_AVAILABLE_LIST:
        raise ValueError(f"Unknown model_type: {model_type}. Choose one of: {', '.join(config.MODEL_WRAPPER_AVAILABLE_LIST)}.")

    model_params = model_params or {}
    split_params = split_params or config.MODEL_WRAPPER_SPLIT_PARAMS_DEFAULT

    MODEL_REGISTRY = {

        config.MODEL_WRAPPER_AVAILABLE_LIST[0]: lambda: LinearRegression(**model_params),

        config.MODEL_WRAPPER_AVAILABLE_LIST[1]: lambda: BaggingRegressor(
            estimator=LinearRegression(),
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[2]: lambda: BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[3]: lambda: RandomForestRegressor(
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[4]: lambda: GradientBoostingRegressor(
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[5]: lambda: KNeighborsRegressor(**model_params),

        config.MODEL_WRAPPER_AVAILABLE_LIST[6]: lambda: SVR(**model_params),

        config.MODEL_WRAPPER_AVAILABLE_LIST[7]: lambda: XGBRegressor(
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[8]: lambda: LGBMRegressor(
            random_state=random_state,
            **model_params
        ),

        config.MODEL_WRAPPER_AVAILABLE_LIST[9]: lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                random_state=random_state,
                **model_params
            ))
        ]),
    }

    user_split = all(v is not None for v in [X_train, X_test, y_train, y_test])

    if verbose:
        if user_split:
            print("NOTE: Using user-provided train/test split.")
        else:
            print("NOTE: No user-provided train/test split. Performing automatic split.")

    if user_split:

        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError(f"Train/test X and y must have matching lengths, not X_train={len(X_train)}, y_train={len(y_train)}, X_test={len(X_test)}, y_test={len(y_test)}.")
        
    else:

        if verbose:
            print("Performing train/test split", end=" ... ")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            random_state=random_state,
            **split_params
        )

        if verbose:
            print("OK")

    model = MODEL_REGISTRY[model_type]()

    runtime_error = None

    try:

        if verbose:
            print(f"Training {config.MODEL_WRAPPER_AVAILABLE[model_type]} model", end=" ... ")
            
        model.fit(X_train, y_train)
        
        if verbose:
            print("OK")
            
    except Exception as e:
        print(f"WARNING: Model training failed: '{e}'")
        runtime_error = f"Model training failed: '{e}'"

    y_pred = None

    if runtime_error is None:
        
        try:
            
            if verbose:
                print(f"Predicting with {config.MODEL_WRAPPER_AVAILABLE[model_type]} model", end=" ... ")

            y_pred = model.predict(X_test)

            if verbose:
                print("OK")
                
        except Exception as e:
            print(f"WARNING: Model prediction failed: '{e}'")
            runtime_error = f"Model prediction failed: '{e}'"

    fit_metrics = None

    if runtime_error is None:
        
        try:
            
            fit_metrics = modelfit(
                observed=y_test,
                expected=y_pred,
                remove_nan=True,
                verbose=verbose
            )            
            
        except Exception as e:
            print(f"WARNING: Calculation of fit metrics failed: '{e}'")
            runtime_error = f"Calculation of fit metrics failed: '{e}'"

    predictive_model = PredictiveModel(
        y_pred=y_pred,
        model=model,
        fit_metrics=fit_metrics,
        params={
            "model_type": model_type,
            "model_params": model_params,
            "split_params": split_params,
            "random_state": random_state,
            "user_split": user_split
        },
        data={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        runtime_error=runtime_error,
        analysis_description = config.PREDICTIVE_MODEL_DESCRIPTION,
        timestamp = create_timestamp(
            function="model_wrapper",
            process = f"Creation of {config.MODEL_WRAPPER_AVAILABLE[model_type]} model",
            status="OK" if runtime_error is None else runtime_error
            )
    )

    return predictive_model