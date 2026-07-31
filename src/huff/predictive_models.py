#-----------------------------------------------------------------------
# Name:        predictive_models (huff package)
# Purpose:     Creating ML predictive models for Market Area Analyses
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.1
# Last update: 2026-07-31 13:26
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
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
from huff.helper import create_timestamp, add_timestamp, print_modelfit


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
    
    def predict(
        self,
        df: pd.DataFrame = None,
        X_cols: list = None,        
        ):
        
        """
        Predict target values for new or stored test data.

        Parameters
        ----------
        df : pandas.DataFrame or None
            Predictor data for prediction. If ``None``, stored test data is used.
        X_cols : list or None
            List of predictor columns to use from ``df``. Required if ``df`` is provided.

        Returns
        -------
        numpy.ndarray or pandas.Series or None
            Predicted target values or ``None`` if prediction failed.

        Raises
        ------
        KeyError
            If user-specified ``df`` is missing required predictor columns.

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
        >>> result = model_wrapper(y, X, model_type="xgb")
        >>> result.predict(df=X, X_cols=["VF", "K", "K_KKr", "Dist_Min2"])
        """

        if X_cols is None:
            X_cols = []
        
        if df is not None and len(X_cols) > 0:
            
            X_train = self.data["X_train"]
            
            X_cols_missing = []
            
            for X_col in X_train.columns:
                
                if X_col not in X_cols:
                    X_cols_missing.append(X_col)
                    
            if len(X_cols_missing) > 0:
                raise KeyError(f"No prediction possible because of missing columns in user-specified X data: {', '.join(X_cols_missing)}.")
        
            X_pred = df[X_cols]
            
        else:
            
            print("NOTE: No X data specified by user. Using test data from model for prediction.")
        
            X_pred = self.data["X_test"]
            
        model = self.model
        model_runtime_error = self.runtime_error
        
        y_pred = None
        
        if model_runtime_error is not None:            
            print(model_runtime_error)                                            
        else:            
            try:
                y_pred = model.predict(X_pred)
            except Exception as e:
                print(f"WARNING: Model prediction failed: '{e}'")
            
        return y_pred
        
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

class PredictiveModels:
    
    def __init__(
        self,
        predictive_models,
        y_test_models,
        models_fit_metrics_df,
        model_wrapper_errors,
        best_model,
        timestamp
        ):
        
        """
        Initialize PredictiveModels container.

        Parameters
        ----------
        predictive_models : list
            List of ``PredictiveModel`` instances or ``None`` placeholders.
        y_test_models : pandas.DataFrame
            DataFrame with observed test values and predictions for each model.
        models_fit_metrics_df : pandas.DataFrame
            DataFrame with goodness-of-fit metrics for each model.
        model_wrapper_errors : list
            List of error messages from failed model runs.
        timestamp : str
            Timestamp or metadata for the wrapper creation.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> from huff.predictive_models import models_wrapper
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
        >>> models = {"xgb": {"model_params": {}}, "rf": {"model_params": {}}}
        >>> result = models_wrapper(y, X, models=models)
        >>> result.models_fit_metrics_df
        """
                
        self.predictive_models = predictive_models
        self.y_test_models = y_test_models
        self.models_fit_metrics_df = models_fit_metrics_df
        self.model_wrapper_errors = model_wrapper_errors
        self.best_model = best_model
        self.timestamp = timestamp
        
    def find_best_model(
        self,
        fit_metrics: list = None,
        verbose: bool = False
        ):
        
        if fit_metrics is None:

            fit_metrics = []

        else:            

            fit_metrics_invalid = []
            for fit_metric in fit_metrics:
                if fit_metric not in config.GOODNESS_OF_FIT_VALUES:
                    fit_metrics_invalid.append(fit_metric)

            if len(fit_metrics_invalid) > 0:
                fit_metrics = [x for x in fit_metrics if x not in fit_metrics_invalid]
                print(f"Specified list 'fit_metrics' contains invalid values which are skipped: {', '.join(fit_metrics_invalid)}.")
        
        if len(fit_metrics) == 0:
            
            fit_metrics = config.GOODNESS_OF_FIT_BESTMODEL_DEFAULT
            
            print(f"NOTE: No (valid) 'fit_metrics' list specified. Using default: {' > '.join(config.GOODNESS_OF_FIT_BESTMODEL_DEFAULT)}.")
        
        if verbose:
            print(f"Identifying best model with respect to {len(fit_metrics)} fit metrics: {' > '.join(fit_metrics)}", end = " ... ")
        
        models_fit_metrics_df = self.models_fit_metrics_df.copy()
        
        best_model = None
        
        for fit_metric in fit_metrics:
            
            if fit_metric in models_fit_metrics_df.index:                
            
                if config.GOODNESS_OF_FIT_OPTIMIZATION[fit_metric] == "min":
                    
                    best_model_extract = models_fit_metrics_df.loc[fit_metric]
                    val_min = best_model_extract.min()
                    best_model = best_model_extract[best_model_extract == val_min].index.tolist()
                    
                    if len(best_model) == 1:
                        break
                                        
                elif config.GOODNESS_OF_FIT_OPTIMIZATION[fit_metric] == "max":
                    
                    best_model_extract = models_fit_metrics_df.loc[fit_metric]
                    val_max = best_model_extract.max()
                    best_model = best_model_extract[best_model_extract == val_max].index.tolist()
                    
                    if len(best_model) == 1:
                        break
                    
        best_model_result = best_model[0]

        best_model_result_name = config.MODEL_WRAPPER_AVAILABLE[best_model_result]
        
        if verbose:
            print("OK")

        if len(best_model) > 1:
            print(f"NOTE: In view of the selection criteria, several models have identical values: {', '.join(best_model)}.")
        else:
            if verbose:
                print(f"The best model is: {best_model_result_name}.")

        self.best_model = best_model_result

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
    fit_metrics = None

    if runtime_error is None and len(X_test) > 0:
        
        try:
            
            if verbose:
                print(f"Predicting with {config.MODEL_WRAPPER_AVAILABLE[model_type]} model", end=" ... ")

            y_pred = model.predict(X_test)

            if verbose:
                print("OK")
                
        except Exception as e:
            print(f"WARNING: Model prediction failed: '{e}'")
            runtime_error = f"Model prediction failed: '{e}'"

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

def models_wrapper(
    y,
    X,
    models: dict = None,    
    X_train: list = None,
    X_test: list = None,
    y_train: list = None,
    y_test: list = None,
    random_state: int = 71,
    verbose: bool = False
    ) -> PredictiveModels:
    
    """
    Run multiple predictive models and compile results.

    Parameters
    ----------
    y : array-like
        Target variable for the full dataset.
    X : array-like or pandas.DataFrame
        Predictor variables for the full dataset.
    models : dict, optional
        Mapping of model keys to parameter dictionaries for ``model_wrapper``.
    X_train : array-like, optional
        User-provided training predictors.
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
    PredictiveModels
        Container with multiple model results, test predictions and fit metrics.

    Examples
    --------
    >>> import pandas as pd
    >>> from huff.predictive_models import models_wrapper
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
    >>> models = {"xgb": {"model_params": {}}, "rf": {"model_params": {}}}
    >>> result = models_wrapper(y, X, models=models)
    >>> result.models_fit_metrics_df
    """
    if models is None:
        models = {}
        
    if len(models) == 0:
        print("WARNING: No models were specified. Returning None.")
        return None
        
    predictive_models = []
    model_wrapper_errors = []
    
    for key, value in models.items():
        
        try:
            
            predictive_model = model_wrapper(
                y = y,
                X = X,
                model_type = key,
                model_params = value.get("model_params"),
                split_params = value.get("split_params"),
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                random_state = random_state,
                verbose = verbose
                )
            
            predictive_models.append(predictive_model)
            
        except Exception as e:
            predictive_models.append(None)
            model_wrapper_errors.append(str(e))
            
    if verbose:
        print(f"Compiling observed and predicted test data for {len(predictive_models)} models", end = " ... ")
    
    y_test_models = pd.DataFrame(predictive_models[0].data["y_test"])
    y_test_models.reset_index(drop=True, inplace=True)
    y_col = y_test_models.columns[0]
    
    for model in predictive_models:
        
        runtime_error = model.runtime_error
        
        if runtime_error is None:
                        
            model_type = model.params["model_type"]
                        
            y_pred = pd.DataFrame(model.y_pred, columns=[f"{y_col}{config.DEFAULT_PREDICTED_SUFFIX}_{model_type}"])
            y_pred.reset_index(drop=True, inplace=True)
            
            y_test_models = pd.concat(
                [
                    y_test_models,
                    y_pred
                    ],
                axis = 1
                )
 
    if verbose:
        print("OK")
        print(f"Compiling goodness-of-fit metrics for {len(predictive_models)} models", end = " ... ")
    
    models_fit_metrics_df = pd.DataFrame(index=config.GOODNESS_OF_FIT_VALUES)
    
    for model in predictive_models:
        
        if runtime_error is None:
                                
            model_type = model.params["model_type"]
        
            model_fit_metrics = model.fit_metrics[1]
            model_fit_metrics = {
                mfm_key: mfm_value
                for mfm_key, mfm_value in model_fit_metrics.items()
                if mfm_key in config.GOODNESS_OF_FIT_VALUES
            }
            model_fit_metrics_df = pd.DataFrame(model_fit_metrics.values(), columns=[model_type])
            model_fit_metrics_df.index = config.GOODNESS_OF_FIT_VALUES
            
            models_fit_metrics_df = pd.concat(
                [
                    models_fit_metrics_df,
                    model_fit_metrics_df
                ],
                axis = 1
            )

    if verbose:
        print("OK")
        
    predictive_models = PredictiveModels(
        predictive_models = predictive_models,
        y_test_models = y_test_models,
        models_fit_metrics_df = models_fit_metrics_df,
        model_wrapper_errors = model_wrapper_errors,
        best_model = None,
        timestamp = create_timestamp(
            function="models_wrapper",
            process = f"Creation of {len(predictive_models)} predictive models",
            status="OK" if len(model_wrapper_errors) == 0 else ' '.join(model_wrapper_errors)
            )
        )
    
    return predictive_models