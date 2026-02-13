#-----------------------------------------------------------------------
# Name:        helper (huff package)
# Purpose:     Huff Model helper functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.19
# Last update: 2026-02-12 19:22
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------

from datetime import datetime
import pandas as pd
import huff.config as config


def check_vars(
    df: pd.DataFrame,
    cols: list,
    check_numeric: bool = True,
    check_zero: bool = True,
    check_constant: bool = True
    ):

    """
    Validate selected columns of a DataFrame for suitability in model fitting and analysis.

    This function checks whether specified columns exist in the DataFrame and optionally
    verifies that they are numeric, strictly positive (non-zero), and non-constant.
    If any check fails, a descriptive Exception is raised summarizing all detected issues.
    The function is implemented in several functions and methods in the `models` module.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the variables to be validated.
    cols : list of str
        List of column names to be checked.
    check_numeric : bool, default=True
        If True, checks whether all specified columns contain numeric values (relevant for all models).
    check_zero : bool, default=True
        If True, checks whether columns contain values less than or equal to zero (relevant esp. for MCI model).
    check_constant : bool, default=True
        If True, checks whether columns contain constant values only.

    Raises
    ------
    Exception
        If one or more (desired) validation checks fail (missing columns, non-numeric values,
        zero or negative values, or constant columns).

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> check_vars(df, ["a", "b"])

    >>> df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 3, 4]})
    >>> check_vars(df, ["a", "b"], check_constant=True)
    Traceback (most recent call last):
        ...
    Exception: The following error(s) occurred with respect to the input dataframe: Column(s) a are constant.
    """

    errors = []

    cols_missing = []

    for col in cols:

        if col not in df.columns:
            cols_missing.append(col)

    if len(cols_missing) > 0:    
        errors.append(f"Column(s) {', '.join(cols_missing)} not in dataframe.")
    
    cols = [col for col in cols if col not in cols_missing]

    if check_numeric:

        cols_not_numeric = []

        for col in cols:

            if not check_numeric_series(df[col]):
                cols_not_numeric.append(col)

        if len(cols_not_numeric) > 0:
            errors.append(f"Non-numeric column(s): {', '.join(cols_not_numeric)}. All stated columns must be numeric.")

    if check_zero:

        cols_with_zeros = []

        for col in cols:

            if (df[col] <= 0).any():

                cols_with_zeros.append(col)
        
        if len(cols_with_zeros) > 0:
            errors.append(f"Column(s) {', '.join(cols_with_zeros)} include(s) values <= 0. All values must be numeric and positive.")

    if check_constant:

        cols_constant = []

        for col in cols:

            if check_constant_values(df[col]):
                cols_constant.append(col)
        
        if len(cols_constant) > 0:
            errors.append(f"Column(s) {', '.join(cols_constant)} are constant. All values must be at least dummy binary.")
    
    if len(errors) > 0:

        raise Exception(f"The following error(s) occured with respect to the input dataframe: {' '.join(errors)}")

def check_numeric_series(values):
    
    """
    Check whether a sequence contains numeric values only.
    This function is used within `check_vars()`.

    Parameters
    ----------
    values : array-like or pandas.Series
        Input values to be checked.

    Returns
    -------
    bool
        True if values are numeric, False otherwise.

    Examples
    --------
    >>> values = pd.Series([1, 2, 3, 4, 5, 6])
    >>> check_numeric_series(values)
    """

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    if not pd.api.types.is_numeric_dtype(values):
        return False
    else:
        return True
    
def check_constant_values(values):

    """
    Check whether a sequence contains constant values only.
    This function is used within `check_vars()`.

    Parameters
    ----------
    values : array-like or pandas.Series
        Input values to be checked.

    Returns
    -------
    bool
        True if all values are identical, False otherwise.
    
    Examples
    --------
    >>> values = pd.Series([1, 1, 1, 1])
    >>> check_numeric_series(values)
    """

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    if values.nunique() == 1:
        return True
    else:
        return False
    
def create_timestamp(
    function: str,
    process: str = "Update",
    status: str = "OK"
    ):

    """
    Create a timestamp dictionary for logging function processes.
    This function is called by `add_timestamp()`.
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    function : str
        Name of the function creating the timestamp.
    process : str, optional
        Description of the process (default: "Update").
    status : str, optional
        Process status label (default: "OK").

    Returns
    -------
    dict
        Dictionary containing package version, function name, process,
        timestamp, and status.
    
    Examples
    --------
    >>> timestamp_dict = create_timestamp(
    ...     function="any_function()",
    ...     process="Doing something",
    ...     status="OK"
    ... )
    """

    now = datetime.now()

    timestamp_dict = {
        "package_version": f"{config.PACKAGE_NAME} {config.PACKAGE_VERSION}",
        "function": function,
        "process": process,
        "datetime": now.strftime("%Y-%m-%d %H-%M-%S"),
        "status": status
    }

    return timestamp_dict

def add_timestamp(
    obj,
    function: str,
    process: str = "Update",
    status: str = "OK",
    verbose: bool = False
    ):

    """
    Add a timestamp entry to an object's metadata.
    This function is built into the `models` module and is called every time an object 
    of the library's internal classes (e.g., InteractionMatrix) is modified.
    Using this function only makes sense internally within the modules.
    
    Parameters
    ----------
    obj : object
        Object to which the timestamp metadata is added.
    function : str
        Name of the function creating the timestamp.
    process : str, optional
        Description of the process (default: "Update").
    status : str, optional
        Process status label (default: "OK").
    verbose : bool, optional
        If True, print informational messages (default: False).

    Returns
    -------
    object
        The input object with updated timestamp metadata.

    Examples
    --------
    >>> add_timestamp(
    ...     self.interaction_matrix,
    ...     function="models.MCIModel.probabilities",
    ...     process="Calculated probabilities"
    ... )
    """

    obj_class = obj.__class__.__name__

    if getattr(obj, "metadata", None) is None:

        if verbose:
            print(f"The class {obj_class} object does not include metadata yet.")

        obj.metadata = {}

    if "timestamp" not in obj.metadata:

        obj.metadata["timestamp"] = {}

    timestamp_dict = create_timestamp(
        function=function,
        process=process,
        status=status
        )

    next_timestamp_index = max(obj.metadata["timestamp"].keys(), default=-1) + 1
    
    obj.metadata["timestamp"][next_timestamp_index] = timestamp_dict
    
    return obj

def print_timestamp(
    obj
    ):

    """
    Print timestamp metadata of an object of the library's internal 
    classes (e.g., InteractionMatrix).
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    obj : object
        Object with timestamp metadata.

    Returns
    -------
    dict or None
        Timestamp metadata if present, otherwise None.

    Examples
    --------
    >>> obj_timestamp = print_timestamp(obj)
    """

    obj_class = obj.__class__.__name__

    if getattr(obj, "metadata", None) is None:

        print(f"Object of class {obj_class} has no metadata")

        return None

    else:

        metadata = getattr(obj, "metadata", None)

        if "timestamp" not in metadata:
            
            print(f"Object of class {obj_class} has no timestamps in the metadata")

            return None

        else:

            print(f"Timestamps for class {obj_class} object")
            
            for key, value in metadata["timestamp"].items():
                
                datetime = value['datetime']
                package_version = value['package_version']
                function = value['function']
                process = value['process']
                status = value['status']
                
                col_green = "\033[92m"
                col_red = "\033[91m"
                col_reset = "\033[0m"
                
                check_sign = "✔"
                check_sign_color = col_green
                
                error_message = ""

                if status != "OK":
                    check_sign = "✖"
                    check_sign_color = col_red
                    error_message = f"| {status}"
                
                print(f"[{datetime}] {check_sign_color}{check_sign}{col_reset} {package_version} | Step {key} | {function} | {process} {error_message}")
                
            return metadata["timestamp"]
        
def print_summary_row(
    output_name, 
    output_value, 
    width=config.SUMMARY_WIDTH
    ):
    
    """
    Print a formatted name-value pair for model summaries.
    This function is a helper function for the summary() methods in the `models` module.
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    output_name : str
        Label of the output value.
    output_value : any
        Value to print.
    width : int, optional
        Field width for alignment (default is 26).

    Examples
    --------
    >>> print_summary_row(
    ...     "No. locations",
    ...     4
    ... )
    """

    value = output_value if output_value is not None else config.SUMMARY_NOT_DEFINED
    print(f"{output_name:<{width}} {value}")

def print_weightings(interaction_matrix):
    
    """
    Retrieves information from an InteractionMatrix object's metadata
    and prints the weightings. This function is a helper function 
    for the summary() methods in the `models` module.
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    interaction_matrix : InteractionMatrix
        Instance of class InteractionMatrix.
    
    Examples
    --------
    >>> print_weightings(self)
    """
    
    customer_origins_metadata = interaction_matrix.get_customer_origins().get_metadata()
    supply_locations_metadata = interaction_matrix.get_supply_locations().get_metadata()
    interaction_matrix_metadata = interaction_matrix.get_metadata()
    
    if (supply_locations_metadata["weighting"][0]["name"] is not None and supply_locations_metadata["weighting"][0]["func"] is not None and supply_locations_metadata["weighting"][0]["param"] is not None) or (customer_origins_metadata['weighting'][0]['param'] is not None):
    
        print("--------------------------------------")

        print("Weightings")

        for key, value in supply_locations_metadata["weighting"].items():

            if value['name'] is not None and value['func'] is not None and value['param'] is not None:

                if supply_locations_metadata["attraction_col"][key] is not None:
                    name = supply_locations_metadata["attraction_col"][key]
                else:
                    if value['name'] is not None:
                        name = value['name']
                    else:
                        name = f"Attraction variable {key}"
                
                if isinstance(value['param'], list):
                    param = ', '.join(str(round(x, config.FLOAT_ROUND)) for x in value['param'])
                elif isinstance(value['param'], (int, float)):
                    param = round(value['param'], config.FLOAT_ROUND)
                else:
                    param = "Invalid format"

                func_description = config.PERMITTED_WEIGHTING_FUNCTIONS[value['func']]['description']

                print_summary_row(
                    name,
                    f"{param} ({func_description})"
                )

            else:
                if key == 0:
                    break

        if customer_origins_metadata['weighting'][0]['param'] is not None:

            tc_weighting = None

            transport_costs_col = "Transport costs"
            if "transport_costs_col" in interaction_matrix_metadata and interaction_matrix_metadata["transport_costs_col"] != config.DEFAULT_COLNAME_TC:
                transport_costs_col = interaction_matrix_metadata["transport_costs_col"]

            if isinstance(customer_origins_metadata['weighting'][0]['param'], list):
                tc_params = ', '.join(str(round(x, config.FLOAT_ROUND)) for x in customer_origins_metadata['weighting'][0]['param'])
            elif isinstance(customer_origins_metadata['weighting'][0]['param'], (int, float)):
                tc_params = round(customer_origins_metadata['weighting'][0]['param'], config.FLOAT_ROUND)
            else:
                tc_params = "Invalid format"

            tc_function = config.PERMITTED_WEIGHTING_FUNCTIONS[customer_origins_metadata["weighting"][0]["func"]]["description"]
            tc_weighting = f"{tc_params} ({tc_function})"

            print_summary_row(
                transport_costs_col,
                tc_weighting
            )

    return [
        customer_origins_metadata,
        supply_locations_metadata,
        interaction_matrix_metadata
    ]

def print_interaction_matrix_info(interaction_matrix):

    """
    Retrieves information from an InteractionMatrix object's metadata
    and prints the related information. This function is a helper function 
    for the summary() methods in the `models` module.
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    interaction_matrix : InteractionMatrix
        Instance of class InteractionMatrix.
    
    Examples
    --------
    >>> print_interaction_matrix_info(self)
    """

    customer_origins_metadata = interaction_matrix.get_customer_origins().get_metadata()
    supply_locations_metadata = interaction_matrix.get_supply_locations().get_metadata()
    interaction_matrix_metadata = interaction_matrix.get_metadata()

    print("--------------------------------------")

    interactions = supply_locations_metadata["no_points"] * customer_origins_metadata["no_points"]

    print_summary_row(
        "Interactions",
        interactions
        )

    transport_costs_metrics = config.DEFAULT_NAME_TC

    if interaction_matrix_metadata != {} and "transport_costs" in interaction_matrix_metadata:

        transport_costs_metrics = interaction_matrix_metadata["transport_costs"]["metrics"]            
        
        print_summary_row(
            f"{config.DEFAULT_NAME_TC} type",
            transport_costs_metrics
        )

        if transport_costs_metrics == "distance":
            print_summary_row(
                f"{config.DEFAULT_NAME_TC} unit",
                interaction_matrix_metadata["transport_costs"]["distance_unit"]
            )
        else:
            print_summary_row(
                f"{config.DEFAULT_NAME_TC} unit",
                interaction_matrix_metadata["transport_costs"]["time_unit"]
            )

    return [
        customer_origins_metadata,
        supply_locations_metadata,
        interaction_matrix_metadata
    ]

def print_modelfit(modelfit_results):
    
    """
    Print goodness-of-fit statistics of an output from the function `modelfit()`.
    Using this function only makes sense internally within the modules.

    Parameters
    ----------
    modelfit_results : tuple
        Output from the function `modelfit()` containing goodness-of-fit values.

    Returns
    -------
    tuple
        The unchanged model fitting results.
    """

    maxlen = max(len(str(key)) for key in config.GOODNESS_OF_FIT.keys())

    for gof_key, gof_value in config.GOODNESS_OF_FIT.items():
                    
        if gof_key in config.GOODNESS_OF_FIT.keys():                        
        
            if modelfit_results[1][gof_value] is not None:
                print(f"{gof_key:<{maxlen}}  {round(modelfit_results[1][gof_value], 2)}")

    return modelfit_results