#-----------------------------------------------------------------------
# Name:        helper (huff package)
# Purpose:     Huff Model helper functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.16
# Last update: 2026-02-05 15:29
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
    >>> import pandas as pd
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

        raise Exception(f"The following error(s) occured with respect to the input dataframe: {' '.join(errors)}.")

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

    Parameters
    ----------
    obj : object
        Object with timestamp metadata.

    Returns
    -------
    dict or None
        Timestamp metadata if present, otherwise None.
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
    width=26
    ):
    
    """
    Print a formatted name–value pair for model summaries.
    This function is a helper function for the summary() methods in the `models` module.

    Parameters
    ----------
    output_name : str
        Label of the output value.
    output_value : any
        Value to print.
    width : int, optional
        Field width for alignment (default is 26).
    """

    value = output_value if output_value is not None else "not defined"
    print(f"{output_name:<{width}} {value}")