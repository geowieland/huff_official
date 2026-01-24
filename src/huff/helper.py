#-----------------------------------------------------------------------
# Name:        helper (huff package)
# Purpose:     Huff Model helper functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.10
# Last update: 2026-01-20 07:05
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------

import pandas as pd


def check_vars(
    df: pd.DataFrame,
    cols: list,
    check_numeric: bool = True,
    check_zero: bool = True,
    check_constant: bool = True
    ):

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
    
    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    if not pd.api.types.is_numeric_dtype(values):
        return False
    else:
        return True
    
def check_constant_values(values):

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    if values.nunique() == 1:
        return True
    else:
        return False