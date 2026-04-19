#-----------------------------------------------------------------------
# Name:        data_management (huff package)
# Purpose:     Loading of external data for Huff/MCI model analyses
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.16
# Last update: 2026-04-19 13:12
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import geopandas as gp
from shapely import wkt
from shapely.geometry import Point
from huff.models import SupplyLocations, CustomerOrigins, InteractionMatrix, MarketAreas, InteractionMatrixError, market_shares
import huff.helper as helper
import huff.config as config


def load_geodata(
    data, 
    location_type: str, 
    unique_id: str,
    x_col: str | None = None, 
    y_col: str | None = None,
    data_type: str = "shp", 
    csv_sep: str = ";", 
    csv_decimal: str = ",", 
    csv_encoding: str = "unicode_escape", 
    crs_input: str = "EPSG:4326",
    verbose: bool = False
    ) -> CustomerOrigins | SupplyLocations:
    
    """
    Load spatial data and return it as a CustomerOrigins or SupplyLocations object.

    The function supports CSV, Excel, shapefiles, pandas DataFrames, and
    GeoDataFrames. For tabular data without geometry information, x and y
    coordinate columns must be provided.

    Parameters
    ----------
    data : str or pandas.DataFrame or geopandas.GeoDataFrame
        Input data containing spatial information. Can be a file path or
        an in-memory data object.
    location_type : {"origins", "destinations"}
        Specifies whether customer origins or supply locations are loaded.
    unique_id : str
        Name of the column containing a unique identifier.
    x_col : str, optional
        Name of the column containing x coordinates (required for tabular data).
    y_col : str, optional
        Name of the column containing y coordinates (required for tabular data).
    data_type : {"csv", "xlsx", "shp"}, default="shp"
        File type if data is loaded from disk.
    csv_sep : str, default=";"
        Column separator used in CSV files.
    csv_decimal : str, default=","
        Decimal mark used in CSV files.
    csv_encoding : str, default="unicode_escape"
        Encoding used in CSV files.
    crs_input : str, default="EPSG:4326"
        Coordinate reference system of the input data.

    Returns
    -------
    CustomerOrigins or SupplyLocations
        A CustomerOrigins object if `location_type` is ``"origins"``,
        otherwise a SupplyLocations object.

    Notes
    -----
    If `data` does not contain geometry information, both `x_col` and
    `y_col` must be provided.

    Examples
    --------
    >>> Haslach = load_geodata(
    ...     "data/Haslach.shp", 
    ...     location_type="origins", 
    ...     unique_id="BEZEICHN"
    ... )
    >>> Haslach_supermarkets = load_geodata(
    ...     "data/Haslach_supermarkets.shp", 
    ...     location_type="destinations", 
    ...     unique_id="LFDNR"
    ... )
    >>> Haslach_interaction_matrix = create_interaction_matrix(
    ...     Haslach, 
    ...     Haslach_supermarkets
    ... )
    """
    
    crs_output = config.WGS84_CRS

    if location_type is None or (location_type not in config.PERMITTED_LOCATION_TYPES):
        raise ValueError (f"Error while loading geodata: Argument location_type must be one of the following: {', '.join(config.PERMITTED_LOCATION_TYPES)}")

    if isinstance(data, gp.GeoDataFrame):

        geodata_gpd_original = data

        if not all(geodata_gpd_original.geometry.geom_type == "Point"):
            raise TypeError("Error while loading geodata: Input geopandas.GeoDataFrame must be of type 'Point'")
        
        crs_input = geodata_gpd_original.crs

        geodata_gpd = geodata_gpd_original.to_crs(crs_output)
        geodata_gpd = geodata_gpd[[unique_id, "geometry"]] 

    elif isinstance(data, pd.DataFrame):

        geodata_tab = data

    elif isinstance(data, str):

        if data_type == "shp":

            geodata_gpd_original = gp.read_file(data)

            if not all(geodata_gpd_original.geometry.geom_type == "Point"):
                raise TypeError ("Error while loading geodata: Input shapefile must be of type 'Point'")
            
            crs_input = geodata_gpd_original.crs

            geodata_gpd = geodata_gpd_original.to_crs(crs_output)
            geodata_gpd = geodata_gpd[[unique_id, "geometry"]] 

        elif data_type == "csv" or data_type == "xlsx":

            coord_cols_missing = []
            if x_col is None:
                coord_cols_missing.append("X coordinate column")
            if y_col is None:
                coord_cols_missing.append("Y coordinate column")
            if len(coord_cols_missing) > 0:
                raise ValueError(f"Missing value for {', '.join(coord_cols_missing)}.")
            
            if data_type == "csv":

                geodata_tab = pd.read_csv(
                    data, 
                    sep = csv_sep, 
                    decimal = csv_decimal, 
                    encoding = csv_encoding
                    ) 
                
            if data_type == "xlsx":

                geodata_tab = pd.read_excel(data)

        else:
            raise TypeError("Error while loading geodata: Unknown type of data")
        
    else:
        raise TypeError("Error while loading geodata: Param 'data' must be pandas.DataFrame, geopandas.GeoDataFrame or file (.csv, .xlsx, .shp)")

    if verbose:
        print(f"Checking and importing {location_type} data", end = " ... ")

    if data_type == "csv" or data_type == "xlsx" or (isinstance(data, pd.DataFrame) and not isinstance(data, gp.GeoDataFrame)):
        
        helper.check_vars(
            df = geodata_tab,
            cols = [x_col, y_col],
            check_constant = False
            )

        try:
        
            geodata_gpd_original = gp.GeoDataFrame(
                geodata_tab, 
                geometry = gp.points_from_xy(
                    geodata_tab[x_col], 
                    geodata_tab[y_col]
                    ), 
                crs = crs_input
                )

            geodata_gpd = geodata_gpd_original.to_crs(crs_output)
            geodata_gpd = geodata_gpd[[unique_id, "geometry"]]   
            
        except Exception as e:

            print(f"WARNING: Invalid geometry information in {location_type} input data. Object is imported as pandas.DataFrame without geometry. Error: {str(e)}")

            geodata_gpd = geodata_gpd[[unique_id]]    
    
    if verbose:
        print("OK")
        print(f"Constructing instance of class for {location_type}", end = " ... ")

    metadata = {
        "location_type": location_type,
        "unique_id": unique_id,
        "attraction_col": [None],
        "marketsize_col": None,
        "weighting": {
            0: {
                "name": None,
                "func": None, 
                "param": None
                }
            },
        "crs_input": crs_input,
        "crs_output": crs_output,
        "no_points": len(geodata_gpd),
        }
         
    if location_type == "origins":

        geodata_object = CustomerOrigins(
            geodata_gpd, 
            geodata_gpd_original, 
            metadata,
            None,
            None
            )
                
    elif location_type == "destinations":

        geodata_gpd[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] = 0
        geodata_gpd_original[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] = 0    

        geodata_object = SupplyLocations(
            geodata_gpd = geodata_gpd, 
            geodata_gpd_original = geodata_gpd_original, 
            metadata = metadata,
            isochrones_gdf = None,
            buffers_gdf = None,
            competitor_concentration = None
            )
        
    helper.add_timestamp(
        geodata_object,
        function="data_management.load_geodata",
        process = "Creation by import"
        )
    
    if verbose:
        print("OK")

    return geodata_object

def load_interaction_matrix(
    data,
    customer_origins_col: str,
    supply_locations_col: str,
    attraction_col: list[str],
    transport_costs_col: str,
    transport_costs_metrics: str | None = None,
    transport_costs_distance_unit: str | None = None,
    transport_costs_time_unit: str | None = None,
    flows_col: str | None = None,
    probabilities_col: str | None = None,
    market_size_col: str | None = None,
    customer_origins_coords_col: str | list[str] | None = None,
    supply_locations_coords_col: str | list[str] | None = None,
    data_type: str = "csv", 
    csv_sep: str = ";", 
    csv_decimal: str = ",", 
    csv_encoding: str = "unicode_escape",
    xlsx_sheet: str | None = None,
    crs_input: str = "EPSG:4326",
    crs_output: str = "EPSG:4326",
    check_df_vars: bool = True,
    verbose: bool = False
    ) -> InteractionMatrix:
    
    """    
    Load an interaction matrix from tabular or spatial data and return it as an InteractionMatrix object.

    This function imports interaction data describing flows or probabilities
    between customer origins and supply locations. It constructs 
    CustomerOrigins and SupplyLocations objects including optional geospatial
    information, and normalizes column names to internal defaults.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Input data containing interaction information. Can be a file path
        (CSV/XLSX) or a pandas DataFrame.
    customer_origins_col : str
        Column identifying customer origins.
    supply_locations_col : str
        Column identifying supply locations.
    attraction_col : list of str
        Column(s) describing attraction values of supply locations.
    transport_costs_col : str
        Column describing transport costs between origins and destinations.
    transport_costs_metrics : str, optional
        Description of transport costs metrics: {"distance", "time"}.
    transport_costs_distance_unit : str, optional
        Description of distance unit (e.g., "kilometers").
        Should be None if transport_costs_metrics = "time".
    transport_costs_time_unit : str, optional
        Description of time unit (e.g., "minutes").
        Should be None if transport_costs_metrics = "distance".
    flows_col : str, optional
        Column describing observed flows between origins and destinations.
    probabilities_col : str, optional
        Column describing interaction probabilities.
    market_size_col : str, optional
        Column describing market size at customer origins.
    customer_origins_coords_col : str or list of str, optional
        Column(s) containing coordinates of customer origins.
        If a list, interpreted as `[x, y]`.
    supply_locations_coords_col : str or list of str, optional
        Column(s) containing coordinates of supply locations.
        If a list, interpreted as `[x, y]`.
    data_type : {"csv", "xlsx"}, default="csv"
        File type if loading from disk.
    csv_sep : str, default=";"
        Column separator for CSV files.
    csv_decimal : str, default=","
        Decimal mark for CSV files.
    csv_encoding : str, default="unicode_escape"
        Encoding used in CSV files.
    xlsx_sheet : str, optional
        Excel sheet name if reading from an XLSX file.
    crs_input : str, default="EPSG:4326"
        Coordinate reference system of input geospatial data.
    crs_output : str, default="EPSG:4326"
        Target coordinate reference system for geospatial data.
    check_df_vars : bool, default=True
        If True, validates that required columns exist using `helper.check_vars`.

    Returns
    -------
    InteractionMatrix
        An InteractionMatrix object containing the loaded data, CustomerOrigins,
        SupplyLocations, and associated metadata.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or a recognized file type.
    ValueError
        If `data_type` is invalid, or if coordinate columns are not correctly specified.
    KeyError
        If required columns are missing from the input data.
        
    Notes
    -----
    If `data` does not contain geometry information, `customer_origins_coords_col` and/or
    `supply_locations_coords_col` must be provided if interaction matrix must be interpreted as spatial data.

    Examples
    --------
    >>> Wieland2015_interaction_matrix = load_interaction_matrix(
    ...     data="data/Wieland2015.xlsx",
    ...     customer_origins_col="Quellort",
    ...     supply_locations_col="Zielort",
    ...     attraction_col=[
    ...         "VF", 
    ...         "K", 
    ...         "K_KKr"
    ...         ],
    ...     market_size_col="Sum_Ek1",
    ...     flows_col="Anb_Eink1",
    ...     transport_costs_col="Dist_Min2",
    ...     probabilities_col="MA_Anb1",
    ...     data_type="xlsx"
    ... )
    """
    
    if isinstance(data, pd.DataFrame):
        
        interaction_matrix_df = data

    elif isinstance(data, str):

        if data_type not in ["csv", "xlsx"]:

            raise ValueError ("Error while loading interaction matrix: param 'data_type' must be 'csv' or 'xlsx'")
        
        if data_type == "csv":

            interaction_matrix_df = pd.read_csv(
                data, 
                sep = csv_sep, 
                decimal = csv_decimal, 
                encoding = csv_encoding
                )
                
        elif data_type == "xlsx":

            if xlsx_sheet is not None:

                interaction_matrix_df = pd.read_excel(
                    data, 
                    sheet_name=xlsx_sheet
                    )
                
            else:

                interaction_matrix_df = pd.read_excel(data)
        else:

            raise ValueError("Error while loading interaction matrix: Unknown type of data")
        
    else:
        raise ValueError("Error while loading interaction matrix: param 'data' must be pandas.DataFrame or file (.csv, .xlsx)")
    
    import_key_errors = []   
    if customer_origins_col not in interaction_matrix_df.columns:
        import_key_errors.append(f"Column {customer_origins_col} not in data.")
    if supply_locations_col not in interaction_matrix_df.columns:
        import_key_errors.append(f"Column {supply_locations_col} not in data.")
    if len(import_key_errors) > 0:
        raise InteractionMatrixError(f"Error(s) while importing interaction matrix: {' '.join(import_key_errors)}.")

    import_col_errors = []
    if attraction_col is None or (isinstance(attraction_col, list) and len(attraction_col) == 0):
        import_col_errors.append(f"No attraction column(s) stated.")
    if transport_costs_col is None or (isinstance(transport_costs_col, list) and len(transport_costs_col) == 0):
        import_col_errors.append(f"No transport costs column stated.")
    if len(import_col_errors) > 0:
        raise InteractionMatrixError(f"Error(s) while importing interaction matrix: {' '.join(import_col_errors)}.")
    
    cols_check = attraction_col + [transport_costs_col]
    if flows_col is not None:
        cols_check = cols_check + [flows_col]
    if probabilities_col is not None:
        cols_check = cols_check + [probabilities_col]
    if market_size_col is not None:
        cols_check = cols_check + [market_size_col]

    if check_df_vars:

        if verbose:
            print(f"Checking input variables {', '.join(cols_check)}", end = " ... ")

        helper.check_vars(
            interaction_matrix_df,
            cols = cols_check
            )
        
        if verbose:
            print("OK")
        
    type_errors_customer_origins = []
    
    if customer_origins_coords_col is not None:

        if verbose:
            print(f"Checking and processing customer origins coordinates", end = " ... ")

        if isinstance(customer_origins_coords_col, str):

            if check_df_vars:
                
                helper.check_vars(
                    interaction_matrix_df,
                    cols = [customer_origins_coords_col],
                    check_constant = False
                    )
            
            if customer_origins_coords_col == "geometry":
                customer_origins_geodata_tab = interaction_matrix_df[[customer_origins_col, "geometry"]]
            
            else:

                customer_origins_geodata_tab = interaction_matrix_df[[customer_origins_col, customer_origins_coords_col]]
                
                try:
                    customer_origins_geodata_tab["geometry"] = customer_origins_geodata_tab[customer_origins_coords_col].apply(lambda x: wkt.loads(x))                                    
                except Exception as e:
                    type_errors_customer_origins.append(f"No valid geometry information in column '{customer_origins_coords_col}' for customer origins: {str(e)}.")             
        
        elif isinstance(customer_origins_coords_col, list):

            if len(customer_origins_coords_col) != 2:
                raise ValueError (f"Error while loading interaction matrix: Column {customer_origins_coords_col} must be a geometry column OR TWO columns with X and Y")
            
            if check_df_vars:
                helper.check_vars (
                    df = interaction_matrix_df, 
                    cols = customer_origins_coords_col,
                    check_constant=False
                    )
            
            customer_origins_geodata_tab = interaction_matrix_df[[customer_origins_col, customer_origins_coords_col[0], customer_origins_coords_col[1]]]

            try:
                customer_origins_geodata_tab["geometry"] = customer_origins_geodata_tab.apply(lambda row: Point(row[customer_origins_coords_col[0]], row[customer_origins_coords_col[1]]), axis=1)
            except Exception as e:
                type_errors_customer_origins.append(f"No valid geometry information in columns {', '.join(customer_origins_coords_col)} for customer origins: {str(e)}.")
                    
        if len(type_errors_customer_origins) == 0:

            customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop(columns = customer_origins_coords_col)
            
            customer_origins_geodata_tab = customer_origins_geodata_tab.drop_duplicates()
            
            try:

                customer_origins_geodata_gpd = gp.GeoDataFrame(
                    customer_origins_geodata_tab, 
                    geometry="geometry",
                    crs = crs_input
                    )
                
                customer_origins_geodata_gpd.set_crs(crs_output, inplace=True)

            except Exception as e:
                type_errors_customer_origins.append(f"No valid geometry information for customer origins: {str(e)}.")
                  
        else:

            customer_origins_geodata_gpd = interaction_matrix_df[customer_origins_col]
            customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates()

        if verbose:
            print("OK")

    else:

        customer_origins_geodata_gpd = interaction_matrix_df[customer_origins_col]
        customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates()

    if verbose:
        print("Compiling customer origins data", end = " ... ")

    customer_origins_coords_col_list = []
    if customer_origins_coords_col is not None:
        if isinstance(customer_origins_coords_col, list):
            customer_origins_coords_col_list = customer_origins_coords_col
        elif isinstance(customer_origins_coords_col, str):
            customer_origins_coords_col_list = [customer_origins_coords_col]     
    
    if market_size_col is not None:
        customer_origins_cols = [customer_origins_col] + [market_size_col] + customer_origins_coords_col_list
    else:
        customer_origins_cols = [customer_origins_col] + customer_origins_coords_col_list        
        
    customer_origins_geodata_original_tab = interaction_matrix_df[customer_origins_cols].drop_duplicates()
    
    if len(customer_origins_coords_col_list) > 0 and len(type_errors_customer_origins) == 0:

        if isinstance(customer_origins_coords_col, list):
            customer_origins_geodata_original_tab["geometry"] = customer_origins_geodata_original_tab.apply(lambda row: Point(row[customer_origins_coords_col[0]], row[customer_origins_coords_col[1]]), axis=1)
        
        elif isinstance(customer_origins_coords_col, str):
            customer_origins_geodata_original_tab = customer_origins_geodata_original_tab.rename(
                columns = {
                    customer_origins_coords_col: "geometry"
                }
            )

    customer_origins_geodata_gpd_original = customer_origins_geodata_original_tab

    if len(customer_origins_coords_col_list) > 0 and len(type_errors_customer_origins) == 0:
        
        customer_origins_geodata_gpd_original = gp.GeoDataFrame(
            customer_origins_geodata_original_tab, 
            geometry="geometry",
            crs = crs_input
            )
                
    customer_origins_metadata = {
        "location_type": "origins",
        "unique_id": customer_origins_col,
        "attraction_col": [None],
        "marketsize_col": market_size_col,
        "weighting": {
            0: {
                "name": None,
                "func": None, 
                "param": None
                }
            },
        "crs_input": crs_input,
        "crs_output": crs_output,
        "no_points": len(customer_origins_geodata_gpd),
        }

    customer_origins = CustomerOrigins(
        geodata_gpd = customer_origins_geodata_gpd,
        geodata_gpd_original = customer_origins_geodata_gpd_original,
        metadata = customer_origins_metadata,
        isochrones_gdf = None,
        buffers_gdf = None
        )
    
    status = "OK"
    if len(type_errors_customer_origins) > 0:
        status = f"{' '.join(type_errors_customer_origins)}"
     
    helper.add_timestamp(
        customer_origins,
        function="data_management.load_interaction_matrix",
        process = "Creation by import",
        status=status
        )
    
    if verbose:
        print("OK")

    type_errors_supply_locations = []

    if supply_locations_coords_col is not None:

        if verbose:
            print("Checking and processing supply locations coordinates", end = " ... ")

        if isinstance(supply_locations_coords_col, str):

            if check_df_vars:
                
                helper.check_vars(
                    interaction_matrix_df,
                    cols = [supply_locations_coords_col],
                    check_constant = False
                    )

            if supply_locations_coords_col == "geometry":
                supply_locations_geodata_tab = interaction_matrix_df[[supply_locations_col, "geometry"]]

            else:

                supply_locations_geodata_tab = interaction_matrix_df[[supply_locations_col, supply_locations_coords_col]]

                try:
                    supply_locations_geodata_tab["geometry"] = supply_locations_geodata_tab[supply_locations_coords_col].apply(lambda x: wkt.loads(x))
                except Exception as e:
                    type_errors_supply_locations.append(f"No valid geometry information in column '{supply_locations_coords_col}' for supply locations: {str(e)}.")

        elif isinstance(supply_locations_coords_col, list):

            if len(supply_locations_coords_col) != 2:
                raise ValueError (f"Error while loading interaction matrix: Column {supply_locations_coords_col} must be a geometry column OR TWO columns with X and Y")
            
            if check_df_vars:
                helper.check_vars (
                    df = interaction_matrix_df, 
                    cols = supply_locations_coords_col,
                    check_constant=False
                    )

            supply_locations_geodata_tab = interaction_matrix_df[[supply_locations_col, supply_locations_coords_col[0], supply_locations_coords_col[1]]]

            try:
                supply_locations_geodata_tab["geometry"] = supply_locations_geodata_tab.apply(lambda row: Point(row[supply_locations_coords_col[0]], row[supply_locations_coords_col[1]]), axis=1)
            except Exception as e:
                type_errors_supply_locations.append(f"No valid geometry information in columns '{', '.join(supply_locations_coords_col)}' for supply locations: {str(e)}.")

        if len(type_errors_supply_locations) == 0:

            supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop(columns = customer_origins_coords_col)
            
            supply_locations_geodata_tab = supply_locations_geodata_tab.drop_duplicates()
            
            try:

                supply_locations_geodata_gpd = gp.GeoDataFrame(
                    supply_locations_geodata_tab, 
                    geometry="geometry",
                    crs = crs_input
                    )
                
                supply_locations_geodata_gpd.set_crs(crs_output, inplace=True)

            except Exception as e:
                type_errors_supply_locations.append(f"No valid geometry information for supply locations: {str(e)}.")
                  
        else:

            supply_locations_geodata_gpd = interaction_matrix_df[supply_locations_col]
            supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates()

        if verbose:
            print("OK")

    else:

        supply_locations_geodata_gpd = interaction_matrix_df[supply_locations_col]
        supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates()

    if verbose:
        print("Compiling supply locations data", end = " ... ")

    supply_locations_coords_col_list = []
    if supply_locations_coords_col is not None:
        if isinstance(supply_locations_coords_col, list):
            supply_locations_coords_col_list = supply_locations_coords_col
        elif isinstance(supply_locations_coords_col, str):
            supply_locations_coords_col_list = [supply_locations_coords_col]     
    
    if len(attraction_col) > 0:
        supply_locations_cols = [supply_locations_col] + attraction_col + supply_locations_coords_col_list
    else:
        supply_locations_cols = [supply_locations_col] + supply_locations_coords_col_list        
        
    supply_locations_geodata_original_tab = interaction_matrix_df[supply_locations_cols].drop_duplicates()
    
    if len(supply_locations_coords_col_list) > 0 and len(type_errors_supply_locations) == 0: 

        if isinstance(supply_locations_coords_col, list):
            supply_locations_geodata_original_tab["geometry"] = supply_locations_geodata_original_tab.apply(lambda row: Point(row[supply_locations_coords_col[0]], row[supply_locations_coords_col[1]]), axis=1)
        
        elif isinstance(supply_locations_coords_col, str):
            supply_locations_geodata_original_tab = supply_locations_geodata_original_tab.rename(
                columns = {
                    supply_locations_coords_col: "geometry"
                }
            )

    supply_locations_geodata_gpd_original = supply_locations_geodata_original_tab
    
    if len(supply_locations_coords_col_list) > 0 and len(type_errors_supply_locations) == 0:        
        
        supply_locations_geodata_gpd_original = gp.GeoDataFrame(
            supply_locations_geodata_original_tab, 
            geometry = "geometry",
            crs = crs_input
            )
        
    supply_locations_geodata_gpd_original[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] = 0
               
    supply_locations_metadata = {
        "location_type": "destinations",
        "unique_id": supply_locations_col,
        "attraction_col": attraction_col,
        "marketsize_col": None,
        "weighting": {
            0: {
                "name": None,
                "func": None, 
                "param": None
                }
            },
        "crs_input": crs_input,
        "crs_output": crs_output,
        "no_points": len(supply_locations_geodata_gpd),
        }
    
    supply_locations = SupplyLocations(
        geodata_gpd = supply_locations_geodata_gpd,
        geodata_gpd_original = supply_locations_geodata_gpd_original,
        metadata = supply_locations_metadata,
        isochrones_gdf = None,
        buffers_gdf = None,
        competitor_concentration = None
        )
    
    status = "OK"
    if len(type_errors_supply_locations) > 0:
        status = f"{' '.join(type_errors_supply_locations)}"

    helper.add_timestamp(
        supply_locations,
        function="data_management.load_interaction_matrix",
        process = "Creation by import",
        status=status
        )
    
    if verbose:
        print("OK")
        print("Compiling interaction matrix", end = " ... ")
    
    interaction_matrix_df = interaction_matrix_df.rename(
        columns = {
            customer_origins_col: config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            supply_locations_col: config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
            attraction_col[0]: config.DEFAULT_COLNAME_ATTRAC,
            transport_costs_col: config.DEFAULT_COLNAME_TC
        }
        )

    if flows_col is not None:
        interaction_matrix_df = interaction_matrix_df.rename(
            columns = {
                flows_col: config.DEFAULT_COLNAME_FLOWS
            }
            )

    if probabilities_col is not None:
        interaction_matrix_df = interaction_matrix_df.rename(
            columns = {
                probabilities_col: config.DEFAULT_COLNAME_PROBABILITY
            }
            )

    if market_size_col is not None:
        interaction_matrix_df = interaction_matrix_df.rename(
            columns = {
                market_size_col: config.DEFAULT_COLNAME_MARKETSIZE
            }
            )
    
    metadata = {
        "fit": {
            "function": None,
            "fit_by": None
        },
        "transport_costs_col": transport_costs_col,
        "transport_costs": {
            "metrics": transport_costs_metrics,
            "distance_unit": transport_costs_distance_unit,
            "time_unit": transport_costs_time_unit
        }
    }

    interaction_matrix = InteractionMatrix(
        interaction_matrix_df=interaction_matrix_df,
        customer_origins=customer_origins,
        supply_locations=supply_locations,
        metadata=metadata
        )
    
    status = "OK"
    coord_type_errors = type_errors_customer_origins + type_errors_supply_locations
    if len(coord_type_errors) > 0:
        print(f"WARNING: No import as geodata due to invalid geometry information in input data: {' '.join(coord_type_errors)}")
        status = f"No import as geodata due to invalid geometry information in input data: {' '.join(coord_type_errors)}"
    
    helper.add_timestamp(
        interaction_matrix,
        function="data_management.load_interaction_matrix",
        process = "Creation by import",
        status = status
        )
    
    if verbose:
        print("OK")
    
    return interaction_matrix

def load_marketareas(
    data,
    supply_locations_col: str,
    total_col: str,  
    data_type = "csv", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape",
    xlsx_sheet: str | None = None,
    check_df_vars: bool = True
    ) -> MarketAreas:

    """
    data : str or pandas.DataFrame
        Input data containing market area information. Can be a file path
        or an in-memory DataFrame.
    supply_locations_col : str
        Name of the column identifying supply locations.
    total_col : str
        Name of the column containing total market area values.
    data_type : {"csv", "xlsx"}, default="csv"
        File type if data is loaded from disk.
    csv_sep : str, default=";"
        Column separator used in CSV files.
    csv_decimal : str, default=","
        Decimal mark used in CSV files.
    csv_encoding : str, default="unicode_escape"
        Encoding used in CSV files.
    xlsx_sheet : str, optional
        Name of the Excel sheet to read, if reading from an XLSX file.
    check_df_vars : bool, default=True
        If True, checks whether required columns are present in the input data
        using helper.check_vars.

    Returns
    -------
    MarketAreas
        A MarketAreas object containing the imported data and metadata.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or recognized file type.
    ValueError
        If `data_type` is not "csv" or "xlsx" when providing a file path.
    KeyError
        If required columns (`supply_locations_col` or `total_col`) are missing
        in the input data.

    Examples
    --------
    >>> wieland2015_totalmarketareas = load_marketareas(
    ...     data="data/Wieland2015.xlsx",
    ...     supply_locations_col="Zielort",
    ...     total_col="Anb_Eink",
    ...     data_type="xlsx",
    ...     xlsx_sheet="total_marketareas"
    ... )
    >>> huff_model_fit2 = wieland2015_totalmarketareas.add_to_model(huff_model_fit2)    
    """

    if isinstance(data, pd.DataFrame):
        market_areas_df = data
    elif isinstance(data, str):
        if data_type not in ["csv", "xlsx"]:
            raise ValueError ("Error while loading market areas: data_type must be 'csv' or 'xlsx'")
        if data_type == "csv":
            market_areas_df = pd.read_csv(
                data, 
                sep = csv_sep, 
                decimal = csv_decimal, 
                encoding = csv_encoding
                )    
        elif data_type == "xlsx":            
            if xlsx_sheet is not None:
                market_areas_df = pd.read_excel(
                    data, 
                    sheet_name=xlsx_sheet
                    )
            else:
                market_areas_df = pd.read_excel(data)
        else:
            raise TypeError("Error while loading market areas: Unknown type of data")
    else:
        raise TypeError("Error while loading market areas: data must be pandas.DataFrame or file (.csv, .xlsx)")
    
    if supply_locations_col not in market_areas_df.columns:
        raise KeyError (f"Error while loading market areas: Column {supply_locations_col} not in data")
    if total_col not in market_areas_df.columns:
        raise KeyError (f"Error while loading market areas: Column {total_col} not in data")
    
    if check_df_vars:
        helper.check_vars(
            market_areas_df,
            cols = [total_col]
            )    
   
    market_areas_df = market_areas_df.rename(
        columns = {
            supply_locations_col: config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
            total_col: config.DEFAULT_COLNAME_TOTAL_MARKETAREA
        }
        )
    
    metadata = {
        "unique_id": supply_locations_col,
        "total_col": total_col,
        "no_points": len(market_areas_df),
        }
    
    market_areas = MarketAreas(
        market_areas_df,
        metadata
        )
    
    helper.add_timestamp(
        market_areas,
        function="data_management.load_marketareas",
        process = "Creation by import"
        )
    
    return market_areas

def survey_to_matrix(
    survey_data,
    customer_origins_col: str,
    supply_locations_col: str,
    attraction_col: list[str] | None = None,
    transport_costs_col: str | None = None,
    flows_col: str | None = None,
    aggregate_at_col: str | None = None,
    customer_origins_coords_col = None,
    supply_locations_coords_col = None,
    data_type = "csv", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape",
    xlsx_sheet: str | None = None,
    check_df_vars: bool = True,
    verbose: bool = False
    ) -> pd.DataFrame:
    
    """
    Convert survey data into a fully expanded origin-destination matrix.

    This function transforms a survey-style table of observed origin ->
    destination choices into a complete origin-destination (OD) DataFrame
    containing one row for every possible origin-destination pair that
    occurs in the input. It aggregates counts (and optional observed flows),
    computes market shares per origin, and attaches optional attributes such
    as attraction values, transport costs and coordinates when available.

    Parameters
    ----------
    survey_data : str or pandas.DataFrame
        Input survey data. Either a path to a CSV/XLSX file or a pandas
        DataFrame already loaded into memory.
    customer_origins_col : str
        Column name that identifies the customer origin (unique origin id).
    supply_locations_col : str
        Column name that identifies the supply location (unique destination id).
    attraction_col : list[str], optional
        Column name(s) providing attraction values for supply locations. If
        provided, these columns are merged into the resulting OD table (one
        record per supply location).
    transport_costs_col : str, optional
        Column name providing transport costs (distance or time) for the
        observed OD pairs. When present, the column is joined to the result
        by the OD identifier.
    flows_col : str, optional
        Column name containing observed flows (e.g. trips, purchases). If
        provided, flows are aggregated by OD and an additional market-share
        column for the flows is computed named
        `f"{config.DEFAULT_COLNAME_PROBABILITY}_{flows_col}"`.
    aggregate_at_col : str, optional
        Reserved parameter intended to indicate a column used to aggregate
        supply locations prior to expansion (for example grouping outlets
        into chains). Note: in the current implementation this parameter is
        validated for existence but not otherwise used by the function.
    customer_origins_coords_col : str or list[str], optional
        Column name or two-element list of column names containing the
        coordinates for customer origins. If a list is supplied it is
        interpreted as `[x_col, y_col]`.
    supply_locations_coords_col : str or list[str], optional
        Column name or two-element list of column names containing the
        coordinates for supply locations. If a list is supplied it is
        interpreted as `[x_col, y_col]`.
    data_type : {"csv", "xlsx"}, default="csv"
        File type when `survey_data` is a file path.
    csv_sep : str, default=";"
        Separator for CSV files.
    csv_decimal : str, default="," 
        Decimal mark for CSV files.
    csv_encoding : str, default="unicode_escape"
        Encoding for CSV files.
    xlsx_sheet : str, optional
        Sheet name for XLSX input.
    check_df_vars : bool, default=True
        If True, performs presence and basic validity checks on required
        columns using `helper.check_vars`.
    verbose : bool, default=False
        If True, print progress messages.

    Returns
    -------
    pandas.DataFrame
        Expanded interaction matrix with one row per origin-destination
        pair. The DataFrame always contains at least the following columns:
        - the original `customer_origins_col` and `supply_locations_col`,
        - the interaction identifier stored in `config.DEFAULT_COLNAME_INTERACTION`,
        - `count`: aggregated number of observed occurrences for this OD pair,
        - a market-share column per origin created by `market_shares()` using
          the `count` column (the default market-share column name is
          `config.DEFAULT_COLNAME_PROBABILITY`).

    Raises
    ------
    TypeError
        If `survey_data` is not a DataFrame or a supported file type.
    ValueError
        If `data_type` is invalid or coordinate arguments are provided in
        an incorrect format.
    KeyError
        If required columns (e.g. `customer_origins_col` or
        `supply_locations_col`) or user-specified coordinate columns are
        missing from the input data.

    Examples
    --------
    >>> df_od = survey_to_matrix(
    ...     survey_data="data/survey.csv",
    ...     customer_origins_col="origin_id",
    ...     supply_locations_col="store_id",
    ...     attraction_col=["floorspace"],
    ...     transport_costs_col="drive_time",
    ...     flows_col="trips",
    ...     data_type="csv"
    ... )
    """

    if attraction_col is None:
        attraction_col = []

    if isinstance(survey_data, pd.DataFrame):
        data = survey_data
    elif isinstance(survey_data, str):
        if data_type not in ["csv", "xlsx"]:
            raise ValueError ("Error while loading survey data: param 'data_type' must be 'csv' or 'xlsx'")
        if data_type == "csv":
            data = pd.read_csv(
                survey_data, 
                sep = csv_sep, 
                decimal = csv_decimal, 
                encoding = csv_encoding
                )    
        elif data_type == "xlsx":
            if xlsx_sheet is not None:
                data = pd.read_excel(
                    survey_data, 
                    sheet_name=xlsx_sheet
                    )
            else:
                data = pd.read_excel(survey_data)
        else:
            raise TypeError("Error while loading survey data: Unknown type of data")
    else:
        raise TypeError("Error while loading survey data: param 'data' must be pandas.DataFrame or file (.csv, .xlsx)")
    
    if customer_origins_col not in data.columns:
        raise KeyError (f"Error while loading survey data: Column {customer_origins_col} not in data")
    if supply_locations_col not in data.columns:
        raise KeyError (f"Error while loading survey data: Column {supply_locations_col} not in data")
    
    if check_df_vars: 
        if flows_col is not None:
            helper.check_vars(
                data,
                cols = [flows_col],
                check_zero=False
                )
            
        if customer_origins_coords_col is not None and customer_origins_coords_col != []:
            
            if isinstance(customer_origins_coords_col, str):
                
                if customer_origins_coords_col not in survey_data.columns:
                    raise KeyError (f"Error while loading survey data: Column {customer_origins_coords_col} not in data.")
                
                customer_origins_coords_col = [customer_origins_coords_col]
                
            elif isinstance(customer_origins_coords_col, list):
                
                missing_cols = [col for col in customer_origins_coords_col if col not in survey_data.columns]
                
                if len(missing_cols) > 0:
                    raise KeyError (f"Error while loading survey data: Columns {', '.join(missing_cols)} not in data.")
                
            else:
                
                customer_origins_coords_col = None
                
                print(f"WARNING: Parameter 'customer_origins_coords_col' was stated in an unknown format and is ignored: {customer_origins_coords_col}")
                                
        if supply_locations_coords_col is not None and supply_locations_coords_col != []:
            
            if isinstance(supply_locations_coords_col, str):
                
                if supply_locations_coords_col not in survey_data.columns:
                    raise KeyError (f"Error while loading survey data: Column {supply_locations_coords_col} not in data.")
                
                supply_locations_coords_col = [supply_locations_coords_col]
                
            elif isinstance(supply_locations_coords_col, list):
                
                missing_cols = [col for col in supply_locations_coords_col if col not in survey_data.columns]
                
                if len(missing_cols) > 0:
                    raise KeyError (f"Error while loading survey data: Columns {', '.join(missing_cols)} not in data.")
                
            else:               
               
                supply_locations_coords_col = None
                
                print(f"WARNING: Parameter 'supply_locations_coords_col' was stated in an unknown format and is ignored: {supply_locations_coords_col}")
    
    if aggregate_at_col is not None:

        if aggregate_at_col not in data.columns:
            raise KeyError(f"Column specified data to be aggregated {aggregate_at_col} is not in the data frame.")
        
        customer_origins_col = aggregate_at_col

        if verbose:
            print(f"NOTE: Survey data will be aggregated as the consumer aggregation column '{aggregate_at_col}'.")

    if verbose:
        print("Compiling survey data", end = " ... ")

    customer_origins = (
        data[[customer_origins_col]]
        .dropna()
        .drop_duplicates()
        )
    supply_locations = (
        data[[supply_locations_col]]
        .dropna()
        .drop_duplicates()
        )
    customer_origins["pseudo_key"] = 1
    supply_locations["pseudo_key"] = 1

    origins_x_locations = (
        customer_origins
        .merge(supply_locations, on="pseudo_key")
        .drop(columns="pseudo_key")
        )
    origins_x_locations[config.DEFAULT_COLNAME_INTERACTION] = origins_x_locations[customer_origins_col].astype(str)+config.MATRIX_OD_SEPARATOR+origins_x_locations[supply_locations_col].astype(str)
    
    data["count"] = 1
    
    data_agg = data.groupby(
        [
            customer_origins_col, 
            supply_locations_col
            ],
        as_index=False
        )["count"].sum()
    
    data_agg[config.DEFAULT_COLNAME_INTERACTION] = data_agg[customer_origins_col].astype(str)+config.MATRIX_OD_SEPARATOR+data_agg[supply_locations_col].astype(str)
    
    data_agg = origins_x_locations.merge(
        data_agg[[config.DEFAULT_COLNAME_INTERACTION, "count"]],
        left_on = config.DEFAULT_COLNAME_INTERACTION,
        right_on=config.DEFAULT_COLNAME_INTERACTION,
        how = "left"
    )
    
    data_agg["count"] = data_agg["count"].fillna(0)
    
    data_agg = market_shares(
        df = data_agg,
        turnover_col = "count",
        ref_col = customer_origins_col,
        check_df_vars = False
        )

    if verbose:
        print("OK")
    
    if flows_col is not None:

        if verbose:
            print(f"Flows in column '{flows_col}' in survey data are compiled", end = " ... ")
        
        data_agg_flowscol = data.groupby(
            [
                customer_origins_col, 
                supply_locations_col
                ],
            as_index=False
            )[[flows_col]].sum()
        
        data_agg_flowscol[config.DEFAULT_COLNAME_INTERACTION] = data_agg_flowscol[customer_origins_col].astype(str)+config.MATRIX_OD_SEPARATOR+data_agg_flowscol[supply_locations_col].astype(str)
        
        data_agg_flowscol = data_agg.merge(
            data_agg_flowscol[[config.DEFAULT_COLNAME_INTERACTION, flows_col]],
            left_on = config.DEFAULT_COLNAME_INTERACTION,
            right_on=config.DEFAULT_COLNAME_INTERACTION,
            how = "left"
        )
        
        data_agg_flowscol[flows_col] = data_agg_flowscol[flows_col].fillna(0)
        
        data_agg_flowscol = market_shares(
            df = data_agg_flowscol,
            turnover_col = flows_col,
            ref_col = customer_origins_col,
            marketshares_col = f"{config.DEFAULT_COLNAME_PROBABILITY}_{flows_col}",
            check_df_vars = False
            )
        
        data_agg = pd.merge(
            data_agg,
            data_agg_flowscol[[config.DEFAULT_COLNAME_INTERACTION, flows_col, f"{config.DEFAULT_COLNAME_PROBABILITY}_{flows_col}"]],
            left_on = config.DEFAULT_COLNAME_INTERACTION,
            right_on = config.DEFAULT_COLNAME_INTERACTION
        )
        
        if verbose:
            print("OK")
    
    data_agg = data_agg.sort_values(
        by = [
            customer_origins_col,
            supply_locations_col
        ]
    )
    
    if len(attraction_col) > 0:
        
        if verbose:
            print(f"Checking and including {len(attraction_col)} attraction columns", end = " ... ")

        helper.check_vars(
            data,
            cols = attraction_col,
            check_zero=False
            )
        
        data_attraction = data[[supply_locations_col] + attraction_col].drop_duplicates(subset=[supply_locations_col])
        
        data_agg = data_agg.merge(
            data_attraction,
            on = supply_locations_col,
            how = "left"
        ) 

        if verbose:
            print("OK")       
      
    if transport_costs_col is not None:

        if verbose:
            print(f"Checking and including transport costs column '{transport_costs_col}'", end = " ... ")
        
        helper.check_vars(
            data,
            cols = [transport_costs_col],
            check_zero=False
            )
        
        data_transport_costs = data[[customer_origins_col, supply_locations_col, transport_costs_col]]
        data_transport_costs[config.DEFAULT_COLNAME_INTERACTION] = data_transport_costs[customer_origins_col].astype(str)+config.MATRIX_OD_SEPARATOR+data_transport_costs[supply_locations_col].astype(str)
        data_transport_costs = data_transport_costs[[config.DEFAULT_COLNAME_INTERACTION, transport_costs_col]].drop_duplicates(subset=[config.DEFAULT_COLNAME_INTERACTION])
        
        data_agg = data_agg.merge(
            data_transport_costs,
            on=config.DEFAULT_COLNAME_INTERACTION,
            how="left"
        )        
        
        if verbose:
            print("OK")

    if customer_origins_coords_col is not None:

        if verbose:
            print(f"Checking and including customer origins coordinates", end = " ... ")
        
        helper.check_vars(
            data,
            cols = customer_origins_coords_col,
            check_zero=False
            )
        
        data_customer_origins_coords = data[[customer_origins_col] + customer_origins_coords_col].drop_duplicates(subset=[customer_origins_col])
                
        data_agg = data_agg.merge(
            data_customer_origins_coords,
            on = customer_origins_col,
            how = "left"
        )
        
        if verbose:
            print("OK")
        
    if supply_locations_coords_col is not None:

        if verbose:
            print(f"Checking and including supply locations coordinates", end = " ... ")
        
        helper.check_vars(
            data,
            cols = supply_locations_coords_col,
            check_zero=False
            )
        
        data_supply_locations_coords = data[[supply_locations_col] + supply_locations_coords_col].drop_duplicates(subset=[supply_locations_col])
        
        data_agg = data_agg.merge(
            data_supply_locations_coords,
            on = supply_locations_col,
            how = "left"
        )
        
        if verbose:
            print("OK")
    
    return data_agg