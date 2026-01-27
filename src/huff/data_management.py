#-----------------------------------------------------------------------
# Name:        data_management (huff package)
# Purpose:     Loading of external data for Huff/MCI model analyses
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.4
# Last update: 2026-01-27 18:07
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import geopandas as gp
from shapely import wkt
from shapely.geometry import Point
from huff.models import SupplyLocations, CustomerOrigins, InteractionMatrix, MarketAreas, market_shares
import huff.helper as helper
import huff.config as config


def load_geodata(
    data, 
    location_type: str, 
    unique_id: str,
    x_col: str = None, 
    y_col: str = None,
    data_type = "shp", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape", 
    crs_input = "EPSG:4326"    
    ):

    if location_type is None or (location_type not in config.PERMITTED_LOCATION_TYPES):
        raise ValueError (f"Error while loading geodata: Argument location_type must be one of the following: {', '.join(config.PERMITTED_LOCATION_TYPES)}")

    if isinstance(data, gp.GeoDataFrame):
        geodata_gpd_original = data
        if not all(geodata_gpd_original.geometry.geom_type == "Point"):
            raise TypeError ("Error while loading geodata: Input geopandas.GeoDataFrame must be of type 'Point'")
        crs_input = geodata_gpd_original.crs
    elif isinstance(data, pd.DataFrame):
        geodata_tab = data
    elif isinstance(data, str):
        if data_type == "shp":
            geodata_gpd_original = gp.read_file(data)
            if not all(geodata_gpd_original.geometry.geom_type == "Point"):
                raise TypeError ("Error while loading geodata: Input shapefile must be of type 'Point'")
            crs_input = geodata_gpd_original.crs
        elif data_type == "csv" or data_type == "xlsx":
            if x_col is None:
                raise ValueError ("Error while loading geodata: Missing value for X coordinate column")
            if y_col is None:
                raise ValueError ("Error while loading geodata: Missing value for Y coordinate column")
        elif data_type == "csv":
            geodata_tab = pd.read_csv(
                data, 
                sep = csv_sep, 
                decimal = csv_decimal, 
                encoding = csv_encoding
                ) 
        elif data_type == "xlsx":
            geodata_tab = pd.read_excel(data)
        else:
            raise TypeError("Error while loading geodata: Unknown type of data")
    else:
        raise TypeError("Error while loading geodata: Param 'data' must be pandas.DataFrame, geopandas.GeoDataFrame or file (.csv, .xlsx, .shp)")

    if data_type == "csv" or data_type == "xlsx" or (isinstance(data, pd.DataFrame) and not isinstance(data, gp.GeoDataFrame)):
        
        helper.check_vars(
            df = geodata_tab,
            cols = [x_col, y_col]
            )
        
        geodata_gpd_original = gp.GeoDataFrame(
            geodata_tab, 
            geometry = gp.points_from_xy(
                geodata_tab[x_col], 
                geodata_tab[y_col]
                ), 
            crs = crs_input
            )
        
    crs_output = config.WGS84_CRS

    geodata_gpd = geodata_gpd_original.to_crs(crs_output)
    geodata_gpd = geodata_gpd[[unique_id, "geometry"]]   
    
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
    
    metadata = helper.add_timestamp(
        function = "load_geodata",
        metadata = metadata
        )
    
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
            geodata_gpd, 
            geodata_gpd_original, 
            metadata,
            None,
            None
            )

    return geodata_object

def load_interaction_matrix(
    data,
    customer_origins_col: str,
    supply_locations_col: str,
    attraction_col: list,
    transport_costs_col: str,
    flows_col: str = None,
    probabilities_col: str = None,
    market_size_col: str = None,
    customer_origins_coords_col = None,
    supply_locations_coords_col = None,
    data_type = "csv", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape",
    xlsx_sheet: str = None,
    crs_input = "EPSG:4326",
    crs_output = "EPSG:4326",
    check_df_vars: bool = True
    ):    

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
            raise TypeError("Error while loading interaction matrix: Unknown type of data")
    else:
        raise TypeError("Error while loading interaction matrix: param 'data' must be pandas.DataFrame or file (.csv, .xlsx)")
    
    if customer_origins_col not in interaction_matrix_df.columns:
        raise KeyError (f"Error while loading interaction matrix: Column {customer_origins_col} not in data")
    if supply_locations_col not in interaction_matrix_df.columns:
        raise KeyError (f"Error while loading interaction matrix: Column {supply_locations_col} not in data")
    
    cols_check = attraction_col + [transport_costs_col]
    if flows_col is not None:
        cols_check = cols_check + [flows_col]
    if probabilities_col is not None:
        cols_check = cols_check + [probabilities_col]
    if market_size_col is not None:
        cols_check = cols_check + [market_size_col]

    if check_df_vars:
        helper.check_vars(
            interaction_matrix_df,
            cols = cols_check
            )   
    
    if customer_origins_coords_col is not None:

        if isinstance(customer_origins_coords_col, str):

            if customer_origins_coords_col not in interaction_matrix_df.columns:
                raise KeyError (f"Error while loading interaction matrix: Column {customer_origins_coords_col} not in data.")    
            
            customer_origins_geodata_tab = interaction_matrix_df[[customer_origins_col, customer_origins_coords_col]]
            customer_origins_geodata_tab = customer_origins_geodata_tab.drop_duplicates()
            customer_origins_geodata_tab["geometry"] = customer_origins_geodata_tab[customer_origins_coords_col].apply(lambda x: wkt.loads(x))
            
            customer_origins_geodata_gpd = gp.GeoDataFrame(
                customer_origins_geodata_tab, 
                geometry="geometry",
                crs = crs_input
                )
            
            customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop(
                columns = customer_origins_coords_col
                )

        elif isinstance(customer_origins_coords_col, list):

            if len(customer_origins_coords_col) != 2:
                raise ValueError (f"Error while loading interaction matrix: Column {customer_origins_coords_col} must be a geometry column OR TWO columns with X and Y")
            
            helper.check_vars (
                df = interaction_matrix_df, 
                cols = customer_origins_coords_col
                )

            customer_origins_geodata_tab = interaction_matrix_df[[customer_origins_col, customer_origins_coords_col[0], customer_origins_coords_col[1]]]
            customer_origins_geodata_tab = customer_origins_geodata_tab.drop_duplicates()
            customer_origins_geodata_tab["geometry"] = customer_origins_geodata_tab.apply(lambda row: Point(row[customer_origins_coords_col[0]], row[customer_origins_coords_col[1]]), axis=1)
            customer_origins_geodata_gpd = gp.GeoDataFrame(
                customer_origins_geodata_tab, 
                geometry="geometry"
                )
                      
        customer_origins_geodata_gpd.set_crs(crs_output, inplace=True)

    else:

        customer_origins_geodata_gpd = interaction_matrix_df[customer_origins_col]
        customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates()

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
    
    if isinstance(customer_origins_coords_col, list):
        customer_origins_geodata_original_tab["geometry"] = customer_origins_geodata_original_tab.apply(lambda row: Point(row[customer_origins_coords_col[0]], row[customer_origins_coords_col[1]]), axis=1)
    elif isinstance(customer_origins_coords_col, str):
        customer_origins_geodata_original_tab = customer_origins_geodata_original_tab.rename(
            columns = {
                customer_origins_coords_col: "geometry"
            }
        )

    customer_origins_geodata_gpd_original = customer_origins_geodata_original_tab

    if len(customer_origins_coords_col_list) > 0:
        
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
    
    customer_origins_metadata = helper.add_timestamp(
        function = "load_interaction_matrix",
        metadata = customer_origins_metadata
        )

    customer_origins = CustomerOrigins(
        geodata_gpd = customer_origins_geodata_gpd,
        geodata_gpd_original = customer_origins_geodata_gpd_original,
        metadata = customer_origins_metadata,
        isochrones_gdf = None,
        buffers_gdf = None
        )

    if supply_locations_coords_col is not None:

        if isinstance(supply_locations_coords_col, str):

            if supply_locations_coords_col not in interaction_matrix_df.columns:
                raise KeyError (f"Error while loading interaction matrix: Column {supply_locations_coords_col} not in data.")    
            
            supply_locations_geodata_tab = interaction_matrix_df[[supply_locations_col, supply_locations_coords_col]]
            supply_locations_geodata_tab = supply_locations_geodata_tab.drop_duplicates()
            supply_locations_geodata_tab["geometry"] = supply_locations_geodata_tab[supply_locations_coords_col].apply(lambda x: wkt.loads(x))
            supply_locations_geodata_gpd = gp.GeoDataFrame(
                supply_locations_geodata_tab, 
                geometry="geometry",
                crs = crs_input)
            supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop(
                columns = supply_locations_coords_col
                )

        if isinstance(supply_locations_coords_col, list):

            if len(supply_locations_coords_col) != 2:
                raise ValueError (f"Error while loading interaction matrix: Column {supply_locations_coords_col} must be a geometry column OR TWO columns with X and Y")
            
            helper.check_vars (
                df = interaction_matrix_df, 
                cols = supply_locations_coords_col
                )

            supply_locations_geodata_tab = interaction_matrix_df[[supply_locations_col, supply_locations_coords_col[0], supply_locations_coords_col[1]]]
            supply_locations_geodata_tab = supply_locations_geodata_tab.drop_duplicates()
            supply_locations_geodata_tab["geometry"] = supply_locations_geodata_tab.apply(lambda row: Point(row[supply_locations_coords_col[0]], row[supply_locations_coords_col[1]]), axis=1)
            supply_locations_geodata_gpd = gp.GeoDataFrame(supply_locations_geodata_tab, geometry="geometry")
                      
        supply_locations_geodata_gpd.set_crs(crs_output, inplace=True)

    else:

        supply_locations_geodata_gpd = interaction_matrix_df[supply_locations_col]
        supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates()

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
    
    if isinstance(supply_locations_coords_col, list):
        supply_locations_geodata_original_tab["geometry"] = supply_locations_geodata_original_tab.apply(lambda row: Point(row[supply_locations_coords_col[0]], row[supply_locations_coords_col[1]]), axis=1)
    elif isinstance(supply_locations_coords_col, str):
        supply_locations_geodata_original_tab = supply_locations_geodata_original_tab.rename(
            columns = {
                supply_locations_coords_col: "geometry"
            }
        )

    supply_locations_geodata_gpd_original = supply_locations_geodata_original_tab
    
    if len(supply_locations_coords_col_list) > 0:        
        
        supply_locations_geodata_gpd_original = gp.GeoDataFrame(
            supply_locations_geodata_original_tab, 
            geometry = "geometry",
            crs = crs_input
            )
               
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
    
    supply_locations_metadata = helper.add_timestamp(
        function = "load_interaction_matrix",
        metadata = supply_locations_metadata
        )

    supply_locations = SupplyLocations(
        geodata_gpd = supply_locations_geodata_gpd,
        geodata_gpd_original = supply_locations_geodata_gpd_original,
        metadata = supply_locations_metadata,
        isochrones_gdf = None,
        buffers_gdf = None
        )
    
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
    }
    
    metadata = helper.add_timestamp(
        function = "load_interaction_matrix",
        metadata = metadata
        )

    interaction_matrix = InteractionMatrix(
        interaction_matrix_df=interaction_matrix_df,
        customer_origins=customer_origins,
        supply_locations=supply_locations,
        metadata=metadata
        )
    
    return interaction_matrix

def load_marketareas(
    data,
    supply_locations_col: str,
    total_col: str,  
    data_type = "csv", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape",
    xlsx_sheet: str = None,
    check_df_vars: bool = True
    ):    

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
        raise KeyError (f"Error while loading market areas: Column {supply_locations_col} not in data")
    
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
    
    metadata = helper.add_timestamp(
        function = "load_marketareas",
        metadata = metadata
        )
    
    market_areas = MarketAreas(
        market_areas_df,
        metadata
        )
    
    return market_areas

def survey_to_matrix(
    survey_data,
    customer_origins_col: str,
    supply_locations_col: str,
    attraction_col: list = [],
    transport_costs_col: str = None,
    flows_col: str = None,
    customer_origins_coords_col = None,
    supply_locations_coords_col = None,
    data_type = "csv", 
    csv_sep = ";", 
    csv_decimal = ",", 
    csv_encoding="unicode_escape",
    xlsx_sheet: str = None,
    check_df_vars: bool = True
    ):

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
    
    if flows_col is not None:       
        
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
    
    data_agg = data_agg.sort_values(
        by = [
            customer_origins_col,
            supply_locations_col
        ]
    )
    
    if len(attraction_col) > 0:
        
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
      
    if transport_costs_col is not None:
        
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
        
    if customer_origins_coords_col is not None:
        
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
        
    if supply_locations_coords_col is not None:
        
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
    
    return data_agg   