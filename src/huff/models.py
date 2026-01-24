#-----------------------------------------------------------------------
# Name:        models (huff package)
# Purpose:     Huff Model classes and functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.7.1
# Last update: 2026-01-24 ??:??
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import geopandas as gp
import numpy as np
import time
from statsmodels.formula.api import ols
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from shapely import wkt
import copy
import huff.helper as helper
import huff.config as config
import huff.goodness_of_fit as gof
from huff.ors import Client, TimeDistanceMatrix, Isochrone
from huff.gistools import overlay_difference, distance_matrix, buffers


class CustomerOrigins:

    def __init__(
        self,
        geodata_gpd, 
        geodata_gpd_original, 
        metadata,
        isochrones_gdf,
        buffers_gdf
        ):

        self.geodata_gpd = geodata_gpd
        self.geodata_gpd_original = geodata_gpd_original
        self.metadata = metadata
        self.isochrones_gdf = isochrones_gdf
        self.buffers_gdf = buffers_gdf

    def get_geodata_gpd(self):

        return self.geodata_gpd

    def get_geodata_gpd_original(self):

        return self.geodata_gpd_original

    def get_metadata(self):
        
        return self.metadata
    
    def get_isochrones_gdf(self):

        return self.isochrones_gdf

    def get_buffers_gdf(self):

        return self.buffers_gdf

    def summary(self):

        metadata = self.metadata

        print("Customer Origins")
        print("No. locations              " + str(metadata["no_points"]))
        
        if metadata["marketsize_col"] is None:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column         not defined")
        else:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column         " + metadata["marketsize_col"])
        
        if metadata["weighting"][0]["func"] is None and metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_TC} weighting    not defined")
        elif metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
            print(f"{config.DEFAULT_NAME_TC} weighting   " + str(round(metadata["weighting"][0]["param"],3)) + " (" + config.PERMITTED_WEIGHTING_FUNCTIONS[metadata["weighting"][0]["func"]]["description"] + ")")
        elif metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            print(f"{config.DEFAULT_NAME_TC} weighting   " + str(round(metadata["weighting"][0]["param"][0],3)) + ", " + str(round(metadata["weighting"][0]["param"][1],3)) + " (" + config.PERMITTED_WEIGHTING_FUNCTIONS[metadata["weighting"][0]["func"]]["description"] + ")")

        print("Unique ID column           " + metadata["unique_id"])
        print("Input CRS                  " + str(metadata["crs_input"]))

        if self.isochrones_gdf is None:
            print("Including isochrones       NO")
        else:
            print("Including isochrones       YES")

        if self.buffers_gdf is None:
            print("Including buffers          NO")
        else:
            print("Including buffers          YES")

        return metadata
    
    def define_marketsize(
        self,
        marketsize_col: str,
        verbose: bool = config.VERBOSE
        ):

        geodata_gpd_original = self.geodata_gpd_original
        metadata = self.metadata

        if marketsize_col not in geodata_gpd_original.columns:
            raise KeyError(f"Error while defining market size variable: Column {marketsize_col} not in data")
        else:
            metadata["marketsize_col"] = marketsize_col

        self.metadata = metadata

        if verbose:
            print(f"Set market size variable to column {marketsize_col}")

        return self

    def define_transportcosts_weighting(
        self,
        func = "power",
        param_lambda = -2,
        verbose: bool = config.VERBOSE
        ):

        """
        metadata["weighting"] = {
            0: {
                "name": "t_ij",
                "func": "power",
                "param": -2
            }
        }
        
        metadata["weighting"] = {
            0: {
                "name": "t_ij",
                "func": "logistic",
                "param": [10, -0.5]
            }
        }
        """
                
        metadata = self.metadata

        if func not in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
            raise ValueError(f"Error while defining transport costs weighting: Parameter 'func' was set to {func}. Permitted weighting functions are: {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")

        if not isinstance(param_lambda, config.PERMITTED_WEIGHTING_FUNCTIONS[func]["type"]):
            raise TypeError(f"Error while defining transport costs weighting: Function type {func} requires {config.PERMITTED_WEIGHTING_FUNCTIONS[func]['no_params']} parameter(s) in a {config.PERMITTED_WEIGHTING_FUNCTIONS[func]['type']}")
        
        metadata["weighting"][0]["name"] = config.DEFAULT_COLNAME_TC
        metadata["weighting"][0]["func"] = func

        if isinstance(param_lambda, list):
            metadata["weighting"][0]["param"] = [float(param_lambda[0]), float(param_lambda[1])]
        else:
            metadata["weighting"][0]["param"] = float(param_lambda)

        self.metadata = metadata
        
        if verbose:
            print(f"Set transport costs weighting to {config.PERMITTED_WEIGHTING_FUNCTIONS[func]['description']} with parameter(s) {param_lambda}")

        return self

    def isochrones(
        self,
        segments: list = [5, 10, 15],
        range_type: str = "time",
        intersections: str = "true",
        profile: str = "driving-car",
        donut: bool = True,
        ors_server: str = "https://api.openrouteservice.org/v2/",
        ors_auth: str = None,
        timeout: int = 10,
        delay: int = 1,
        save_output: bool = True,
        output_filepath: str = "customer_origins_isochrones.shp",
        output_crs: str = "EPSG:4326",
        verbose: bool = config.VERBOSE
        ):

        geodata_gpd = self.get_geodata_gpd()
        metadata = self.get_metadata()

        isochrones_gdf = get_isochrones( 
            geodata_gpd = geodata_gpd,
            unique_id_col = metadata["unique_id"],
            segments = segments,
            range_type = range_type,
            intersections = intersections,
            profile = profile,
            donut = donut,
            ors_server = ors_server,
            ors_auth = ors_auth,
            timeout = timeout,
            delay = delay,
            save_output = save_output,
            output_filepath = output_filepath,
            output_crs = output_crs,
            verbose = verbose
            )

        self.isochrones_gdf = isochrones_gdf

        return self

    def buffers(
        self,
        segments_distance: list = [500, 1000],
        donut: bool = True,
        save_output: bool = True,
        output_filepath: str = "customer_origins_buffers.shp",
        output_crs: str = "EPSG:4326",
        verbose: bool = config.VERBOSE
        ):

        geodata_gpd_original = self.get_geodata_gpd_original()
        metadata = self.metadata

        buffers_gdf = buffers(
            point_gdf = geodata_gpd_original,
            unique_id_col = metadata["unique_id"],
            distances = segments_distance,
            donut = donut,
            save_output = save_output,
            output_filepath = output_filepath,
            output_crs = output_crs,
            verbose = verbose
            )
        
        self.buffers_gdf = buffers_gdf

        return self

class SupplyLocations:

    def __init__(
        self,
        geodata_gpd, 
        geodata_gpd_original, 
        metadata,
        isochrones_gdf,
        buffers_gdf
        ):

        self.geodata_gpd = geodata_gpd
        self.geodata_gpd_original = geodata_gpd_original
        self.metadata = metadata
        self.isochrones_gdf = isochrones_gdf
        self.buffers_gdf = buffers_gdf

    def get_geodata_gpd(self):

        return self.geodata_gpd

    def get_geodata_gpd_original(self):

        return self.geodata_gpd_original

    def get_metadata(self):

        return self.metadata
    
    def get_isochrones_gdf(self):

        return self.isochrones_gdf
    
    def get_buffers_gdf(self):

        return self.buffers_gdf

    def summary(self):

        metadata = self.metadata

        print("Supply Locations")
        print("No. locations         " + str(metadata["no_points"]))

        if metadata["attraction_col"][0] is None or metadata["attraction_col"] == []:
            print(f"{config.DEFAULT_NAME_ATTRAC} column(s)  not defined")
        else:
            print(f"{config.DEFAULT_NAME_ATTRAC} column(s)  " + ",".join(metadata["attraction_col"]))
        
        if metadata["weighting"][0]["func"] is None and metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_ATTRAC} weighting  not defined")
        else:
            print(f"{config.DEFAULT_NAME_ATTRAC} weighting  " + config.PERMITTED_WEIGHTING_FUNCTIONS[metadata["weighting"][0]["func"]]["description"] + " with gamma = " + str(round(metadata["weighting"][0]["param"],3)))
        
        print("Unique ID column      " + metadata["unique_id"])
        print("Input CRS             " + str(metadata["crs_input"]))

        if self.isochrones_gdf is None:
            print("Including isochrones  NO")
        else:
            print("Including isochrones  YES")

        return metadata

    def define_attraction(
        self,
        attraction_col: str,
        verbose: bool = config.VERBOSE
        ):

        geodata_gpd_original = self.geodata_gpd_original
        metadata = self.metadata

        if attraction_col not in geodata_gpd_original.columns:
            raise KeyError (f"Error while defining attraction variable: Column {attraction_col} not in data")
        else:
            metadata["attraction_col"][0] = attraction_col

        self.metadata = metadata

        if verbose:
            print(f"Set attraction variable to column {attraction_col}")

        return self
    
    def define_attraction_weighting(
        self,
        func = "power",
        param_gamma = 1
        ):

        metadata = self.metadata

        if metadata["attraction_col"] is None:
            raise ValueError(f"Error while defining attraction weighting: {config.DEFAULT_NAME_ATTRAC} column is not yet defined. Use SupplyLocations.define_attraction()")
        
        if func not in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
            raise ValueError(f"Error while defining attraction weighting: Parameter 'func' was set to {func}. Permitted weighting functions are: {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")
        
        metadata["weighting"][0]["name"] = config.DEFAULT_COLNAME_ATTRAC
        metadata["weighting"][0]["func"] = func
        metadata["weighting"][0]["param"] = float(param_gamma)
        
        self.metadata = metadata

        return self

    def add_var(
        self,
        var: str = None,
        func: str = None,
        param: float = None
        ):

        metadata = self.metadata

        if metadata["attraction_col"] is None:
            raise ValueError (f"Error while adding utility variable: {config.DEFAULT_NAME_ATTRAC} column is not yet defined. Use SupplyLocations.define_attraction()")

        no_attraction_vars = len(metadata["attraction_col"])
        new_key = no_attraction_vars

        metadata["attraction_col"] = metadata["attraction_col"] + [var] 

        metadata["weighting"][new_key] = {
            "name": var,
            "func": func,
            "param": param
            }

        self.metadata = metadata

        return self
    
    def change_attraction_values(
        self,
        new_attraction_values: dict
        ):

        """
        new_attraction_values: dict = {
            0: {
                "location": "1",
                "attraction_col": "A_j",
                "new_value": 2700
            }
        }             
        """

        metadata = self.metadata
        unique_id = metadata["unique_id"]

        geodata_gpd_original = self.get_geodata_gpd_original()

        if len(new_attraction_values) > 0:

            for key, entry in new_attraction_values.items():

                if entry["attraction_col"] not in geodata_gpd_original.columns:
                    raise KeyError(f"Supply locations data does not contain attraction column {entry['attraction_col']}")
                if len(entry) < 3:
                    raise KeyError(f"New data entry {key} for supply locations is not complete")
                if "location" not in entry or entry["location"] is None:
                    raise KeyError(f"No 'location' key in new data entry {key}")
                if "attraction_col" not in entry or entry["attraction_col"] is None:
                    raise KeyError(f"No 'attraction_col' key in new data entry {key}")
                if "new_value" not in entry or entry["new_value"] is None:
                    raise KeyError(f"No 'new_value' key in new data entry {key}")

                geodata_gpd_original.loc[geodata_gpd_original[unique_id].astype(str) == str(entry["location"]), entry["attraction_col"]] = entry["new_value"]

        self.geodata_gpd_original = geodata_gpd_original

        return self

    def add_new_destinations(
        self,
        new_destinations,
        ):
        
        geodata_gpd_original = self.get_geodata_gpd_original()
        geodata_gpd = self.get_geodata_gpd()
        metadata = self.get_metadata()

        new_destinations_gpd_original = new_destinations.get_geodata_gpd_original()
        new_destinations_gpd_original[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] = 1
        
        new_destinations_gpd = new_destinations.get_geodata_gpd()
        new_destinations_gpd[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] = 1
        
        new_destinations_metadata = new_destinations.get_metadata()

        if list(new_destinations_gpd_original.columns) != list(geodata_gpd_original.columns):
            raise KeyError("Error while adding new destinations: Supply locations and new destinations data have different column names.")
        if list(new_destinations_gpd.columns) != list(geodata_gpd.columns):
            raise KeyError("Error while adding new destinations: Supply locations and new destinations data have different column names.")

        geodata_gpd_original = pd.concat(
            [
                geodata_gpd_original, 
                new_destinations_gpd_original
                ], 
            ignore_index=True
            )
                
        geodata_gpd = pd.concat(
            [
                geodata_gpd, 
                new_destinations_gpd
                ], 
                ignore_index=True
            )
        
        metadata["no_points"] = metadata["no_points"]+new_destinations_metadata["no_points"]
        
        self.geodata_gpd = geodata_gpd
        self.geodata_gpd_original = geodata_gpd_original
        self.metadata = metadata

        return self
    
    def isochrones(
        self,
        segments: list = [5, 10, 15],
        range_type: str = "time",
        intersections: str = "true",
        profile: str = "driving-car",
        donut: bool = True,
        ors_server: str = "https://api.openrouteservice.org/v2/",
        ors_auth: str = None,
        timeout: int = 10,
        delay: int = 1,
        save_output: bool = True,
        output_filepath: str = "supply_locations_isochrones.shp",
        output_crs: str = "EPSG:4326",
        verbose: bool = config.VERBOSE
        ):

        geodata_gpd = self.get_geodata_gpd()
        metadata = self.get_metadata()

        isochrones_gdf = get_isochrones( 
            geodata_gpd = geodata_gpd,
            unique_id_col = metadata["unique_id"],
            segments = segments,
            range_type = range_type,
            intersections = intersections,
            profile = profile,
            donut = donut,
            ors_server = ors_server,
            ors_auth = ors_auth,
            timeout = timeout,
            delay = delay,
            save_output = save_output,
            output_filepath = output_filepath,
            output_crs = output_crs,
            verbose = verbose
            )

        self.isochrones_gdf = isochrones_gdf

        return self

    def buffers(
        self,
        segments_distance: list = [500, 1000],
        donut: bool = True,
        save_output: bool = True,
        output_filepath: str = "supply_locations_buffers.shp",
        output_crs: str = "EPSG:4326"
        ):

        geodata_gpd_original = self.get_geodata_gpd_original()
        metadata = self.metadata

        buffers_gdf = buffers(
            point_gdf = geodata_gpd_original,
            unique_id_col = metadata["unique_id"],
            distances = segments_distance,
            donut = donut,
            save_output = save_output,
            output_filepath = output_filepath,
            output_crs = output_crs
            )
        
        self.buffers_gdf = buffers_gdf

        return self    

class InteractionMatrix:

    def __init__(
        self, 
        interaction_matrix_df,
        customer_origins,
        supply_locations,
        metadata
        ):

        self.interaction_matrix_df = interaction_matrix_df
        self.customer_origins = customer_origins
        self.supply_locations = supply_locations
        self.metadata = metadata

    def get_interaction_matrix_df(self):
        return self.interaction_matrix_df
    
    def get_customer_origins(self):
        return self.customer_origins

    def get_supply_locations(self):
        return self.supply_locations
    
    def get_metadata(self):
        return self.metadata

    def summary(self):

        customer_origins_metadata = self.get_customer_origins().get_metadata()
        supply_locations_metadata = self.get_supply_locations().get_metadata()
        interaction_matrix_metadata = self.get_metadata()

        print("Interaction Matrix")
        print("----------------------------------")
        
        print("Supply locations    " + str(supply_locations_metadata["no_points"]))
        if supply_locations_metadata["attraction_col"][0] is None:
            print(f"{config.DEFAULT_NAME_ATTRAC} column   not defined")
        else:
            print(f"{config.DEFAULT_NAME_ATTRAC} column   " + supply_locations_metadata["attraction_col"][0])
        print("Customer origins    " + str(customer_origins_metadata["no_points"]))
        if customer_origins_metadata["marketsize_col"] is None:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column not defined")
        else:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column  " + customer_origins_metadata["marketsize_col"])

        if interaction_matrix_metadata != {} and "transport_costs" in interaction_matrix_metadata:
            print("----------------------------------")
            if interaction_matrix_metadata["transport_costs"]["network"]:
                print(f"{config.DEFAULT_NAME_TC} type Time")
                print(f"{config.DEFAULT_NAME_TC} unit " + interaction_matrix_metadata["transport_costs"]["time_unit"])
            else:
                print(f"{config.DEFAULT_NAME_TC} type Distance")
                print(f"{config.DEFAULT_NAME_TC} unit " + interaction_matrix_metadata["transport_costs"]["distance_unit"])

        print("----------------------------------")
        print("Partial utilities")
        print("                    Weights")

        if supply_locations_metadata["weighting"][0]["func"] is None and supply_locations_metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_ATTRAC}          not defined")
        else:
            if supply_locations_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_ATTRAC}          " + str(round(supply_locations_metadata["weighting"][0]["param"],3)) + " (" + supply_locations_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_ATTRAC}          NA" + " (" + supply_locations_metadata["weighting"][0]["func"] + ")")   

        if customer_origins_metadata["weighting"][0]["func"] is None and customer_origins_metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_TC}     not defined")
        elif customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
            if customer_origins_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_TC}     " + str(round(customer_origins_metadata["weighting"][0]["param"],3)) + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_TC}     NA" + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
        elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            if customer_origins_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_TC}    " + str(round(customer_origins_metadata["weighting"][0]["param"][0],3)) + ", " + str(round(customer_origins_metadata["weighting"][0]["param"][1],3)) + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_TC}     NA" + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")

        attrac_vars = supply_locations_metadata["attraction_col"]
        attrac_vars_no = len(attrac_vars)
        
        if attrac_vars_no > 1:
                        
            for key, attrac_var in enumerate(attrac_vars):
                
                if key == 0:
                    continue
                
                if key not in supply_locations_metadata["weighting"].keys():
                    
                    print(f"{attrac_vars[key][:16]:16}    not defined")
                    
                else:
                    
                    if supply_locations_metadata["weighting"][key]["func"] is None and supply_locations_metadata["weighting"][key]["param"]:
                        
                        print(f"{attrac_vars[key][:16]:16}    not defined")

                    else:

                        if supply_locations_metadata["weighting"][key]["param"] is not None:

                            name = supply_locations_metadata["weighting"][key]["name"]
                            param = supply_locations_metadata["weighting"][key]["param"]
                            func = supply_locations_metadata["weighting"][key]["func"]
                            
                            print(f"{name[:16]:16}    {round(param, 3)} ({func})")

                        else:

                            print(f"{attrac_vars[key][:16]:16}    NA" + " (" + supply_locations_metadata["weighting"][0]["func"] + ")") 


        print("----------------------------------")

        if interaction_matrix_metadata != {} and "fit" in interaction_matrix_metadata and interaction_matrix_metadata["fit"]["function"] is not None:
            print("Parameter estimation")
            print("Fit function        " + interaction_matrix_metadata["fit"]["function"])
            print("Fit by              " + interaction_matrix_metadata["fit"]["fit_by"])
            if interaction_matrix_metadata["fit"]["function"] == "huff_ml_fit":
                print("Fit method          " + interaction_matrix_metadata["fit"]["method"] + " (Converged: " + str(interaction_matrix_metadata["fit"]["minimize_success"]) + ")")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata
            ]

    def transport_costs(
        self,
        network: bool = True,
        range_type: str = "time",
        time_unit: str = "minutes",
        distance_unit: str = "kilometers",
        ors_server: str = "https://api.openrouteservice.org/v2/",
        ors_auth: str = None,
        save_output: bool = False,
        remove_duplicates: bool = True,
        output_filepath: str = "transport_costs_matrix.csv",
        shp_save_output: bool = False,
        shp_output_filepath: str = "lines.shp",
        shp_output_crs: str = "EPSG:4326",
        verbose: bool = config.VERBOSE
        ):

        if not network and range_type == "time":
            if verbose:
                print ("NOTE: Calculating euclidean distances (network = False). Setting range_type = 'distance'")
            range_type = "distance"
        
        if verbose:
            if remove_duplicates:
                print("Preparing data for transport costs matrix calculation, including removing duplicates", end = " ... ")
            else:
                print("Preparing data for transport costs matrix calculation", end = " ... ")

        interaction_matrix_df = self.get_interaction_matrix_df()
        interaction_matrix_metadata = self.get_metadata()

        customer_origins = self.get_customer_origins()
        
        customer_origins_geodata_gpd = customer_origins.get_geodata_gpd()
        customer_origins_metadata = customer_origins.get_metadata()
        customer_origins_uniqueid = customer_origins_metadata["unique_id"]
        
        if remove_duplicates:
            customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates(subset=customer_origins_uniqueid)
        
        customer_origins_coords = [[point.x, point.y] for point in customer_origins_geodata_gpd.geometry]
        customer_origins_ids = customer_origins_geodata_gpd[customer_origins_uniqueid].tolist()

        supply_locations = self.get_supply_locations()
        
        supply_locations_geodata_gpd = supply_locations.get_geodata_gpd()
        supply_locations_metadata = supply_locations.get_metadata()
        supply_locations_uniqueid = supply_locations_metadata["unique_id"]
        
        if remove_duplicates:
            supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates(subset=supply_locations_uniqueid)
        
        supply_locations_coords = [[point.x, point.y] for point in supply_locations_geodata_gpd.geometry]
        supply_locations_ids = supply_locations_geodata_gpd[supply_locations_uniqueid].tolist()
   
        locations_coords = customer_origins_coords + supply_locations_coords        
        
        customer_origins_index = list(range(len(customer_origins_coords)))
        locations_coords_index = list(range(len(customer_origins_index), len(locations_coords)))

        if verbose:
            print("OK")

        if network:

            ors_client = Client(
                server = ors_server,
                auth = ors_auth
                )
            time_distance_matrix = ors_client.matrix(
                locations = locations_coords,
                save_output = save_output,
                output_filepath = output_filepath, 
                sources = customer_origins_index,
                destinations = locations_coords_index,
                range_type = range_type,
                verbose = verbose
                )
            
            if time_distance_matrix.get_metadata() is None:
                raise ValueError ("Error in transport costs calculation: No transport costs matrix was built. Probably ORS server error. Check output above and try again later.")

            transport_costs_matrix = time_distance_matrix.get_matrix()
            transport_costs_matrix_config = time_distance_matrix.get_config()
            range_type = transport_costs_matrix_config[config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]]

            transport_costs_matrix[config.MATRIX_COL_SOURCE] = transport_costs_matrix[config.MATRIX_COL_SOURCE].astype(int)
            transport_costs_matrix[config.MATRIX_COL_SOURCE] = transport_costs_matrix[config.MATRIX_COL_SOURCE].map(dict(enumerate(customer_origins_ids)))
            
            transport_costs_matrix[config.MATRIX_COL_DESTINATION] = transport_costs_matrix[config.MATRIX_COL_DESTINATION].astype(int)
            transport_costs_matrix[config.MATRIX_COL_DESTINATION] = transport_costs_matrix[config.MATRIX_COL_DESTINATION].map(dict(enumerate(supply_locations_ids)))
            
            transport_costs_matrix[f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}"] = transport_costs_matrix[config.MATRIX_COL_SOURCE].astype(str)+config.MATRIX_OD_SEPARATOR+transport_costs_matrix[config.MATRIX_COL_DESTINATION].astype(str)
            transport_costs_matrix = transport_costs_matrix[[f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}", range_type]]

            interaction_matrix_df = interaction_matrix_df.merge(
                transport_costs_matrix,
                left_on=config.DEFAULT_COLNAME_INTERACTION,
                right_on=f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}"
                )
            
            interaction_matrix_df[config.DEFAULT_COLNAME_TC] = interaction_matrix_df[range_type]
            if time_unit == "minutes":
                interaction_matrix_df[config.DEFAULT_COLNAME_TC] = interaction_matrix_df[config.DEFAULT_COLNAME_TC]/60
            if time_unit == "hours":
                interaction_matrix_df[config.DEFAULT_COLNAME_TC] = interaction_matrix_df[config.DEFAULT_COLNAME_TC]/60/60

            interaction_matrix_df = interaction_matrix_df.drop(columns=[f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}", range_type])

        else:

            distance_matrix_result = distance_matrix(
                sources = customer_origins_coords,
                destinations = supply_locations_coords,
                sources_uid = customer_origins_ids,
                destinations_uid = supply_locations_ids,
                unit = "m",                
                save_output = shp_save_output,
                output_filepath = shp_output_filepath,
                output_crs = shp_output_crs,
                verbose = verbose
                )
            
            distance_matrix_result_flat = [distance for sublist in distance_matrix_result[0] for distance in sublist]
            
            interaction_matrix_df[config.DEFAULT_COLNAME_TC] = distance_matrix_result_flat

            if distance_unit == "kilometers":
                interaction_matrix_df[config.DEFAULT_COLNAME_TC] = interaction_matrix_df[config.DEFAULT_COLNAME_TC]/1000

        interaction_matrix_metadata["transport_costs"] = {
            "network": network,
            config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]: range_type,
            "time_unit": time_unit,
            "distance_unit": distance_unit,
            "ors_server": ors_server,
            "ors_auth": ors_auth
            }

        self.interaction_matrix_df = interaction_matrix_df
        self.metadata = interaction_matrix_metadata

        return self
    
    def define_weightings(
        self,
        vars_funcs: dict
        ):
        
        """
        vars_funcs = {
            0: {
                "name": "A_j",
                "func": "power",
                "param": 1
            },
            1: {
                "name": "t_ij",
                "func": "logistic"
            },
            2: {
                "name": "second_attraction_variable",
                "func": "power"
            },
            3: {
                "name": "third_attraction_variable",
                "func": "exponential"
            },
            ...
        }
        """

        supply_locations_metadata = self.supply_locations.metadata
        customer_origins_metadata = self.customer_origins.metadata

        supply_locations_metadata["weighting"][0]["name"] = vars_funcs[0]["name"]
        supply_locations_metadata["weighting"][0]["func"] = vars_funcs[0]["func"]
        if "param" in vars_funcs[0]:
            supply_locations_metadata["weighting"][0]["param"] = vars_funcs[0]["param"]
    
        customer_origins_metadata["weighting"][0]["name"] = vars_funcs[1]["name"]
        customer_origins_metadata["weighting"][0]["func"] = vars_funcs[1]["func"]
        if "param" in vars_funcs[1]:
            customer_origins_metadata["weighting"][0]["param"] = vars_funcs[1]["param"]

        if len(vars_funcs) > 2:
            
            for key, var in vars_funcs.items():

                if key < 2:
                    continue
                
                if key not in supply_locations_metadata["weighting"]:
                    supply_locations_metadata["weighting"][key-1] = {
                        "name": "attrac"+str(key),
                        "func": config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0],
                        "param": None
                        }

                supply_locations_metadata["weighting"][key-1]["name"] = var["name"]
                supply_locations_metadata["weighting"][key-1]["func"] = var["func"]

                if "param" in var:
                    supply_locations_metadata["weighting"][key-1]["param"] = var["param"]

        self.supply_locations.metadata = supply_locations_metadata
        self.customer_origins.metadata = customer_origins_metadata

    def utility(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        if verbose:
            print("Calculating utility", end = " ... ")
        
        interaction_matrix_df = self.interaction_matrix_df

        interaction_matrix_metadata = self.get_metadata()

        if config.DEFAULT_COLNAME_TC not in interaction_matrix_df.columns:
            raise KeyError("Error in utility calculation: No transport cost variable in interaction matrix")
        if config.DEFAULT_COLNAME_ATTRAC not in interaction_matrix_df.columns:
            raise KeyError("Error in utility calculation: No attraction variable in interaction matrix")
        if interaction_matrix_df[config.DEFAULT_COLNAME_TC].isna().all():
            raise ValueError(f"Error in utility calculation: {config.DEFAULT_NAME_TC} variable is not defined")
        if interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC].isna().all():
            raise ValueError(f"Error in utility calculation: {config.DEFAULT_NAME_ATTRAC} variable is not defined")

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC]
                )
        
        customer_origins = self.customer_origins
        customer_origins_metadata = customer_origins.get_metadata()
        tc_weighting = customer_origins_metadata["weighting"][0]
       
        if tc_weighting["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = weighting(
                values = interaction_matrix_df[config.DEFAULT_COLNAME_TC],
                func = tc_weighting["func"],
                b = tc_weighting["param"][0],
                c = tc_weighting["param"][1]
                )
        else:
            interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = weighting(
                values = interaction_matrix_df[config.DEFAULT_COLNAME_TC],
                func = tc_weighting["func"],
                b = tc_weighting["param"]
                )      
                           
        supply_locations = self.supply_locations
        supply_locations_metadata = supply_locations.get_metadata()
        attraction_weighting = supply_locations_metadata["weighting"][0]        
        
        interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED] = weighting(
            values = interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC],
            func = attraction_weighting["func"],
            b = attraction_weighting["param"]
            ) 
        
        attrac_vars = supply_locations_metadata["attraction_col"]
        attrac_vars_no = len(attrac_vars)
        attrac_var_key = 0

        if attrac_vars_no > 1:
            
            for key, attrac_var in enumerate(attrac_vars):
                
                attrac_var_key = key
                if attrac_var_key == 0:
                    continue
                
                name = supply_locations_metadata["weighting"][attrac_var_key]["name"]
                param = supply_locations_metadata["weighting"][attrac_var_key]["param"]
                func = supply_locations_metadata["weighting"][attrac_var_key]["func"]

                interaction_matrix_df[name+config.DEFAULT_WEIGHTED_SUFFIX] = weighting(
                    values = interaction_matrix_df[name],
                    func = func,
                    b = param
                    )
                
                interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED] = interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED]*interaction_matrix_df[name+config.DEFAULT_WEIGHTED_SUFFIX]

                interaction_matrix_df = interaction_matrix_df.drop(columns=[name+config.DEFAULT_WEIGHTED_SUFFIX])

        interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY] = interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED]*interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED]
        
        interaction_matrix_df = interaction_matrix_df.drop(columns=[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED, config.DEFAULT_COLNAME_TC_WEIGHTED])

        interaction_matrix_metadata["model"] = {
            "model_type": config.MODELS_LIST[0]
            }

        self.interaction_matrix_df = interaction_matrix_df
        self.metadata = interaction_matrix_metadata
        
        if verbose:
            print("OK")

        return self
    
    def probabilities (
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE        
        ):

        interaction_matrix_df = self.interaction_matrix_df

        if config.DEFAULT_COLNAME_UTILITY not in interaction_matrix_df.columns or interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY].isna().all():
            self.utility()
            interaction_matrix_df = self.interaction_matrix_df

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_UTILITY]
                )

        if verbose:
            print("Calculating probabilities", end = " ... ")

        utility_i = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_CUSTOMER_ORIGINS)[config.DEFAULT_COLNAME_UTILITY].sum())
        utility_i = utility_i.rename(columns = {config.DEFAULT_COLNAME_UTILITY: config.DEFAULT_COLNAME_UTILITY_SUM})

        interaction_matrix_df = interaction_matrix_df.merge(
            utility_i,
            left_on=config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            right_on=config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            how="inner"
            )

        interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] = (interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY]) / (interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY_SUM])

        interaction_matrix_df = interaction_matrix_df.drop(columns=[config.DEFAULT_COLNAME_UTILITY_SUM])

        self.interaction_matrix_df = interaction_matrix_df
        
        if verbose:
            print("OK")

        return self
        
    def flows(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix_df = self.interaction_matrix_df

        if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
            raise KeyError("Error in flows calculation: No market size variable in interaction matrix")
        if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
            raise ValueError(f"Error in flows calculation: {config.DEFAULT_NAME_MARKETSIZE} column in customer origins not defined. Use CustomerOrigins.define_marketsize()")

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_MARKETSIZE]
                )

        if interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY].isna().all():
            self.probabilities()
            interaction_matrix_df = self.interaction_matrix_df

        if verbose:
            print("Calculating flows", end = " ... ")

        interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS] = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] * interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE]

        self.interaction_matrix_df = interaction_matrix_df
        
        if verbose:
            print("OK")

        return self

    def marketareas(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix_df = self.interaction_matrix_df
        
        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_FLOWS],
                check_zero = False
                )
        
        if verbose:
            print("Calculating total market areas", end = " ... ")
        
        market_areas_df = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_SUPPLY_LOCATIONS)[config.DEFAULT_COLNAME_FLOWS].sum())
        market_areas_df = market_areas_df.reset_index(drop=False)
        market_areas_df = market_areas_df.rename(columns={config.DEFAULT_COLNAME_FLOWS: config.DEFAULT_COLNAME_TOTAL_MARKETAREA})

        huff_model = HuffModel(
            self,
            market_areas_df
            )
        
        if verbose:
            print("OK")
        
        return huff_model

    def hansen(
        self,
        from_origins: bool = True,
        exclude_self: bool = True,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix_df = self.interaction_matrix_df

        if exclude_self:

            interaction_matrix_df = interaction_matrix_df[interaction_matrix_df[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS] != interaction_matrix_df[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS]]

        if from_origins:

            if config.DEFAULT_COLNAME_UTILITY not in interaction_matrix_df.columns or interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY].isna().all():
                self.utility(check_df_vars = check_df_vars)
                interaction_matrix_df = self.interaction_matrix_df
                
            if verbose:
                print(f"Calculating {config.MODELS['Hansen']['description']} from origins", end = " ... ")

            hansen_df = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_CUSTOMER_ORIGINS)[config.DEFAULT_COLNAME_UTILITY].sum()).reset_index()
            hansen_df = hansen_df.rename(columns = {config.DEFAULT_COLNAME_UTILITY: "A_i"})

        else:
            
            if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
                raise KeyError(f"Error in {config.MODELS['Hansen']['description']} calculation: Interaction matrix does not contain market size variable")
            if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
                raise ValueError(f"Error in {config.MODELS['Hansen']['description']} calculation: Customer origins market size is not available")
            
            if verbose:
                print(f"Calculating {config.MODELS['Hansen']['description']} from destinations", end = " ... ")
                       
            customer_origins_metadata = self.customer_origins.get_metadata()
            tc_weighting = customer_origins_metadata["weighting"][0]
            if tc_weighting["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0]:
                interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = interaction_matrix_df[config.DEFAULT_COLNAME_TC] ** tc_weighting["param"]
            elif tc_weighting["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]:
                interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = np.exp(tc_weighting["param"] * interaction_matrix_df[config.DEFAULT_COLNAME_TC])
            elif tc_weighting["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
                interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = 1+np.exp(tc_weighting["param"][0] + tc_weighting["param"][1] * interaction_matrix_df[config.DEFAULT_COLNAME_TC])
            else:
                raise ValueError (f"Error in {config.MODELS['Hansen']['description']} calculation: {config.DEFAULT_NAME_TC} weighting is not defined.")
                        
            interaction_matrix_df["U_ji"] = interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE]*interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED]           
            hansen_df = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_SUPPLY_LOCATIONS)["U_ji"].sum()).reset_index()
            hansen_df = hansen_df.rename(columns = {"U_ji": config.DEFAULT_COLNAME_ATTRAC})

        if verbose:
            print("OK")

        return hansen_df

    def mci_transformation(
        self,
        cols: list = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC],
        verbose: bool = config.VERBOSE
        ):

        if verbose:
            print(f"Processing log-centering transformation for {', '.join(cols)}", end = " ... ")

        cols = cols + [config.DEFAULT_COLNAME_PROBABILITY]

        interaction_matrix_df = self.interaction_matrix_df

        interaction_matrix_df = log_centering_transformation(
            df = interaction_matrix_df,
            ref_col = config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            cols = cols
            )
        
        self.interaction_matrix_df = interaction_matrix_df

        if verbose:
            print("OK")
            
        return self

    def mci_fit(
        self,
        cols: list = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC],
        alpha = 0.05,
        verbose: bool = config.VERBOSE
        ):

        supply_locations = self.get_supply_locations()
        supply_locations_metadata = supply_locations.get_metadata()

        customer_origins = self.get_customer_origins()
        customer_origins_metadata = customer_origins.get_metadata()

        interaction_matrix_df = self.get_interaction_matrix_df()

        interaction_matrix_metadata = self.get_metadata()

        cols_t = [col + config.DEFAULT_LCT_SUFFIX for col in cols]

        if f"{config.DEFAULT_COLNAME_PROBABILITY}{config.DEFAULT_LCT_SUFFIX}" not in interaction_matrix_df.columns:
            interaction_matrix = self.mci_transformation(
                cols = cols
                )
            interaction_matrix_df = self.get_interaction_matrix_df()

        if verbose:
            print(f"Performing {config.MODELS['MCI']['description']} estimation", end = " ... ")
                
        mci_formula = f'{config.DEFAULT_COLNAME_PROBABILITY}{config.DEFAULT_LCT_SUFFIX} ~ {" + ".join(cols_t)} -1'

        mci_ols_model = ols(mci_formula, data = interaction_matrix_df).fit()

        mci_ols_coefficients = mci_ols_model.params
        mci_ols_coef_standarderrors = mci_ols_model.bse
        mci_ols_coef_t = mci_ols_model.tvalues
        mci_ols_coef_p = mci_ols_model.pvalues
        mci_ols_coef_ci = mci_ols_model.conf_int(alpha = alpha)

        coefs = {}
        for i, col in enumerate(cols_t):
            coefs[i] = {
                "Coefficient": col[:-5],
                "Estimate": float(mci_ols_coefficients[col]),
                "SE": float(mci_ols_coef_standarderrors[col]),
                "t": float(mci_ols_coef_t[col]),
                "p": float(mci_ols_coef_p[col]),
                "CI_lower": float(mci_ols_coef_ci.loc[col, 0]),
                "CI_upper": float(mci_ols_coef_ci.loc[col, 1]),
                }

        customer_origins_metadata["weighting"][0] = {
            "func": config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0],
            "param": mci_ols_coefficients[f"{config.DEFAULT_COLNAME_TC}{config.DEFAULT_LCT_SUFFIX}"]
            }

        coefs2 = coefs.copy()
        for key, value in list(coefs2.items()):
            if value["Coefficient"] == config.DEFAULT_COLNAME_TC:
                del coefs2[key]

        for key, value in coefs2.items():
            supply_locations_metadata["weighting"][key] = {
                "func": config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0],
                "param": value["Estimate"]
            }

            supply_locations_metadata["attraction_col"].append(None)
            supply_locations_metadata["attraction_col"][key] = value["Coefficient"]

        customer_origins.metadata = customer_origins_metadata
        supply_locations.metadata = supply_locations_metadata
        
        interaction_matrix_metadata = {
            "fit": {
                "function": "mci_fit",
                "fit_by": "probabilities",
                "method": "OLS"
                }
            }
               
        interaction_matrix = InteractionMatrix(
            interaction_matrix_df,
            customer_origins,
            supply_locations,
            metadata=interaction_matrix_metadata
            )
        
        mci_model = MCIModel(
            interaction_matrix,
            coefs,
            mci_ols_model,
            None
            )
        
        if verbose:
            print("OK")
        
        return mci_model

    def loglik(
        self,
        params,
        fit_by = "probabilities",
        check_df_vars: bool = True,        
        ):
        
        if fit_by not in ["probabilities", "flows"]:
            raise ValueError ("Error in loglik: Parameter 'fit_by' must be 'probabilities' or 'flows'")

        if not isinstance(params, list):
            if isinstance(params, np.ndarray):
                params = params.tolist()
            else:
                raise TypeError("Error in loglik: Parameter 'params' must be a list or np.ndarray with at least 2 parameter values")
        
        if len(params) < 2:
            raise ValueError("Error in loglik: Parameter 'params' must be a list or np.ndarray with at least 2 parameter values")
        
        customer_origins = self.customer_origins
        customer_origins_metadata = customer_origins.get_metadata()
        
        param_gamma, param_lambda = params[0], params[1]
            
        if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            
            if len(params) < 3:
                raise ValueError("Error in loglik: When using logistic weighting, parameter 'params' must be a list or np.ndarray with at least 3 parameter values")
        
            param_gamma, param_lambda, param_lambda2 = params[0], params[1], params[2]

        interaction_matrix_df = self.interaction_matrix_df
        
        supply_locations = self.supply_locations
        supply_locations_metadata = supply_locations.get_metadata()        

        supply_locations_metadata["weighting"][0]["param"] = float(param_gamma)
        supply_locations.metadata = supply_locations_metadata

        if customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
            
            if len(params) >= 2:

                customer_origins_metadata["weighting"][0]["param"] = float(param_lambda)
            
            else:
                
                raise ValueError (f"Error in loglik: Huff Model with transport cost weighting of type {customer_origins_metadata['weighting'][0]['func']} must have >= 2 input parameters")
        
        elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            
            if len(params) >= 3:
                
                customer_origins_metadata["weighting"][0]["param"] = [float(param_lambda), float(param_lambda2)]
            
            else:

                raise ValueError(f"Error in loglik: Huff Model with transport cost weighting of type {customer_origins_metadata['weighting'][0]['func']} must have >= 3 input parameters")

        if (customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]] and len(params) > 2): 
            
            for key, param in enumerate(params):

                if key <= 1:
                    continue

                supply_locations_metadata["weighting"][key-1]["param"] = float(param)

        if (customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2] and len(params) > 3):

            for key, param in enumerate(params):

                if key <= 2:
                    continue

                supply_locations_metadata["weighting"][key-2]["param"] = float(param)

        customer_origins.metadata = customer_origins_metadata

        if config.DEFAULT_COLNAME_PROBABILITY_OBSERVED not in interaction_matrix_df.columns:
            p_ij_emp = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
        else:
            p_ij_emp = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED]

        if config.DEFAULT_COLNAME_FLOWS_OBSERVED not in interaction_matrix_df.columns:
            E_ij_emp = interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS]         
        else:
            E_ij_emp = interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS_OBSERVED]
        
        interaction_matrix_copy = copy.deepcopy(self)

        interaction_matrix_copy.utility(
            check_df_vars = check_df_vars,
            verbose=False
            )
        interaction_matrix_copy.probabilities(
            check_df_vars = check_df_vars,
            verbose=False
            )
        interaction_matrix_copy.flows(
            check_df_vars = check_df_vars,
            verbose=False
            )

        interaction_matrix_df_copy = interaction_matrix_copy.get_interaction_matrix_df()
        
        if fit_by == "flows":            
                       
            E_ij = interaction_matrix_df_copy[config.DEFAULT_COLNAME_FLOWS]
        
            observed = E_ij_emp
            expected = E_ij            
            
        else:
        
            p_ij = interaction_matrix_df_copy[config.DEFAULT_COLNAME_PROBABILITY]
            
            observed = p_ij_emp
            expected = p_ij
        
        modelfit_metrics = gof.modelfit(
            observed = observed,
            expected = expected
        )

        LL = modelfit_metrics[1]["LL"]
                       
        return -LL
    
    def huff_ml_fit(
        self,
        initial_params: list = [1.0, -2.0],
        method: str = "L-BFGS-B",
        bounds: list = [(0.5, 1), (-3, -1)],        
        constraints: list = [],
        fit_by = "probabilities",
        update_estimates: bool = True,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):
        
        if verbose:
            print("Performing Maximum Likelihood Estimation", end = " ... ")

        supply_locations = self.supply_locations
        supply_locations_metadata = supply_locations.get_metadata()
        
        customer_origins = self.customer_origins
        customer_origins_metadata = customer_origins.get_metadata()

        if customer_origins_metadata["weighting"][0]["param"] is None:            
            params_metadata_customer_origins = 1            
        else:            
            if customer_origins_metadata["weighting"][0]["param"] is not None:
                if isinstance(customer_origins_metadata["weighting"][0]["param"], (int, float)):
                    params_metadata_customer_origins = 1
                else:
                    params_metadata_customer_origins = len(customer_origins_metadata["weighting"][0]["param"])
            
        if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            params_metadata_customer_origins = 2
        else:
            params_metadata_customer_origins = 1
            
        params_metadata_supply_locations = len(supply_locations_metadata["weighting"])
        
        params_metadata = params_metadata_customer_origins+params_metadata_supply_locations

        if len(initial_params) < 2 or len(initial_params) != params_metadata:
            raise ValueError(f"Error in huff_ml_fit: Parameter 'initial_params' must be a list with {str(params_metadata)} entries ({config.DEFAULT_NAME_ATTRAC}: {str(params_metadata_supply_locations)}, {config.DEFAULT_NAME_TC}: {str(params_metadata_customer_origins)}).")
        
        if len(bounds) != len(initial_params):            
            raise ValueError(f"Error in huff_ml_fit: Parameter 'bounds' must have the same length as parameter 'initial_params' ({str(len(bounds))}, {str(len(initial_params))})")

        ml_result = minimize(
            self.loglik,
            initial_params,
            args=(fit_by, check_df_vars),
            method = method,
            bounds = bounds,
            constraints = constraints,
            options={'disp': 3}
            )      

        attrac_vars = len(supply_locations_metadata["weighting"])

        if ml_result.success:
            
            fitted_params = ml_result.x
            
            param_gamma = fitted_params[0]
            supply_locations_metadata["weighting"][0]["param"] = float(param_gamma)
            
            if customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
                            
                param_lambda = fitted_params[1]                
                param_results = [
                    float(param_gamma), 
                    float(param_lambda)
                    ]
                                
                customer_origins_metadata["weighting"][0]["param"] = float(param_lambda)
                
            elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
        
                param_lambda = fitted_params[1]
                param_lambda2 = fitted_params[2]                
                param_results = [
                    float(param_gamma), 
                    float(param_lambda), 
                    float(param_lambda2)
                    ]
                
                customer_origins_metadata["weighting"][0]["param"][0] = float(param_lambda)
                customer_origins_metadata["weighting"][0]["param"][1] = float(param_lambda2)
                
            if attrac_vars > 1:

                if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
                    fitted_params_add = 3
                else:
                    fitted_params_add = 2 

                for key, var in supply_locations_metadata["weighting"].items():

                    if key > len(supply_locations_metadata["weighting"])-fitted_params_add:
                        break
                    
                    param = float(fitted_params[key+fitted_params_add])
                    
                    param_results = param_results + [param]

                    supply_locations_metadata["weighting"][(key+1)]["param"] = float(param)
                    
            if verbose:
                print("OK")
                print(f"Optimization via {method} algorithm succeeded with parameters: {', '.join(str(round(par, 3)) for par in param_results)}.")
   
        else:
            if verbose:
                print("OK")
            print(f"WARNING: Optimiziation via {method} algorithm failed with error message: '{ml_result.message}'. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for all available algorithms.")

        self.supply_locations.metadata = supply_locations_metadata    
        self.customer_origins.metadata = customer_origins_metadata       
        
        if update_estimates:

            if config.DEFAULT_COLNAME_PROBABILITY_OBSERVED not in self.interaction_matrix_df.columns:
                
                self.interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED] = self.interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
                
                if verbose:
                    print("NOTE: Probabilities in interaction matrix are treated as empirical probabilities")
                
            else:
                
                if verbose:
                    print("NOTE: Interaction matrix contains empirical probabilities")

            if config.DEFAULT_COLNAME_FLOWS_OBSERVED not in self.interaction_matrix_df.columns:
                
                self.interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS_OBSERVED] = self.interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS]
                
                if verbose:
                    print("NOTE: Customer interactions in interaction matrix are treated as empirical interactions")
                
            else:
                
                if verbose:
                    print("NOTE: Interaction matrix contains empirical customer interactions")

            if np.isnan(ml_result.x).any():

                print("WARNING: No update of estimates because fit parameters contain NaN")

                update_estimates = False

            else:

                self = self.utility(verbose=False)
                self = self.probabilities(verbose=False)
                self = self.flows(verbose=False)
        
        self.metadata["fit"] = {
            "function": "huff_ml_fit",
            "fit_by": fit_by,
            "initial_params": initial_params,
            "method": method,
            "bounds": bounds,
            "constraints": constraints,
            "minimize_success": ml_result.success,
            "minimize_fittedparams": ml_result.x,
            "update_estimates": update_estimates
            }
        
        if verbose:
            print("OK")
        
        return self

    def change_attraction_values(
        self,
        new_attraction_values: dict,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix_df = self.interaction_matrix_df

        if len(new_attraction_values) > 0:

            for key, entry in new_attraction_values.items():

                if entry["attraction_col"] not in interaction_matrix_df.columns:
                    raise KeyError(f"Supply locations data does not contain attraction column {entry['attraction_col']}")
                if len(entry) < 3:
                    raise KeyError(f"New data entry {key} for supply locations is not complete")
                if "location" not in entry or entry["location"] is None:
                    raise KeyError(f"No 'location' key in new data entry {key}")
                if "attraction_col" not in entry or entry["attraction_col"] is None:
                    raise KeyError(f"No 'attraction_col' key in new data entry {key}")
                if "new_value" not in entry or entry["new_value"] is None:
                    raise KeyError(f"No 'new_value' key in new data entry {key}")

                interaction_matrix_df.loc[interaction_matrix_df[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS].astype(str) == str(entry["location"]), entry["attraction_col"]] = entry["new_value"]

        self.interaction_matrix_df = interaction_matrix_df
        
        if verbose:
            print(f"Set new attraction values for {len(new_attraction_values)} locations")

        return self

    def update(
        self,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix_df = self.get_interaction_matrix_df()
        
        interaction_matrix_metadata = self.get_metadata()

        customer_origins = self.get_customer_origins()

        supply_locations = self.get_supply_locations()        
     
        supply_locations_geodata_gpd = supply_locations.get_geodata_gpd().copy()
        supply_locations_geodata_gpd_new = supply_locations_geodata_gpd[supply_locations_geodata_gpd[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] == 1]
        
        if len(supply_locations_geodata_gpd_new) < 1:
            raise ValueError("Error in InteractionMatrix update: There are no new destinations for an interaction matrix update. Use SupplyLocations.add_new_destinations()")

        supply_locations_geodata_gpd_original = supply_locations.get_geodata_gpd_original().copy()        
        supply_locations_geodata_gpd_original_new = supply_locations_geodata_gpd_original[supply_locations_geodata_gpd_original[f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_update"] == 1]
        if len(supply_locations_geodata_gpd_original_new) < 1:
            raise ValueError("Error in InteractionMatrix update: There are no new destinations for an interaction matrix update. Use SupplyLocations.add_new_destinations()")

        supply_locations_new = SupplyLocations(
            geodata_gpd=supply_locations_geodata_gpd_new,
            geodata_gpd_original=supply_locations_geodata_gpd_original_new,
            metadata=supply_locations.metadata,
            isochrones_gdf=supply_locations.isochrones_gdf,
            buffers_gdf=supply_locations.buffers_gdf
        )

        interaction_matrix_new = create_interaction_matrix(
            customer_origins=customer_origins,
            supply_locations=supply_locations_new
        )
        
        interaction_matrix_new_df = interaction_matrix_new.get_interaction_matrix_df() 
     
        if "transport_costs" not in interaction_matrix_metadata:
            
            print("WARNING: New destination(s) included. No transport costs calculation because not defined in original interaction matrix.")
            
            interaction_matrix_df = pd.concat(
                [
                interaction_matrix_df, 
                interaction_matrix_new_df
                ], 
                ignore_index=True
                )
            
            interaction_matrix_df = interaction_matrix_df.sort_values(by = config.DEFAULT_COLNAME_INTERACTION)
        
            self.interaction_matrix_df = interaction_matrix_df   
            
        else:
            
            network = interaction_matrix_metadata["transport_costs"]["network"]
            range_type = interaction_matrix_metadata["transport_costs"][config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]]
            time_unit = interaction_matrix_metadata["transport_costs"]["time_unit"]
            distance_unit = interaction_matrix_metadata["transport_costs"]["distance_unit"]
            ors_server = interaction_matrix_metadata["transport_costs"]["ors_server"]
            ors_auth = interaction_matrix_metadata["transport_costs"]["ors_auth"]
            
            interaction_matrix_new.transport_costs(
                network=network,
                range_type=range_type,
                time_unit=time_unit,
                distance_unit=distance_unit,
                ors_server=ors_server,
                ors_auth=ors_auth
            )
            
            interaction_matrix_df = pd.concat(
                [
                    interaction_matrix_df, 
                    interaction_matrix_new_df
                ], 
                ignore_index=True
                )
        
            interaction_matrix_df = interaction_matrix_df.sort_values(by = config.DEFAULT_COLNAME_INTERACTION)
            
            self.interaction_matrix_df = interaction_matrix_df
            
            self.utility(verbose=False)
            self.probabilities(verbose=False)
            self.flows(verbose=False)
            
            if verbose:
                print("Interaction matrix was updated")

        return self

class MarketAreas:

    def __init__(
        self, 
        market_areas_df,        
        metadata
        ):

        self.market_areas_df = market_areas_df
        self.metadata = metadata
   
    def get_market_areas_df(self):
        return self.market_areas_df
    
    def get_metadata(self):
        return self.metadata
    
    def add_to_model(
        self,
        model_object,
        output_model = "Huff",
        verbose: bool = config.VERBOSE
        ):
        
        if not isinstance(model_object, (HuffModel, MCIModel, InteractionMatrix)):
            raise TypeError("Error while adding MarketAreas to model: Parameter 'interaction_matrix' must be of class HuffModel,  MCIModel, or InteractionMatrix")
        
        if isinstance(model_object, MCIModel):
            
            model_object_type = config.MODELS["MCI"]["description"]
                        
            model = MCIModel(
                interaction_matrix = model_object.interaction_matrix,
                coefs = model_object.get_coefs_dict(),
                mci_ols_model = model_object.get_mci_ols_model(),
                market_areas_df = self.market_areas_df
                )            
    
        elif isinstance(model_object, HuffModel):
            
            model_object_type = config.MODELS["Huff"]["description"]
            
            model = HuffModel(
                interaction_matrix = model_object.interaction_matrix,
                market_areas_df = self.market_areas_df
            )

        elif isinstance(model_object, InteractionMatrix):
            
            model_object_type = "Interaction Matrix"

            if output_model not in [config.MODELS_LIST[0], config.MODELS_LIST[1]]:
                raise ValueError("Error while adding MarketAreas to model: Parameter 'output_model' must be either 'Huff' or 'MCI'")
            
            if output_model == config.MODELS_LIST[0]:

                model = HuffModel(
                    interaction_matrix=model_object,
                    market_areas_df=self.market_areas_df
                )

            if output_model == config.MODELS_LIST[1]:

                model = MCIModel(
                    coefs=model_object.coefs,
                    mci_ols_model=model_object.mci_ols_model,
                    market_areas_df=self.market_areas_df
                )

        if verbose:
            print(f"{model_object_type} was added to {output_model} model object")
            
        return model

class HuffModel:

    def __init__(
        self,
        interaction_matrix, 
        market_areas_df
        ):

        self.interaction_matrix = interaction_matrix
        self.market_areas_df = market_areas_df

    def get_interaction_matrix_df(self):

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        return interaction_matrix_df
    
    def get_supply_locations(self):

        interaction_matrix = self.interaction_matrix
        supply_locations = interaction_matrix.get_supply_locations()

        return supply_locations

    def get_customer_origins(self):

        interaction_matrix = self.interaction_matrix
        customer_origins = interaction_matrix.get_customer_origins()

        return customer_origins

    def get_market_areas_df(self):

        return self.market_areas_df
        
    def summary(self):

        interaction_matrix = self.interaction_matrix

        customer_origins_metadata = interaction_matrix.get_customer_origins().get_metadata()
        supply_locations_metadata = interaction_matrix.get_supply_locations().get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()

        print(config.MODELS["Huff"]["description"])
        print("----------------------------------")
        print("Supply locations    " + str(supply_locations_metadata["no_points"]))
        if supply_locations_metadata["attraction_col"][0] is None:
            print(f"{config.DEFAULT_NAME_ATTRAC} column   not defined")
        else:
            print(f"{config.DEFAULT_NAME_ATTRAC} column   " + supply_locations_metadata["attraction_col"][0])
        print("Customer origins    " + str(customer_origins_metadata["no_points"]))
        if customer_origins_metadata["marketsize_col"] is None:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column  not defined")
        else:
            print(f"{config.DEFAULT_NAME_MARKETSIZE} column  " + customer_origins_metadata["marketsize_col"])
        print("----------------------------------")

        print("Partial utilities")
        print("                    Weights")

        if supply_locations_metadata["weighting"][0]["func"] is None and supply_locations_metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_ATTRAC}          not defined")
        else:
            if supply_locations_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_ATTRAC}          " + str(round(supply_locations_metadata["weighting"][0]["param"],3)) + " (" + supply_locations_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_ATTRAC}          NA" + " (" + supply_locations_metadata["weighting"][0]["func"] + ")")   

        if customer_origins_metadata["weighting"][0]["func"] is None and customer_origins_metadata["weighting"][0]["param"] is None:
            print(f"{config.DEFAULT_NAME_TC}     not defined")
        elif customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
            if customer_origins_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_TC}     " + str(round(customer_origins_metadata["weighting"][0]["param"],3)) + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_TC}     NA" + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
        elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            if customer_origins_metadata["weighting"][0]["param"] is not None:
                print(f"{config.DEFAULT_NAME_TC}    " + str(round(customer_origins_metadata["weighting"][0]["param"][0],3)) + ", " + str(round(customer_origins_metadata["weighting"][0]["param"][1],3)) + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")
            else:
                print(f"{config.DEFAULT_NAME_TC}     NA" + " (" + customer_origins_metadata["weighting"][0]["func"] + ")")

        attrac_vars = supply_locations_metadata["attraction_col"]
        attrac_vars_no = len(attrac_vars)
        
        if attrac_vars_no > 1:
                        
            for key, attrac_var in enumerate(attrac_vars):
                
                if key == 0:
                    continue
                
                if key not in supply_locations_metadata["weighting"].keys():
                    
                    print(f"{attrac_vars[key][:16]:16}    not defined")
                    
                else:

                    name = supply_locations_metadata["weighting"][key]["name"]
                    func = supply_locations_metadata["weighting"][key]["func"]

                    if "param" in supply_locations_metadata["weighting"][key]: 
                    
                        param = supply_locations_metadata["weighting"][key]["param"]
                        
                        if param is not None:

                            print(f"{name[:16]:16}    {round(param, 3)} ({func})")

                        else:

                            print(f"{attrac_vars[key][:16]:16}    NA ({func})")
                    
        print("----------------------------------")

        huff_modelfit = None

        if interaction_matrix_metadata != {} and "fit" in interaction_matrix_metadata and interaction_matrix_metadata["fit"]["function"] is not None:
            print("Parameter estimation")
            print("Fit function        " + interaction_matrix_metadata["fit"]["function"])
            print("Fit by              " + interaction_matrix_metadata["fit"]["fit_by"])
            if interaction_matrix_metadata["fit"]["function"] == "huff_ml_fit":
                print("Fit method          " + interaction_matrix_metadata["fit"]["method"] + " (Converged: " + str(interaction_matrix_metadata["fit"]["minimize_success"]) + ")")

            huff_modelfit = self.modelfit(by = interaction_matrix_metadata["fit"]["fit_by"])
            
            if huff_modelfit is not None:
                                
                print (f"Goodness-of-fit for {interaction_matrix_metadata['fit']['fit_by']}")

                gof.modelfit_print(huff_modelfit)
                    
                print("----------------------------------")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            huff_modelfit
            ]

    def mci_fit(
        self,
        cols: list = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC],
        alpha = 0.05,
        verbose: bool = config.VERBOSE
        ):
        
        if verbose:
            print(f"Processing estimation of {config.MODELS['MCI']['description']}", end = " ... ")

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()
        interaction_matrix_metadata = interaction_matrix.get_metadata()       

        supply_locations = interaction_matrix.get_supply_locations()
        supply_locations_metadata = supply_locations.get_metadata()

        customer_origins = interaction_matrix.get_customer_origins()
        customer_origins_metadata = customer_origins.get_metadata()
        
        cols_t = [col + config.DEFAULT_LCT_SUFFIX for col in cols]

        if f"{config.DEFAULT_COLNAME_PROBABILITY}{config.DEFAULT_LCT_SUFFIX}" not in interaction_matrix_df.columns:
            interaction_matrix = interaction_matrix.mci_transformation(
                cols = cols
                )
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        mci_formula = f'{config.DEFAULT_COLNAME_PROBABILITY}{config.DEFAULT_LCT_SUFFIX} ~ {" + ".join(cols_t)} -1'

        mci_ols_model = ols(mci_formula, data = interaction_matrix_df).fit()

        mci_ols_coefficients = mci_ols_model.params
        mci_ols_coef_standarderrors = mci_ols_model.bse
        mci_ols_coef_t = mci_ols_model.tvalues
        mci_ols_coef_p = mci_ols_model.pvalues
        mci_ols_coef_ci = mci_ols_model.conf_int(alpha = alpha)

        coefs = {}
        for i, col in enumerate(cols_t):
            coefs[i] = {
                "Coefficient": col[:-5],
                "Estimate": float(mci_ols_coefficients[col]),
                "SE": float(mci_ols_coef_standarderrors[col]),
                "t": float(mci_ols_coef_t[col]),
                "p": float(mci_ols_coef_p[col]),
                "CI_lower": float(mci_ols_coef_ci.loc[col, 0]),
                "CI_upper": float(mci_ols_coef_ci.loc[col, 1]),
                }

        customer_origins_metadata["weighting"][0] = {
            "func": config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0],
            "param": mci_ols_coefficients[f"{config.DEFAULT_COLNAME_TC}{config.DEFAULT_LCT_SUFFIX}"]
            }

        coefs2 = coefs.copy()
        for key, value in list(coefs2.items()):
            if value["Coefficient"] == config.DEFAULT_COLNAME_TC:
                del coefs2[key]

        for key, value in coefs2.items():
            supply_locations_metadata["weighting"][(key)] = {
                "func": config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0],
                "param": value["Estimate"]
            }
            supply_locations_metadata["attraction_col"][key] = value["Coefficient"]

        customer_origins.metadata = customer_origins_metadata
        supply_locations.metadata = supply_locations_metadata
        
        interaction_matrix_metadata = {
            "fit": {
                "function": "mci_fit",
                "fit_by": "probabilities",
                "method": "OLS"
                }
            }
        
        interaction_matrix = InteractionMatrix(
            interaction_matrix_df,
            customer_origins,
            supply_locations,
            metadata=interaction_matrix_metadata
            )
        
        mci_model = MCIModel(
            interaction_matrix,
            coefs,
            mci_ols_model,
            None
            )
        
        if verbose:
            print("OK")
        
        return mci_model

    def loglik(
        self,
        params,
        check_df_vars: bool = True,
        ):

        if not isinstance(params, list):
            if isinstance(params, np.ndarray):
                params = params.tolist()
            else:
                raise ValueError("Error in loglik: Parameter 'params' must be a list or np.ndarray with at least 2 parameter values")
        
        if len(params) < 2:
            raise ValueError("Error in loglik: Parameter 'params' must be a list or np.ndarray with at least 2 parameter values")
                  
        market_areas_df = self.market_areas_df
        
        customer_origins = self.interaction_matrix.customer_origins
        customer_origins_metadata = customer_origins.get_metadata()
        
        param_gamma, param_lambda = params[0], params[1]
            
        if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            
            if len(params) < 3:
                raise ValueError("Error in loglik: When using logistic weighting, parameter 'params' must be a list or np.ndarray with at least 3 parameter values")
        
            param_gamma, param_lambda, param_lambda2 = params[0], params[1], params[2]

        supply_locations = self.interaction_matrix.supply_locations
        supply_locations_metadata = supply_locations.get_metadata()        

        supply_locations_metadata["weighting"][0]["param"] = float(param_gamma)
        supply_locations.metadata = supply_locations_metadata

        if customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
            
            if len(params) >= 2:

                customer_origins_metadata["weighting"][0]["param"] = float(param_lambda)
            
            else:
                
                raise ValueError (f"Error in loglik: Huff Model with transport cost weighting of type {customer_origins_metadata['weighting'][0]['func']} must have >= 2 input parameters")
        
        elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            
            if len(params) >= 3:
                
                customer_origins_metadata["weighting"][0]["param"] = [float(param_lambda), float(param_lambda2)]
            
            else:

                raise ValueError(f"Error in loglik: Huff Model with transport cost weighting of type {customer_origins_metadata['weighting'][0]['func']} must have >= 3 input parameters")

        if (customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]] and len(params) > 2): 
            
            for key, param in enumerate(params):

                if key <= 1:
                    continue

                supply_locations_metadata["weighting"][key-1]["param"] = float(param)

        if (customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2] and len(params) > 3):

            for key, param in enumerate(params):

                if key <= 2:
                    continue

                supply_locations_metadata["weighting"][key-2]["param"] = float(param)

        customer_origins.metadata = customer_origins_metadata        
       
        if config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED not in market_areas_df.columns:
            T_j_emp = market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]
        else:
            T_j_emp = market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED]


        huff_model_copy = copy.deepcopy(self)

        interaction_matrix_copy = copy.deepcopy(huff_model_copy.interaction_matrix)

        interaction_matrix_copy = interaction_matrix_copy.utility(
            check_df_vars = check_df_vars,
            verbose = False
            )
        interaction_matrix_copy = interaction_matrix_copy.probabilities(
            check_df_vars = check_df_vars,
            verbose = False
            )
        interaction_matrix_copy = interaction_matrix_copy.flows(
            check_df_vars = check_df_vars,
            verbose = False
            )

        huff_model_copy = interaction_matrix_copy.marketareas(verbose = False)

        market_areas_df_copy = huff_model_copy.market_areas_df

        observed = T_j_emp
        expected = market_areas_df_copy[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]
        
        modelfit_metrics = gof.modelfit(
            observed = observed,
            expected = expected
        )

        LL = modelfit_metrics[1]["LL"]
        
        return -LL

    def ml_fit(
        self,
        initial_params: list = [1.0, -2.0],
        method: str = "L-BFGS-B",
        bounds: list = [(0.5, 1), (-3, -1)],        
        constraints: list = [],
        fit_by = "probabilities",
        update_estimates: bool = True,
        check_numbers: bool = True,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):
        
        if fit_by in ["probabilities", "flows"]:

            self.interaction_matrix.huff_ml_fit(
                initial_params = initial_params,
                method = method,
                bounds = bounds,        
                constraints = constraints,
                fit_by = fit_by,
                update_estimates = update_estimates,
                check_df_vars = check_df_vars,
                verbose = False
                )
        
        elif fit_by == "totals":
            
            if check_numbers:
                
                market_areas_df = self.market_areas_df
                interaction_matrix_df = self.get_interaction_matrix_df()
                T_j_market_areas_df = sum(market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA])
                T_j_interaction_matrix_df = sum(interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS])
                
                if T_j_market_areas_df != T_j_interaction_matrix_df:
                    print(f"WARNING: Sum of total market areas ({int(T_j_market_areas_df)}) is not equal to sum of customer flows ({int(T_j_interaction_matrix_df)})")
            
            supply_locations = self.interaction_matrix.supply_locations
            supply_locations_metadata = supply_locations.get_metadata()
        
            customer_origins = self.interaction_matrix.customer_origins
            customer_origins_metadata = customer_origins.get_metadata()

            if customer_origins_metadata["weighting"][0]["param"] is None:            
                params_metadata_customer_origins = 1            
            else:            
                if customer_origins_metadata["weighting"][0]["param"] is not None:
                    if isinstance(customer_origins_metadata["weighting"][0]["param"], (int, float)):
                        params_metadata_customer_origins = 1
                    else:
                        params_metadata_customer_origins = len(customer_origins_metadata["weighting"][0]["param"])
            
            if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
                params_metadata_customer_origins = 2
            else:
                params_metadata_customer_origins = 1
                
            params_metadata_supply_locations = len(supply_locations_metadata["weighting"])
            
            params_metadata = params_metadata_customer_origins+params_metadata_supply_locations

            if len(initial_params) < 2 or len(initial_params) != params_metadata:
                raise ValueError(f"Error in ml_fit: Parameter 'initial_params' must be a list with {str(params_metadata)} entries ({config.DEFAULT_NAME_ATTRAC}: {str(params_metadata_supply_locations)}, {config.DEFAULT_NAME_TC}: {str(params_metadata_customer_origins)})")
            
            if len(bounds) != len(initial_params):            
                raise ValueError(f"Error in ml_fit: Parameter 'bounds' must have the same length as parameter 'initial_params' ({str(len(bounds))}, {str(len(initial_params))})")

            ml_result = minimize(
                self.loglik,
                initial_params,
                args=check_df_vars,
                method = method,
                bounds = bounds,
                constraints = constraints,
                options={'disp': 3}
                )
            
            attrac_vars = len(supply_locations_metadata["weighting"])

            if ml_result.success:
                
                fitted_params = ml_result.x
                
                param_gamma = fitted_params[0]
                supply_locations_metadata["weighting"][0]["param"] = float(param_gamma)
                
                if customer_origins_metadata["weighting"][0]["func"] in [config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0], config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[1]]:
                                
                    param_lambda = fitted_params[1]                
                    param_results = [
                        float(param_gamma), 
                        float(param_lambda)
                        ]
                                    
                    customer_origins_metadata["weighting"][0]["param"] = float(param_lambda)
                    
                elif customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
            
                    param_lambda = fitted_params[1]
                    param_lambda2 = fitted_params[2]                
                    param_results = [
                        float(param_gamma), 
                        float(param_lambda), 
                        float(param_lambda2)
                        ]
                    
                    customer_origins_metadata["weighting"][0]["param"][0] = float(param_lambda)
                    customer_origins_metadata["weighting"][0]["param"][1] = float(param_lambda2)
                    
                if attrac_vars > 1:

                    if customer_origins_metadata["weighting"][0]["func"] == config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[2]:
                        fitted_params_add = 3
                    else:
                        fitted_params_add = 2 

                    for key, var in supply_locations_metadata["weighting"].items():

                        if key > len(supply_locations_metadata["weighting"])-fitted_params_add:
                            break
                        
                        param = float(fitted_params[key+fitted_params_add])
                        
                        param_results = param_results + [param]

                        supply_locations_metadata["weighting"][(key+1)]["param"] = float(param)
                        
                if verbose:
                    print(f"Optimization via {method} algorithm succeeded with parameters: {', '.join(str(round(par, 3)) for par in param_results)}.")
    
            else:
                   
                if verbose:
                    print(f"Optimiziation via {method} algorithm failed with error message: '{ml_result.message}'. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for all available algorithms.")

            self.interaction_matrix.supply_locations.metadata = supply_locations_metadata    
            self.interaction_matrix.customer_origins.metadata = customer_origins_metadata

            if update_estimates:

                if config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED not in self.market_areas_df.columns:

                    self.market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED] = self.market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]
                    
                    if verbose:
                        print("NOTE: Total values in market areas df are treated as empirical total values")
                
                else:

                    if verbose:
                        print("NOTE: Total market areas df contains empirical total values")

                if np.isnan(ml_result.x).any():

                    print("WARNING: No update of estimates because fit parameters contain NaN")

                    update_estimates = False

                else:
                                       
                    self.interaction_matrix.utility(check_df_vars = check_df_vars)
                    self.interaction_matrix.probabilities(check_df_vars = check_df_vars)
                    self.interaction_matrix.flows(check_df_vars = check_df_vars) 
                    
                    huff_model_new_marketareas = self.interaction_matrix.marketareas(check_df_vars = check_df_vars)
                    self.market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA] = huff_model_new_marketareas.get_market_areas_df()[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]
            
            self.interaction_matrix.metadata["fit"] = {
                "function": "huff_ml_fit",
                "fit_by": fit_by,
                "initial_params": initial_params,
                "method": method,
                "bounds": bounds,
                "constraints": constraints,
                "minimize_success": ml_result.success,
                "minimize_fittedparams": ml_result.x,
                "update_estimates": update_estimates
                }
            
        else:

            raise ValueError("Error in ml_fit: Parameter 'fit_by' must be 'probabilities', 'flows' or 'totals'")

        return self
    
    def confint(
        self,
        alpha = 0.05,
        repeats = 3,
        sample_size = 0.75,
        replace = True
        ):
        
        if self.interaction_matrix.metadata["fit"] is None or self.interaction_matrix.metadata["fit"] == {}:
            raise ValueError("Error while estimating confidence intervals: Model object does not contain information towards fit procedure")

        keys_necessary = [
            "function", 
            "fit_by", 
            "initial_params",
            "method",
            "bounds",
            "constraints"            
            ]

        for key_necessary in keys_necessary:
            if key_necessary not in self.interaction_matrix.metadata["fit"]:
                raise KeyError(f"Error while estimating confidence intervals: Model object does not contain full information towards fit procedure. Missing key {key_necessary}")
        
        fitted_params_repeats = []

        alpha_lower = alpha/2
        alpha_upper = 1-alpha/2

        huff_model_copy = copy.deepcopy(self)     

        if self.interaction_matrix.metadata["fit"]["fit_by"] in ["probabilities", "flows"]:
        
            for i in range(repeats):
                
                try:
                
                    n_samples = int(len(huff_model_copy.interaction_matrix.interaction_matrix_df)*sample_size)
                    
                    huff_model_copy.interaction_matrix.interaction_matrix_df = huff_model_copy.interaction_matrix.interaction_matrix_df.sample(
                        n = n_samples, 
                        replace = replace
                        )
                    
                    huff_model_copy.ml_fit(
                        initial_params = self.interaction_matrix.metadata["fit"]["initial_params"],
                        method = self.interaction_matrix.metadata["fit"]["method"],
                        bounds = self.interaction_matrix.metadata["fit"]["bounds"],        
                        constraints = self.interaction_matrix.metadata["fit"]["constraints"],
                        fit_by = self.interaction_matrix.metadata["fit"]["fit_by"],
                        update_estimates = True,
                        check_numbers = True
                    )
                    
                    minimize_fittedparams = huff_model_copy.interaction_matrix.metadata["fit"]["minimize_fittedparams"]
                        
                    fitted_params_repeats.append(minimize_fittedparams)

                except Exception as err:

                    print (f"Error in repeat {str(i)}: {err}")
        
        elif self.metadata["fit"]["fit_by"] == "totals":

            for i in range(repeats):

                n_samples = int(len(huff_model_copy.market_areas_df)*sample_size)

                huff_model_copy.market_areas_df = huff_model_copy.market_areas_df.sample(
                    n = n_samples, 
                    replace = replace
                    )
            
                huff_model_copy.interaction_matrix.interaction_matrix_df = huff_model_copy.interaction_matrix.interaction_matrix_df[
                    huff_model_copy.interaction_matrix.interaction_matrix_df[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS].isin(huff_model_copy.market_areas_df[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS])
                    ]
                
                huff_model_copy.ml_fit(
                    initial_params = self.interaction_matrix.metadata["fit"]["initial_params"],
                    method = self.interaction_matrix.metadata["fit"]["method"],
                    bounds = self.interaction_matrix.metadata["fit"]["bounds"],        
                    constraints = self.interaction_matrix.metadata["fit"]["constraints"],
                    fit_by = self.interaction_matrix.metadata["fit"]["fit_by"],
                    update_estimates = True,
                    check_numbers = True
                )
                
                minimize_fittedparams = huff_model_copy.interaction_matrix.metadata["fit"]["minimize_fittedparams"]
                    
                fitted_params_repeats.append(minimize_fittedparams)

        else:
            
            raise ValueError("Error while estimating confidence intervals: Parameter 'fit_by' must be 'probabilities', 'flows' or 'totals'")
        
        fitted_params_repeats_array = np.array(fitted_params_repeats)
        fitted_params_repeats_array_transposed = fitted_params_repeats_array.T
        
        param_ci = pd.DataFrame(columns=["lower", "upper"])

        for i, col in enumerate(fitted_params_repeats_array_transposed):
            
            param_ci.loc[i, "lower"] = np.quantile(col, alpha_lower)
            param_ci.loc[i, "upper"] = np.quantile(col, alpha_upper)
        
        return param_ci

    def update(self):
        
        self.interaction_matrix = self.interaction_matrix.update()
        
        self.market_areas_df = self.interaction_matrix.marketareas().get_market_areas_df()
        
        return self
    
    def modelfit(
        self,
        by = "probabilities"
        ):       
        
        if by == "probabilities":

            interaction_matrix = self.interaction_matrix        
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

            if (config.DEFAULT_COLNAME_PROBABILITY in interaction_matrix_df.columns and config.DEFAULT_COLNAME_PROBABILITY_OBSERVED in interaction_matrix_df.columns):
                
                try:
                
                    huff_modelfit = gof.modelfit(
                        interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED],
                        interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
                    )
                    
                    return huff_modelfit
                    
                except:
                    
                    print("WARNING: Goodness-of-fit metrics could not be calculated due to NaN values.")
                    return None
            
            else:
                
                print("WARNING: Goodness-of-fit metrics could not be calculated. No empirical values of probabilities in interaction matrix.")

                return None
            
        elif by == "flows":

            interaction_matrix = self.interaction_matrix        
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

            if (config.DEFAULT_COLNAME_FLOWS in interaction_matrix_df.columns and config.DEFAULT_COLNAME_FLOWS_OBSERVED in interaction_matrix_df.columns):
                
                try:
                
                    huff_modelfit = gof.modelfit(
                        interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS_OBSERVED],
                        interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS]
                    )
                    
                    return huff_modelfit
                    
                except:
                    
                    print("WARNING: Goodness-of-fit metrics could not be calculated due to NaN values.")
                    return None
            
            else:
                
                print("WARNING: Goodness-of-fit metrics could not be calculated. No empirical values of customer flows in interaction matrix.")

                return None
            
        elif by == "totals":

            market_areas_df = self.market_areas_df

            if (config.DEFAULT_COLNAME_TOTAL_MARKETAREA in market_areas_df.columns and config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED in market_areas_df.columns):
                
                try:
                
                    huff_modelfit = gof.modelfit(
                        market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA_OBSERVED],
                        market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]
                    )
                    
                    return huff_modelfit
                    
                except:
                    
                    print("WARNING: Goodness-of-fit metrics could not be calculated due to NaN values.")
                    return None
                
            else:
                
                print("WARNING: Goodness-of-fit metrics could not be calculated. No empirical values of T_j in market areas data.")

                return None

        else:

            raise ValueError("Error in HuffModel.modelfit: Parameter 'by' must be 'probabilities', 'flows', or 'totals'")
    
class MCIModel:

    def __init__(
        self,
        interaction_matrix: InteractionMatrix,
        coefs: dict,
        mci_ols_model,
        market_areas_df
        ):

        self.interaction_matrix = interaction_matrix
        self.coefs = coefs
        self.mci_ols_model = mci_ols_model
        self.market_areas_df = market_areas_df

    def get_interaction_matrix_df(self):

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        return interaction_matrix_df
    
    def get_supply_locations(self):

        interaction_matrix = self.interaction_matrix
        supply_locations = interaction_matrix.get_supply_locations()

        return supply_locations

    def get_customer_origins(self):

        interaction_matrix = self.interaction_matrix
        customer_origins = interaction_matrix.get_customer_origins()

        return customer_origins
    
    def get_mci_ols_model(self):

        return self.mci_ols_model
    
    def get_coefs_dict(self):

        return self.coefs
    
    def get_market_areas_df(self):

        return self.market_areas_df

    def modelfit(self):
        
        interaction_matrix = self.interaction_matrix        
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()
        
        if (config.DEFAULT_COLNAME_PROBABILITY in interaction_matrix_df.columns and config.DEFAULT_COLNAME_PROBABILITY_OBSERVED in interaction_matrix_df.columns):
            
            try:
            
                mci_modelfit = gof.modelfit(
                    interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED],
                    interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
                )
                
                return mci_modelfit
                
            except:
                
                print("WARNING: Goodness-of-fit metrics could not be calculated due to NaN values.")
                return None
        
        else:
            
            return None
        
    def summary(self):

        interaction_matrix = self.interaction_matrix
        coefs = self.coefs

        customer_origins_metadata = interaction_matrix.get_customer_origins().get_metadata()
        supply_locations_metadata = interaction_matrix.get_supply_locations().get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()

        print("Multiplicative Competitive Interaction Model")
        print("--------------------------------------------")
        print("Supply locations   " + str(supply_locations_metadata["no_points"]))
        print("Customer origins   " + str(customer_origins_metadata["no_points"]))
        print("--------------------------------------------")
        print("Partial utilities")
 
        coefficients_rows = []
        for key, value in coefs.items():
            coefficient_name = value["Coefficient"]
            if coefficient_name == config.DEFAULT_COLNAME_ATTRAC:
                coefficient_name = config.DEFAULT_NAME_ATTRAC
            if coefficient_name == config.DEFAULT_COLNAME_TC:
                coefficient_name = config.DEFAULT_NAME_TC
            coefficients_rows.append({
                "": coefficient_name,
                "Estimate": round(value["Estimate"], 3),
                "SE": round(value["SE"], 3),
                "t": round(value["t"], 3),
                "p": round(value["p"], 3),
                "CI lower": round(value["CI_lower"], 3),
                "CI upper": round(value["CI_upper"], 3)
            })
        coefficients_df = pd.DataFrame(coefficients_rows)
        
        print (coefficients_df)

        print("--------------------------------------------")
        
        mci_modelfit = None

        mci_modelfit = self.modelfit()

        if mci_modelfit is not None:
            
            print ("Goodness-of-fit for probabilities")

            gof.modelfit_print(mci_modelfit)            
            
            print("--------------------------------------------")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            mci_modelfit
            ]
                  
    def utility(
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        check_df_vars: bool = True
        ):
        
        if transformation not in config.MCI_TRANSFORMATIONS_LIST:
            raise ValueError(f"Parameter 'transformation' must be one of the following: {', '.join(config.MCI_TRANSFORMATIONS_LIST)}")

        interaction_matrix = self.interaction_matrix        
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()
        interaction_matrix_metadata = interaction_matrix.get_metadata()

        if interaction_matrix_df[config.DEFAULT_COLNAME_TC].isna().all():
            raise ValueError(f"Error in utility calculation: {config.DEFAULT_NAME_TC} variable {config.DEFAULT_COLNAME_TC} is not defined")
        if interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC].isna().all():
            raise ValueError(f"Error in utility calculation: {config.DEFAULT_NAME_ATTRAC} variable {config.DEFAULT_COLNAME_ATTRAC} is not defined")

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC]
                )

        customer_origins = interaction_matrix.get_customer_origins()
        customer_origins_metadata = customer_origins.get_metadata()
        
        t_ij_weighting = customer_origins_metadata["weighting"][0]["param"]

        if transformation == "ILCT":
            mci_formula = f"{t_ij_weighting}*{config.DEFAULT_COLNAME_TC}"
        else:
            mci_formula = f"{config.DEFAULT_COLNAME_TC}**{t_ij_weighting}"
        
        supply_locations = interaction_matrix.get_supply_locations()
        supply_locations_metadata = supply_locations.get_metadata()
        attraction_col = supply_locations_metadata["attraction_col"]
        attraction_weighting = supply_locations_metadata["weighting"]

        if transformation == "ILCT":
            for key, value in attraction_weighting.items():
                mci_formula = mci_formula + f" + {value['param']}*{attraction_col[key]}"
        else:
            for key, value in attraction_weighting.items():
                mci_formula = mci_formula + f" * {attraction_col[key]}**{value['param']}"

        interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY] = interaction_matrix_df.apply(lambda row: eval(mci_formula, {}, row.to_dict()), axis=1)

        if transformation == "ILCT":
            interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY] = np.exp(interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY])

        interaction_matrix_metadata["model"] = {
            "model_type": config.MODELS_LIST[1],
            "transformation": transformation
            }

        interaction_matrix = InteractionMatrix(
            interaction_matrix_df,
            customer_origins,
            supply_locations,
            metadata=interaction_matrix_metadata
            )
        self.interaction_matrix = interaction_matrix

        return self
    
    def probabilities (
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        verbose: bool = config.VERBOSE
        ):

        interaction_matrix = self.interaction_matrix        
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()
        
        if config.DEFAULT_COLNAME_PROBABILITY in interaction_matrix_df.columns and config.DEFAULT_COLNAME_PROBABILITY_OBSERVED not in interaction_matrix_df.columns:
            
            if verbose:
                print("NOTE: Probabilities in interaction matrix are treated as empirical probabilities")
            
            interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED] = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
        
        else:
            
            if verbose:
                print("NOTE: Interaction matrix contains empirical probabilities")

        if config.DEFAULT_COLNAME_UTILITY not in interaction_matrix_df.columns:
            self.utility(transformation = transformation)
            interaction_matrix = self.interaction_matrix
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        if interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY].isna().all():
            self.utility(transformation = transformation)
            interaction_matrix = self.interaction_matrix
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        utility_i = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_CUSTOMER_ORIGINS)[config.DEFAULT_COLNAME_UTILITY].sum())
        utility_i = utility_i.rename(columns = {config.DEFAULT_COLNAME_UTILITY: config.DEFAULT_COLNAME_UTILITY_SUM})

        interaction_matrix_df = interaction_matrix_df.merge(
            utility_i,
            left_on=config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            right_on=config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            how="inner"
            )

        interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] = (interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY]) / (interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY_SUM])

        interaction_matrix_df = interaction_matrix_df.drop(columns=[config.DEFAULT_COLNAME_UTILITY_SUM])

        interaction_matrix.interaction_matrix_df = interaction_matrix_df
        self.interaction_matrix = interaction_matrix

        return self
        
    def flows (
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        check_df_vars: bool = True
        ):

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
            raise KeyError ("Error in flows calculation: No market size column defined in interaction matrix.")
        
        if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
            raise ValueError (f"Error in flows calculation: {config.DEFAULT_NAME_MARKETSIZE} column in customer origins not defined. Use CustomerOrigins.define_marketsize()")

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_MARKETSIZE]
                )

        if interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY].isna().all():
            self.probabilities(transformation=transformation)
            interaction_matrix = self.interaction_matrix
            interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS] = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] * interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE]

        self.interaction_matrix_df = interaction_matrix_df

        return self

    def marketareas (
        self,
        check_df_vars: bool = True
        ):

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()
        
        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_FLOWS]
                )
        
        market_areas_df = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_SUPPLY_LOCATIONS)[config.DEFAULT_COLNAME_FLOWS].sum())
        market_areas_df = market_areas_df.reset_index(drop=False)
        market_areas_df = market_areas_df.rename(columns={config.DEFAULT_COLNAME_FLOWS: config.DEFAULT_COLNAME_TOTAL_MARKETAREA})

        mci_model = MCIModel(
            interaction_matrix = interaction_matrix,
            coefs = self.get_coefs_dict(),
            mci_ols_model = self.get_mci_ols_model(),
            market_areas_df = market_areas_df
            )

        return mci_model
  
def create_interaction_matrix(
    customer_origins,
    supply_locations,
    requiring_attributes: bool = True,
    remove_duplicates: bool = True
    ):

    if not isinstance(customer_origins, CustomerOrigins):
        raise ValueError ("Error while creating interaction matrix: customer_origins must be of class CustomerOrigins")
    if not isinstance(supply_locations, SupplyLocations):
        raise ValueError ("Error while creating interaction matrix: supply_locations must be of class SupplyLocations")

    customer_origins_metadata = customer_origins.get_metadata()
    supply_locations_metadata = supply_locations.get_metadata()
    
    customer_origins_unique_id = customer_origins_metadata["unique_id"]
    customer_origins_marketsize = customer_origins_metadata["marketsize_col"]    

    if customer_origins_metadata["marketsize_col"] is None and requiring_attributes:
        raise ValueError(f"Error while creating interaction matrix: {config.DEFAULT_NAME_MARKETSIZE} column in customer origins not defined. Use CustomerOrigins.define_marketsize()")        
            
    if supply_locations_metadata["attraction_col"][0] is None and requiring_attributes:
        raise ValueError(f"Error while creating interaction matrix: {config.DEFAULT_NAME_ATTRAC} column in supply locations not defined. Use SupplyLocations.define_attraction()")

    customer_origins_geodata_gpd = pd.DataFrame(customer_origins.get_geodata_gpd())
    customer_origins_geodata_gpd_original = pd.DataFrame(customer_origins.get_geodata_gpd_original())
    
    if remove_duplicates:
        customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates(subset=customer_origins_unique_id)
        customer_origins_geodata_gpd_original = customer_origins_geodata_gpd_original.drop_duplicates(subset=customer_origins_unique_id)
    
    if customer_origins_metadata["marketsize_col"] is None:
        customer_origins_geodata_gpd_original[customer_origins_marketsize] = np.nan    
    
    customer_origins_geodata_gpd[customer_origins_unique_id] = customer_origins_geodata_gpd[customer_origins_unique_id].astype(str)
    customer_origins_geodata_gpd_original[customer_origins_unique_id] = customer_origins_geodata_gpd_original[customer_origins_unique_id].astype(str)

    customer_origins_data = pd.merge(
        customer_origins_geodata_gpd,
        customer_origins_geodata_gpd_original[
            [
                customer_origins_unique_id, 
                customer_origins_marketsize
                ]
            ],
        left_on = customer_origins_unique_id,
        right_on = customer_origins_unique_id 
        )
    customer_origins_data = customer_origins_data.rename(
        columns = {
            customer_origins_unique_id: config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            customer_origins_marketsize: config.DEFAULT_COLNAME_MARKETSIZE,
            "geometry": f"{config.DEFAULT_COLNAME_CUSTOMER_ORIGINS}_coords"
            }
        )

    supply_locations_unique_id = supply_locations_metadata["unique_id"]
    supply_locations_attraction = supply_locations_metadata["attraction_col"][0]   

    supply_locations_geodata_gpd = pd.DataFrame(supply_locations.get_geodata_gpd())
    supply_locations_geodata_gpd_original = pd.DataFrame(supply_locations.get_geodata_gpd_original())
    
    if remove_duplicates:
        supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates(subset=supply_locations_unique_id)
        supply_locations_geodata_gpd_original = supply_locations_geodata_gpd_original.drop_duplicates(subset=supply_locations_unique_id)
        
    if supply_locations_metadata["attraction_col"][0] is None:
        supply_locations_geodata_gpd_original[supply_locations_attraction] = np.nan
    
    supply_locations_data = pd.merge(
        supply_locations_geodata_gpd,
        supply_locations_geodata_gpd_original[[supply_locations_unique_id, supply_locations_attraction]],
        left_on = supply_locations_unique_id,
        right_on = supply_locations_unique_id 
        )
    supply_locations_data = supply_locations_data.rename(columns = {
        supply_locations_unique_id: config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
        supply_locations_attraction: config.DEFAULT_COLNAME_ATTRAC,
        "geometry": f"{config.DEFAULT_COLNAME_SUPPLY_LOCATIONS}_coords"
        }
        )

    interaction_matrix_df = customer_origins_data.merge(
        supply_locations_data, 
        how = "cross"
        )
    interaction_matrix_df[config.DEFAULT_COLNAME_INTERACTION] = interaction_matrix_df[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].astype(str)+config.MATRIX_OD_SEPARATOR+interaction_matrix_df[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS].astype(str)
    interaction_matrix_df[config.DEFAULT_COLNAME_TC] = None
    interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY] = None
    interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] = None
    interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS] = None

    metadata = {}

    interaction_matrix = InteractionMatrix(
        interaction_matrix_df,
        customer_origins,
        supply_locations,
        metadata
        )
    
    return interaction_matrix

def market_shares(
    df: pd.DataFrame,
    turnover_col: str,
    ref_col: str = None,
    marketshares_col: str = config.DEFAULT_COLNAME_PROBABILITY,
    drop_total_col: bool = True,
    check_df_vars: bool = True
    ):

    if check_df_vars:
        helper.check_vars(
            df = df,
            cols = [turnover_col]
            )
    
    if ref_col is not None:

        if ref_col not in df.columns:
            raise KeyError(f"Error while calculating market shares: Column '{ref_col}' not in dataframe.")
        
        ms_refcol = pd.DataFrame(df.groupby(ref_col)[turnover_col].sum())
        ms_refcol = ms_refcol.rename(columns = {turnover_col: f"{turnover_col}{config.DEFAULT_TOTAL_SUFFIX}"})
        ms_refcol = ms_refcol.reset_index()

        df = df.merge(
            ms_refcol,
            how = "left",
            left_on = ref_col,
            right_on= ref_col 
        )

    else:

        ms_norefcol = pd.DataFrame([df[turnover_col].sum()], columns=[f"{turnover_col}{config.DEFAULT_TOTAL_SUFFIX}"])
        ms_norefcol = ms_norefcol.reset_index()

        df["key_temp"] = 1
        ms_norefcol["key_temp"] = 1
        df = pd.merge(
            df, 
            ms_norefcol, 
            on="key_temp"
            ).drop(
                "key_temp", 
                axis=1
                )

    df[marketshares_col] = df[turnover_col]/df[f"{turnover_col}{config.DEFAULT_TOTAL_SUFFIX}"]
    
    if drop_total_col:
        df = df.drop(columns=f"{turnover_col}{config.DEFAULT_TOTAL_SUFFIX}")

    return df

def get_isochrones(
    geodata_gpd: gp.GeoDataFrame,
    unique_id_col: str,
    segments: list = [5, 10, 15],
    range_type: str = "time",
    intersections: str = "true",
    profile: str = "driving-car",
    donut: bool = True,
    ors_server: str = "https://api.openrouteservice.org/v2/",
    ors_auth: str = None,    
    timeout = 10,
    delay = 1,
    save_output: bool = True,
    output_filepath: str = "isochrones.shp",
    output_crs: str = "EPSG:4326",
    verbose: bool = config.VERBOSE
    ):

    coords = [(point.x, point.y) for point in geodata_gpd.geometry]
    
    unique_id_values = geodata_gpd[unique_id_col].values

    ors_client = Client(
        server = ors_server,
        auth = ors_auth
        )
    
    isochrones_gdf = gp.GeoDataFrame(columns=[unique_id_col, "geometry"])
    
    if range_type == "time":
        segments = [segment*60 for segment in segments]
    if range_type == "distance":
        segments = [segment*1000 for segment in segments]

    i = 0

    for x, y in coords:
        
        isochrone_output = ors_client.isochrone(
            locations = [[x, y]],
            segments = segments,
            range_type = range_type,
            intersections = intersections,
            profile = profile,
            timeout = timeout,
            save_output = False,
            output_crs = output_crs,
            verbose = verbose
            )
        
        if isochrone_output.status_code != 200:
            continue        
        
        isochrone_gdf = isochrone_output.get_isochrones_gdf()
        
        if donut:
            isochrone_gdf = overlay_difference(
                polygon_gdf = isochrone_gdf, 
                sort_col = config.ORS_SEGMENT_COL,
                verbose=verbose
                )
            
        time.sleep(delay)

        isochrone_gdf[unique_id_col] = unique_id_values[i]
        
        isochrone_gdf = isochrone_gdf.rename(columns={config.ORS_SEGMENT_COL: config.DEFAULT_SEGMENTS_COL})
        
        if range_type == "time":
            isochrone_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_min"] = isochrone_gdf[config.DEFAULT_SEGMENTS_COL]/60
        if range_type == "distance":
            isochrone_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_km"] = isochrone_gdf[config.DEFAULT_SEGMENTS_COL]/1000        

        isochrones_gdf = pd.concat(
            [
                isochrones_gdf, 
                isochrone_gdf
                ], 
            ignore_index=True
            )
        
        i = i+1

    if len(isochrones_gdf) == 0:
        raise ValueError("Error in isochrones calculation: No isochrones were retrieved. Probably ORS server error. Check output above and try again later.")

    isochrones_gdf[config.DEFAULT_SEGMENTS_COL] = isochrones_gdf[config.DEFAULT_SEGMENTS_COL].astype(int)
    
    if range_type == "time":
        isochrones_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_min"] = isochrones_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_min"].astype(int)
    if range_type == "distance":
        isochrones_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_km"] = isochrones_gdf[f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_km"].astype(int)
    
    isochrones_gdf.set_crs(
        output_crs, 
        allow_override=True, 
        inplace=True
        )
        
    if save_output:

        isochrones_gdf.to_file(filename = output_filepath)

    return isochrones_gdf

def log_centering_transformation(
    df: pd.DataFrame,
    ref_col: str,
    cols: list,
    suffix: str = config.DEFAULT_LCT_SUFFIX
    ):
   
    helper.check_vars(
        df = df,
        cols = cols
        )
    
    if ref_col not in df.columns:
        raise KeyError(f"Error in log-centering transformation: Column '{ref_col}' not in dataframe.")

    def lct (x):

        x_geom = np.exp(np.log(x).mean())
        x_lct = np.log(x/x_geom)

        return x_lct
    
    for var in cols:
        
        unique_values = df[var].unique()
        
        if set(unique_values).issubset({0, 1}):
            
            df[var+suffix] = df[var]
            
            print (f"Column {var} is a dummy variable and requires/allows no log-centering transformation")
            
            continue

        if (df[var] <= 0).any():
            
            df[var+suffix] = float("nan")
            
            print (f"Column {var} contains values <= 0. No log-centering transformation possible.")
            
            continue

        var_t = df.groupby(ref_col)[var].apply(lct)
        var_t = var_t.reset_index()
        
        df[var+suffix] = var_t[var]

    return df

def weighting(
    values: pd.Series,
    func: str,
    b: float,
    c: float = None,
    a: float = 1.0
    ):
    
    if func not in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
        raise ValueError (f"Parameter 'func' must be one of {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")
    
    if not helper.check_numeric_series(values):
        raise TypeError("Vector given by parameter 'series' is not numeric")    
    
    result = None
    
    calc_formula = config.PERMITTED_WEIGHTING_FUNCTIONS[func]["function"]
    
    calc_dict = {"a": a, "b": b, "values": values, "np": np}
    
    if "c" in calc_formula:
        if c is None:
            raise ValueError("Parameter 'c' must be provided for this function")
        calc_dict["c"] = c
        
    result = eval(calc_formula, {}, calc_dict)
    
    return result