#-----------------------------------------------------------------------
# Name:        models (huff package)
# Purpose:     Huff Model classes and functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.8.1
# Last update: 2026-02-13 20:33
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import geopandas as gp
import numpy as np
import time
from statsmodels.formula.api import ols
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import copy
import huff.helper as helper
import huff.config as config
import huff.goodness_of_fit as gof
from huff.ors import Client, TimeDistanceMatrix, Isochrone
from huff.gistools import overlay_difference, distance_matrix, buffers, map_with_basemap, distance_matrix_from_gdf


class CustomerOrigins:

    """
    Container for customer origin locations and related spatial data.

    Stores original and processed geodata, metadata, isochrones, and buffers.

    Attributes
    ----------
    geodata_gpd : geopandas.GeoDataFrame
        Processed geospatial data for customer origins.
    geodata_gpd_original : geopandas.GeoDataFrame
        Original geospatial data before processing.
    metadata : dict
        Metadata about the customer origins dataset.
    isochrones_gdf : geopandas.GeoDataFrame or None
        Isochrones related to the customer origins, if available.
    buffers_gdf : geopandas.GeoDataFrame or None
        Buffers around customer origins, if available.
    """

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
        
        """
        Return the processed geodata stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial data contained in `self.geodata_gpd`.
        """
    
        return self.geodata_gpd

    def get_geodata_gpd_original(self):
    
        """
        Return the original geodata stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial data contained in `self.geodata_gpd_original`.
        """

        return self.geodata_gpd_original

    def get_metadata(self):
        
        """
        Return the metadata stored in this object.

        Returns
        -------
        dict
            Metadata associated with this CustomerOrigins object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'marketsize_col': str
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        """
        
        return self.metadata
    
    def get_isochrones_gdf(self):

        """
        Return the isochrone data stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial data of isochrones contained in the CustomerOrigins object.
            If no isochrones were retrieved, None is returned.
        """

        return self.isochrones_gdf

    def get_buffers_gdf(self):

        """
        Return the buffer data stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Buffers GeoDataFrame if available, otherwise None.
        """

        return self.buffers_gdf

    def summary(self):

        """
        Prints a summary of the CustomerOrigins object including no. of points, weightings etc..

        Returns
        -------
        dict
            Metadata associated with this CustomerOrigins object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'marketsize_col': str
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        """

        metadata = self.metadata

        print(config.DEFAULT_NAME_CUSTOMER_ORIGINS)
        print("======================================")

        helper.print_summary_row(
            "No. locations",
            metadata["no_points"]
        )
        
        helper.print_summary_row(
            config.DEFAULT_NAME_MARKETSIZE,
            metadata["marketsize_col"]
        )
        
        if metadata['weighting'][0]['param'] is not None:

            if isinstance(metadata['weighting'][0]['param'], list):
                tc_params = ', '.join(str(round(x, config.FLOAT_ROUND)) for x in metadata['weighting'][0]['param'])
            elif isinstance(metadata['weighting'][0]['param'], (int, float)):
                tc_params = round(metadata['weighting'][0]['param'], config.FLOAT_ROUND)
            else:
                tc_params = "Invalid format"

            func_description = config.PERMITTED_WEIGHTING_FUNCTIONS[metadata["weighting"][0]["func"]]["description"]

            tc_weighting = f"{tc_params} ({func_description})"

        else:
            tc_weighting = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_TC} weighting",
            tc_weighting
        )

        helper.print_summary_row(
            "Unique ID column",
            metadata["unique_id"]
        )
        helper.print_summary_row(
            "Input CRS",
            metadata["crs_input"]
        )

        helper.print_summary_row(
            "Isochrones",
            "YES" if self.isochrones_gdf is not None else "NO"
        )

        helper.print_summary_row(
            "Buffers",
            "YES" if self.buffers_gdf is not None else "NO"
        )

        print("--------------------------------------")

        return metadata
    
    def show_log(self):

        """
        Shows all timestamp logs of the CustomerOrigins object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp            
    
    def define_marketsize(
        self,
        marketsize_col: str,
        verbose: bool = config.VERBOSE
        ):

        """
        Define the column in the customer origins original data to be used as market size.

        Parameters
        ----------
        marketsize_col : str
            Name of the column representing market size.
        verbose : bool, optional
            If True, print status messages (default: False).

        Returns
        -------
        self : CustomerOrigins
            The instance with updated market size column and metadata.

        Example
        -------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_marketsize("pop")
        """

        geodata_gpd_original = self.geodata_gpd_original
        metadata = self.metadata
        
        helper.check_vars(
            df = geodata_gpd_original,
            cols = [marketsize_col],            
            check_zero = False,
            check_constant = False
            )

        metadata["marketsize_col"] = marketsize_col        

        helper.add_timestamp(
            self,
            function="models.CustomerOrigins.define_marketsize",
            process = f"Defined '{marketsize_col}' as market size column"
            )

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
        Define the weighting function for transport costs in the model.

        This function sets the weighting method for transport costs (`t_ij`) in 
        the model metadata. The weighting can follow different functional forms 
        (e.g., "power", "logistic") and is parameterized by `param_lambda`. 
        This affects how transport costs are transformed before use in calculations.

        Parameters
        ----------
        func : str, default="power"
            The weighting function to use. Must be one of the permitted functions 
            defined in `config.PERMITTED_WEIGHTING_FUNCTIONS_LIST`. Example functions:
            - "power": param_lambda is a single numeric value (exponent).
            - "logistic": param_lambda is a list of two numeric values [k, x0].
        param_lambda : float or list of float, default=-2
            Parameter(s) for the weighting function. The expected type and number 
            of parameters depend on `func`:
            - For "power": a single float (exponent).
            - For "logistic": a list of two floats [k, x0] (steepness and midpoint).
        verbose : bool, default=config.VERBOSE
            If True, prints detailed messages about the steps performed in the function.
            
        Returns
        -------
        self
            Returns the instance (CustomerOrigins) with updated metadata containing the transport
            costs weighting definition.

        Raises
        ------
        ValueError
            If `func` is not in the list of permitted weighting functions.
        TypeError
            If `param_lambda` is not of the expected type for the selected function.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
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
        
        helper.add_timestamp(
            self,
            function="models.CustomerOrigins.define_transportcosts_weighting",
            process = f"Defined transport costs weighting with {func} function"
            )
            
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
        
        """
        Retrieve isochrones for customer origins.

        This function calculates isochrones (areas reachable within specified 
        travel times or distances) around each location in the dataset. It uses 
        the OpenRouteService API via `get_isochrones()` and stores the result in 
        the instance (CustomerOrigins). Optionally, it can save the output as a shapefile.

        See the ORS API documentation: https://openrouteservice.org/dev/#/api-docs
        See the current API restrictions: https://openrouteservice.org/restrictions/

        Parameters
        ----------
        segments : list of int, default=[5, 10, 15]
            List of travel time or distance intervals for which isochrones should 
            be calculated (in minutes or km depending on `range_type`).
        range_type : str, default="time"
            Defines whether `segments` are interpreted as "time" or "distance".
        intersections : str, default="true"
            Whether to calculate intersections between isochrones. Accepts "true" or "false".
        profile : str, default="driving-car"
            Travel mode profile used by the routing service (e.g., "driving-car", 
            "cycling-regular", "foot-walking").
            See https://openrouteservice.org/dev/#/api-docs/v2/isochrones/{profile}/post for available profiles.
        donut : bool, default=True
            If True, returns donut-shaped isochrones (rings instead of cumulative areas).
        ors_server : str, default="https://api.openrouteservice.org/v2/"
            URL of the OpenRouteService API server.
        ors_auth : str, optional
            API key for OpenRouteService.
        timeout : int, default=10
            Timeout in seconds for API requests.
        delay : int, default=1
            Delay in seconds between consecutive API requests to avoid rate limits.
        save_output : bool, default=True
            If True, saves the resulting isochrones to a shapefile.
        output_filepath : str, default="customer_origins_isochrones.shp"
            File path to save the shapefile if `save_output` is True.
        output_crs : str, default="EPSG:4326"
            Coordinate reference system for the output geometries.
        verbose : bool, default=config.VERBOSE
            If True, prints detailed messages about the steps performed while 
            retrieving and processing the isochrones, including API requests 
            and file saving.

        Returns
        -------
        self
            Returns the instance with the `isochrones_gdf` attribute containing 
            the retrieved isochrones as a GeoDataFrame. If retrieval fails, 
            `isochrones_gdf` is set to None.

        Raises
        ------
        ValueError
            If `get_isochrones()` raises a ValueError.
        Exception
            For any other exceptions raised during isochrone retrieval.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.isochrones(
        ...     segments=[2, 4, 6],
        ...     range_type = "time",
        ...     profile = "foot-walking",
        ...     save_output=True,
        ...     ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
        ...     output_filepath="Haslach_iso.shp",
        ...     output_crs="EPSG:31467",
        ...     delay=0.2
        ... )
        """

        geodata_gpd = self.get_geodata_gpd()
        metadata = self.get_metadata()

        error_message = ""

        try:
        
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
        
        except ValueError as e:
            isochrones_gdf = None
            error_message = str(e)

        except Exception as e:
            isochrones_gdf = None
            error_message = str(e)
        
        if error_message == "" and len(isochrones_gdf) == 0:
            error_message = "Isochrones data are empty"
        
        helper.add_timestamp(
            self,
            function="models.CustomerOrigins.isochrones",
            process="Retrieved isochrones",
            status = "OK" if error_message == "" else error_message
            )

        self.isochrones_gdf = isochrones_gdf

        return self

    def buffers(
        self,
        segments_distance: list = None,
        donut: bool = True,
        save_output: bool = True,
        output_filepath: str = "customer_origins_buffers.shp",
        output_crs: str = "EPSG:4326",
        verbose: bool = config.VERBOSE
        ):

        """
        Create buffer zones around customer origin points.

        Parameters
        ----------
        segments_distance : list, optional
            Distances (in meters) for buffer rings (default: [500, 1000]).
        donut : bool, optional
            If True, create donut-shaped rings (default: True).
        save_output : bool, optional
            If True, save buffers to a file (default: True).
        output_filepath : str, optional
            File path for saving buffers (default: "customer_origins_buffers.shp").
        output_crs : str, optional
            Coordinate reference system for output (default: "EPSG:4326").
        verbose : bool, optional
            If True, print status messages (default taken from config.VERBOSE).

        Returns
        -------
        self : CustomerOrigins
            The instance with `buffers_gdf` added.

        Example
        -------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach_buf = Haslach.buffers(
        ...     segments_distance=[500,1000,1500],
        ...     save_output=True,
        ...     output_filepath="Haslach_buf.shp",
        ...     output_crs="EPSG:31467"
        ... )
        """

        if segments_distance is None:
            segments_distance = [500, 1000]

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
                
        helper.add_timestamp(
            self,
            function="models.CustomerOrigins.buffers",
            process=f"Created buffers for {len(segments_distance)} segments"
            )    

        self.buffers_gdf = buffers_gdf

        return self
    
    def plot(
        self,
        point_style,
        polygon_style = {},
        save_output: bool = True,
        output_filepath: str = "customer_origins.png",
        output_dpi = 300,
        zoom: int = 15,
        legend: bool = True,
        map_title: str = "Map of customer origins with OSM basemap",
        verbose: bool = False
        ):

        """
        Plot customer origins with optional isochrones or buffer layers.

        Parameters
        ----------
        point_style : dict
            Style for point geometries.
        polygon_style : dict, optional
            Style for polygons (isochrones or buffers) (default: {}).
        save_output : bool, optional
            If True, save map to file (default: True).
        output_filepath : str, optional
            Path for saving the map (default: "customer_origins.png").
        output_dpi : int, optional
            Resolution for saved image (default: 300).
        zoom : int, optional
            Zoom level for basemap (default: 15).
        legend : bool, optional
            If True, include a legend (default: True).
        map_title : str, optional
            Title of the map (default: "Map of customer origins with OSM basemap").
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        list
            [matplotlib Figure or PIL Image, layers plotted, styles used]

        Example
        -------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach_buf = Haslach.buffers(
        ...     segments_distance=[500,1000,1500],
        ...     save_output=True,
        ...     output_filepath="Haslach_buf.shp",
        ...     output_crs="EPSG:31467"
        ... )
        >>> Haslach.plot(
        ...     point_style = {
        ...         "name": "Districts",
        ...         "color": "black",
        ...         "alpha": 1,
        ...         "size": 15,
        ...     },
        ...     polygon_style= {
        ...         "name": "Buffers",
        ...         "color": {
        ...             "buffer": {
        ...                 500: "midnightblue", 
        ...                 1000: "blue", 
        ...                 1500: "dodgerblue",
        ...             }
        ...     },            
        ...     "alpha": 0.3
        ...     },
        ...     map_title = "Districts in Haslach with buffers",    
        ... )
        """

        customer_origins_gdf = self.get_geodata_gpd_original()

        if "geometry" not in customer_origins_gdf.columns:
            print("No map plot possible because customer origins lack geometry.")
            return None

        if verbose:
            print("Extracting layers from CustomerOrigins object", end = " ... ")

        layers_to_plot = [customer_origins_gdf]
        layers_to_plot_included = ["Customer origins points"] 

        if self.get_isochrones_gdf() is not None:
            customer_origins_gdf_iso = self.get_isochrones_gdf()
            layers_to_plot = layers_to_plot + [customer_origins_gdf_iso]
            layers_to_plot_included = layers_to_plot_included + ["Isochrones"]
            color_key = f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_min"
        else:
            if self.get_buffers_gdf() is not None:
                customer_origins_gdf_buf = self.get_buffers_gdf()
                layers_to_plot = layers_to_plot + [customer_origins_gdf_buf]
                layers_to_plot_included = layers_to_plot_included + ["Buffers"]
                color_key = config.DEFAULT_SEGMENTS_COL

        if verbose:
            print("OK")
            print(f"NOTE: CustomerOrigins object contains {len(layers_to_plot)} layers to plot: {', '.join(layers_to_plot_included)}.")
      
        layer_styles = {
            0: point_style
            }

        if polygon_style is not None and len(polygon_style) > 0:
            
            layer_styles[1] = polygon_style

            color_key_old = next(iter(layer_styles[1]["color"]))
            layer_styles[1]["color"][color_key] = layer_styles[1]["color"].pop(color_key_old)

        assert len(layers_to_plot) == len(layer_styles), f"Error while trying to plot customer origins: There are {len(layers_to_plot)} layers to plot but {len(layer_styles)} plot styles were stated." 

        map_osm = map_with_basemap(
            layers = layers_to_plot,
            styles = layer_styles,
            save_output = save_output,
            output_filepath = output_filepath,
            output_dpi = output_dpi,
            legend = legend,
            map_title = map_title,
            zoom = zoom,
            verbose = False
            )
        
        return [
            map_osm,
            layers_to_plot,
            layer_styles
            ]

class SupplyLocations:

    """
    Container for supply location data and related spatial information.

    Stores original and processed geodata, metadata, isochrones, and buffers.

    Attributes
    ----------
    geodata_gpd : geopandas.GeoDataFrame
        Processed geospatial data for supply locations.
    geodata_gpd_original : geopandas.GeoDataFrame
        Original geospatial data before processing.
    metadata : dict
        Metadata about the supply locations dataset.
    isochrones_gdf : geopandas.GeoDataFrame or None
        Isochrones related to the supply locations, if available.
    buffers_gdf : geopandas.GeoDataFrame or None
        Buffers around supply locations, if available.
    """

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

        """
        Return the processed geodata stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial data contained in `self.geodata_gpd`.
        """

        return self.geodata_gpd

    def get_geodata_gpd_original(self):

        """
        Return the original geodata stored in this object.

        Returns
        -------
        geopandas.GeoDataFrame
            The geospatial data contained in `self.geodata_gpd`.
        """

        return self.geodata_gpd_original

    def get_metadata(self):

        """
        Return the metadata stored in this object.

        Returns
        -------
        dict
            Metadata associated with this SupplyLocations object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'attraction_col': list
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        """

        return self.metadata
    
    def get_isochrones_gdf(self):
        """
        Return the isochrones GeoDataFrame for supply locations.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Isochrones GeoDataFrame if available, otherwise None.
        """

        return self.isochrones_gdf
    
    def get_buffers_gdf(self):

        """
        Return the buffers GeoDataFrame for supply locations.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Buffers GeoDataFrame if available, otherwise None.
        """

        return self.buffers_gdf

    def summary(self):

        """
        Prints a summary of the SupplyLocations object including no. of points, weightings etc..

        Returns
        -------
        dict
            Metadata associated with this SupplyLocations object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'attraction_col': list
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        """

        metadata = self.metadata

        print(config.DEFAULT_NAME_SUPPLY_LOCATIONS)        
        print("======================================")

        helper.print_summary_row(
            "No. locations",
            metadata["no_points"]
        )

        if metadata["attraction_col"][0] is not None:
            if isinstance(metadata["attraction_col"], list):
                attrac_cols = ', '.join(str(x) for x in metadata["attraction_col"])
            elif isinstance(metadata["attraction_col"], str):
                attrac_cols = ', '.join([metadata["attraction_col"]])
            else:
                attrac_cols = metadata["attraction_col"]
        else:
            attrac_cols = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_ATTRAC} column(s)",
            attrac_cols
        )
        
        if len(metadata["weighting"]) > 0 and metadata["weighting"][0]["func"] is not None:
            print("Weightings")

        for key, value in metadata["weighting"].items():

            if value['name'] is not None and value['func'] is not None and value['param'] is not None:

                if metadata["attraction_col"][key] is not None:
                    name = metadata["attraction_col"][key]
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

                helper.print_summary_row(
                    name,
                    f"{param} ({func_description})"
                )

            else:
                if key == 0:
                    break

        helper.print_summary_row(
            "Unique ID column",
            metadata["unique_id"]
        )

        helper.print_summary_row(
            "Input CRS",
            metadata["crs_input"]
        )

        helper.print_summary_row(
            "Isochrones",
            "YES" if self.isochrones_gdf is not None else "NO"
        )

        helper.print_summary_row(
            "Buffers",
            "YES" if self.buffers_gdf is not None else "NO"
        )

        print("======================================")

        return metadata

    def show_log(self):
        
        """
        Shows all timestamp logs of the SupplyLocations object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp 

    def define_attraction(
        self,
        attraction_col: str,
        verbose: bool = config.VERBOSE
        ):

        """
        Define the column in the supply locations original data 
        representing attraction values for supply locations.

        Parameters
        ----------
        attraction_col : str
            Name of the column to be used as the attraction variable.
        verbose : bool, optional
            If True, prints status messages (default is config.VERBOSE).

        Returns
        -------
        self : SupplyLocations
            The same object with updated metadata.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        """

        geodata_gpd_original = self.geodata_gpd_original
        metadata = self.metadata
        
        helper.check_vars(
            df = geodata_gpd_original,
            cols = [attraction_col],            
            check_zero = False,
            check_constant = False
            )

        if attraction_col not in geodata_gpd_original.columns:
            raise KeyError (f"Error while defining attraction variable: Column {attraction_col} not in data")
        else:
            metadata["attraction_col"][0] = attraction_col
        
        helper.add_timestamp(
            self,
            function="models.SupplyLocations.define_attraction",
            process = f"Defined '{attraction_col}' as attraction column"
            )

        if verbose:
            print(f"Set attraction variable to column {attraction_col}")

        return self
    
    def define_attraction_weighting(
        self,
        func = "power",
        param_gamma = 1
        ):

        """
        Set the weighting function and parameter for the attraction column.
        Must be called after define_attraction().

        Parameters
        ----------
        func : str, optional
            Weighting function to use (default is "power").
        param_gamma : float, optional
            Parameter for the weighting function (default is 1).

        Returns
        -------
        self : SupplyLocations
            The same object with updated metadata.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        >>> Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        """

        metadata = self.metadata

        if metadata["attraction_col"] is None:
            raise ValueError(f"Error while defining attraction weighting: {config.DEFAULT_NAME_ATTRAC} column is not yet defined. Use SupplyLocations.define_attraction()")
        
        if func not in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
            raise ValueError(f"Error while defining attraction weighting: Parameter 'func' was set to {func}. Permitted weighting functions are: {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")
        
        metadata["weighting"][0]["name"] = config.DEFAULT_COLNAME_ATTRAC
        metadata["weighting"][0]["func"] = func
        metadata["weighting"][0]["param"] = float(param_gamma)
        
        helper.add_timestamp(
            self,
            function="models.SupplyLocations.define_attraction_weighting",
            process = f"Defined attraction weighting with {func} function with gamma = {param_gamma}"
            )

        return self

    def add_var(
        self,
        var: str = None,
        func: str = None,
        param: float = None
        ):

        """
        Add a new attraction variable `var` and its weighting.

        Parameters
        ----------
        var : str
            Name of the new attraction variable.
        func : str
            Weighting function for the variable.
        param : float
            Parameter for the weighting function.

        Returns
        -------
        self : SupplyLocations
            Updated object with the new variable added.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        >>> Haslach_supermarkets.add_var(
        ...     var = "Parkplaetze",
        ...     func = "power",
        ...     param = 1
        ... )
        """
            
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

        helper.add_timestamp(
            self,
            function="models.SupplyLocations.add_var",
            process = f"Added '{var}' as attraction variable #{new_key}"
            )
        
        return self
    
    def change_attraction_values(
        self,
        new_attraction_values: dict
        ):

        """
        Update attraction values for supply locations.

        Parameters
        ----------
        new_attraction_values : dict
            Dictionary mapping location identifiers to new attraction values.
            Each entry must contain:
            - 'location': identifier of the supply location
            - 'attraction_col': column name of the attraction variable
            - 'new_value': new value to assign

        Returns
        -------
        self : SupplyLocations
            Updated object with modified attraction values.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        >>> new_attraction_values = {
        ...     0: {
        ...         "location": "1",
        ...         "attraction_col": "VKF_qm",
        ...         "new_value": 2700
        ...     }
        ... }
        ... Haslach_supermarkets.change_attraction_values(new_attraction_values)
        """

        if len(new_attraction_values) == 0:
            raise KeyError("Dictionary 'new_attraction_values' is empty")

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
        
        helper.add_timestamp(
            self,
            function="models.SupplyLocations.change_attraction_values",
            process = f"Changed {len(new_attraction_values)} attraction values"
            )

        return self

    def add_new_destinations(
        self,
        new_destinations,
        ):

        """
        Add new supply locations to the current SupplyLocations object.

        Parameters
        ----------
        new_destinations : SupplyLocations
            Object containing additional supply locations to append.

        Returns
        -------
        self : SupplyLocations
            Updated object with new destinations included.

        Raises
        ------
        TypeError
            If `new_destinations` is not a SupplyLocations object.
        KeyError
            If the column names of the new destinations do not match the existing data.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_new_supermarket = load_geodata(
        ...     "data/Haslach_new_supermarket.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        >>> Haslach_supermarkets.add_new_destinations(Haslach_new_supermarket)
        """
        
        if not isinstance(new_destinations, SupplyLocations):
            raise TypeError("Parameter 'new_destinations' must be a SupplyLocations object")

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
        
        helper.add_timestamp(
            self,
            function="models.SupplyLocations.add_new_destinations",
            process = f"Added {len(new_destinations_gpd_original)} new supply locations"
            )
        
        self.geodata_gpd = geodata_gpd
        self.geodata_gpd_original = geodata_gpd_original
        self.metadata = metadata

        return self
    
    def isochrones(
        self,
        segments: list = None,
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

        """
        Retrieve isochrones for supply locations.

        This function calculates isochrones (areas reachable within specified 
        travel times or distances) around each location in the dataset. It uses 
        the OpenRouteService API via `get_isochrones()` and stores the result in 
        the instance (SupplyLocations). Optionally, it can save the output as a shapefile.

        See the ORS API documentation: https://openrouteservice.org/dev/#/api-docs
        See the current API restrictions: https://openrouteservice.org/restrictions/

        Parameters
        ----------
        segments : list of int, default=[5, 10, 15]
            List of travel time or distance intervals for which isochrones should 
            be calculated (in minutes or km depending on `range_type`).
        range_type : str, default="time"
            Defines whether `segments` are interpreted as "time" or "distance".
        intersections : str, default="true"
            Whether to calculate intersections between isochrones. Accepts "true" or "false".
        profile : str, default="driving-car"
            Travel mode profile used by the routing service (e.g., "driving-car", 
            "cycling-regular", "foot-walking").
            See https://openrouteservice.org/dev/#/api-docs/v2/isochrones/{profile}/post for available profiles.
        donut : bool, default=True
            If True, returns donut-shaped isochrones (rings instead of cumulative areas).
        ors_server : str, default="https://api.openrouteservice.org/v2/"
            URL of the OpenRouteService API server.
        ors_auth : str, optional
            API key for OpenRouteService.
        timeout : int, default=10
            Timeout in seconds for API requests.
        delay : int, default=1
            Delay in seconds between consecutive API requests to avoid rate limits.
        save_output : bool, default=True
            If True, saves the resulting isochrones to a shapefile.
        output_filepath : str, default="customer_origins_isochrones.shp"
            File path to save the shapefile if `save_output` is True.
        output_crs : str, default="EPSG:4326"
            Coordinate reference system for the output geometries.
        verbose : bool, default=config.VERBOSE
            If True, prints detailed messages about the steps performed while 
            retrieving and processing the isochrones, including API requests 
            and file saving.

        Returns
        -------
        self
            Returns the instance with the `isochrones_gdf` attribute containing 
            the retrieved isochrones as a GeoDataFrame. If retrieval fails, 
            `isochrones_gdf` is set to None.

        Raises
        ------
        ValueError
            If `get_isochrones()` raises a ValueError.
        Exception
            For any other exceptions raised during isochrone retrieval.

        Example
        --------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.isochrones(
        ...     segments=[2, 4, 6],
        ...     range_type = "time",
        ...     profile = "foot-walking",
        ...     save_output=True,
        ...     ors_auth="5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f",
        ...     output_filepath="Haslach_supermarkets_iso.shp",
        ...     output_crs="EPSG:31467",
        ...     delay=0.2
        ... )
        """

        if segments is None:
            segments = [5, 10, 15]

        geodata_gpd = self.get_geodata_gpd()
        metadata = self.get_metadata()

        error_message = ""

        try:

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
            
        except ValueError as e:
            isochrones_gdf = None
            error_message = str(e)

        except Exception as e:
            isochrones_gdf = None
            error_message = str(e)
        
        if error_message == "" and len(isochrones_gdf) == 0:
            error_message = "Isochrones data are empty"
      
        helper.add_timestamp(
            self,
            function="models.SupplyLocations.isochrones",
            process="Retrieved isochrones",
            status = "OK" if error_message == "" else error_message
            )

        self.isochrones_gdf = isochrones_gdf

        return self

    def buffers(
        self,
        segments_distance: list = None,
        donut: bool = True,
        save_output: bool = True,
        output_filepath: str = "supply_locations_buffers.shp",
        output_crs: str = "EPSG:4326"
        ):

        """
        Create buffer zones around supply location points.

        Parameters
        ----------
        segments_distance : list, optional
            Distances (in meters) for buffer rings (default: [500, 1000]).
        donut : bool, optional
            If True, create donut-shaped rings (default: True).
        save_output : bool, optional
            If True, save buffers to a file (default: True).
        output_filepath : str, optional
            File path for saving buffers (default: "supply_locations_buffers.shp").
        output_crs : str, optional
            Coordinate reference system for output (default: "EPSG:4326").
        verbose : bool, optional
            If True, print status messages (default: False).

        Returns
        -------
        self : CustomerOrigins
            The instance with `buffers_gdf` added.

        Example
        -------
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets_buf = Haslach_supermarkets.buffers(
        ...     segments_distance=[250,500,750,1000],
        ...     save_output=True,
        ...     output_filepath="Haslach_supermarkets_buf.shp",
        ...     output_crs="EPSG:31467"
        ... )
        """

        if segments_distance is None:
            segments_distance = [500, 1000]

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

        helper.add_timestamp(
            self,
            function="models.SupplyLocations.buffers",
            process=f"Created buffers for {len(segments_distance)} segments"
            )
        
        return self
    
    def plot(
        self,
        point_style={},
        polygon_style={},
        save_output: bool = True,
        output_filepath: str = "supply_locations.png",
        output_dpi = 300,
        zoom: int = 15,
        legend: bool = True,
        map_title: str = "Map of supply locations with OSM basemap",
        verbose: bool = False
        ):

        """
        Plot customer origins with optional isochrones or buffer layers.

        Parameters
        ----------
        point_style : dict
            Style for point geometries.
        polygon_style : dict, optional
            Style for polygons (isochrones or buffers) (default: {}).
        save_output : bool, optional
            If True, save map to file (default: True).
        output_filepath : str, optional
            Path for saving the map (default: "customer_origins.png").
        output_dpi : int, optional
            Resolution for saved image (default: 300).
        zoom : int, optional
            Zoom level for basemap (default: 15).
        legend : bool, optional
            If True, include a legend (default: True).
        map_title : str, optional
            Title of the map (default: "Map of customer origins with OSM basemap").
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        list
            [matplotlib Figure or PIL Image, layers plotted, styles used]

        Example
        -------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach_buf = Haslach.buffers(
        ...     segments_distance=[500,1000,1500],
        ...     save_output=True,
        ...     output_filepath="Haslach_buf.shp",
        ...     output_crs="EPSG:31467"
        ... )
        >>> Haslach.plot(
        ...     point_style = {
        ...         "name": "Supermarket chains",
        ...         "color": {
        ...             "Name": {
        ...                 "Aldi Süd": "blue",
        ...                 "Edeka": "yellow",
        ...                 "Lidl": "red",
        ...                 "Netto": "orange",
        ...                 "Real": "darkblue",
        ...                 "Treff 3000": "fuchsia"
        ...                 }
        ...             },
        ...         "alpha": 1,
        ...         "size": 30
        ...     },
        ...     polygon_style = {
        ...         "name": "Isochrones",
        ...         "color": {
        ...             "segm_min": {
        ...                 3: "midnightblue", 
        ...                 6: "blue", 
        ...                 9: "dodgerblue", 
        ...                 12: "deepskyblue", 
        ...                 15: "aqua"
        ...                 }
        ...             },            
        ...         "alpha": 0.3
        ...     },
        ... map_title = "Grocery stores in Haslach with isochrones",    
        ... )
        """

        supply_locations_gdf = self.get_geodata_gpd_original()

        if "geometry" not in supply_locations_gdf.columns:
            print("No map plot possible because supply locations lack geometry.")
            return None

        if verbose:
            print("Extracting layers from SupplyLocations object", end = " ... ")

        layers_to_plot = [supply_locations_gdf]
        layers_to_plot_included = ["Supply locations points"] 

        if self.get_isochrones_gdf() is not None:
            supply_locations_gdf_iso = self.get_isochrones_gdf()
            layers_to_plot = layers_to_plot + [supply_locations_gdf_iso]
            layers_to_plot_included = layers_to_plot_included + ["Isochrones"]
            color_key = f"{config.DEFAULT_SEGMENTS_COL_ABBREV}_min"
        else:
            if self.get_buffers_gdf() is not None:
                supply_locations_gdf_buf = self.get_buffers_gdf()
                layers_to_plot = layers_to_plot + [supply_locations_gdf_buf]
                layers_to_plot_included = layers_to_plot_included + ["Buffers"]
                color_key = config.DEFAULT_SEGMENTS_COL            

        if verbose:
            print("OK")
            print(f"NOTE: SupplyLocations object contains {len(layers_to_plot)} layers to plot: {', '.join(layers_to_plot_included)}.")
      
        layer_styles = {
            0: point_style
            }

        if polygon_style is not None and len(polygon_style) > 0 and len(layers_to_plot) > 1:
            layer_styles[1] = polygon_style

            color_key_old = next(iter(layer_styles[1]["color"]))
            layer_styles[1]["color"][color_key] = layer_styles[1]["color"].pop(color_key_old)

        assert len(layers_to_plot) == len(layer_styles), f"Error while trying to plot supply locations: There are {len(layers_to_plot)} layers to plot but {len(layer_styles)} plot styles were stated." 

        map_osm = map_with_basemap(
            layers = layers_to_plot,
            styles = layer_styles,
            save_output = save_output,
            output_filepath = output_filepath,
            output_dpi = output_dpi,
            legend = legend,
            map_title = map_title,
            zoom = zoom,
            verbose = verbose
            )
        
        return [
            map_osm,
            layers_to_plot,
            layer_styles
            ]
        
class InteractionMatrixError(Exception):
    """
    Error class for any errors in interaction matrix calculations
    """
    pass

class InteractionMatrix:

    """
    Container for origin-destination interaction matrix used in spatial market models.

    The InteractionMatrix stores all possible customer origin–supply location
    combinations together with model inputs (e.g., travel times, distances,
    and attributes) and provides the basis for all implemented model analyses.

    Parameters
    ----------
    interaction_matrix_df : pandas.DataFrame
        DataFrame containing origin–destination pairs and associated variables.
    customer_origins : CustomerOrigins
        CustomerOrigins object used to build the interaction matrix.
    supply_locations : SupplyLocations
        SupplyLocations object used to build the interaction matrix.
    metadata : dict
        Metadata describing model configuration and processing steps.
    """

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

        """
        Return the interaction matrix data stored in this object.

        Returns
        -------
        pandas.DataFrame
            The interaction matrix contained in `self.interaction_matrix_df`.
        """

        return self.interaction_matrix_df
    
    def get_customer_origins(self):

        """
        Return the CustomerOrigins object stored in this object.

        Returns
        -------
        huff.models.CustomerOrigins
            The CustomerOrigins object contained in the InteractionMatrix object.
            Instances of CustomerOrigins include: geodata_gpd (geopandas.GeoDataFrame),
            geodata_gpd_original (geopandas.GeoDataFrame), metadata (dict),
            isochrones_gdf (geopandas.GeoDataFrame or None), 
            buffers_gdf (geopandas.GeoDataFrame or None).
        """

        return self.customer_origins

    def get_supply_locations(self):

        """
        Return the SupplyLocations object stored in this object.

        Returns
        -------
        huff.models.SupplyLocations
            The SupplyLocations object contained in the InteractionMatrix object.
            Instances of SupplyLocations include: geodata_gpd (geopandas.GeoDataFrame),
            geodata_gpd_original (geopandas.GeoDataFrame), metadata (dict),
            isochrones_gdf (geopandas.GeoDataFrame or None), 
            buffers_gdf (geopandas.GeoDataFrame or None).
        """

        return self.supply_locations
    
    def get_metadata(self):

        """
        Return the metadata stored in this object.

        Returns
        -------
        dict
            Metadata associated with this InteractionMatrix object. Keys may include:
            - 'fit': dict
        """
        
        return self.metadata

    def summary(self):

        """
        Prints a summary of the InteractionMatrix object including information on
        the included CustomerOrigins and SupplyLocations objects

        Returns
        -------
        list
        - Metadata associated with the CustomerOrigins object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'marketsize_col': str
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        - Metadata associated with the SupplyLocations object. Keys may include:
            - 'location_type': str
            - 'unique_id': str        
            - 'attraction_col': list
            - 'weighting': dict
            - 'crs_input': str
            - 'crs_output': str
            - 'no_points': int
        - Metadata associated with the InteractionMatrix object. Keys may include:
            - 'fit': dict
        """

        customer_origins_metadata = self.get_customer_origins().get_metadata()
        supply_locations_metadata = self.get_supply_locations().get_metadata()
        interaction_matrix_metadata = self.get_metadata()

        print(config.DEFAULT_NAME_INTERACTION_MATRIX)
        print("======================================")
                
        helper.print_summary_row(
            config.DEFAULT_NAME_SUPPLY_LOCATIONS,
            supply_locations_metadata["no_points"]
        )

        if supply_locations_metadata["attraction_col"][0] is not None:
            if isinstance(supply_locations_metadata["attraction_col"], list):
                attrac_cols = ', '.join(str(x) for x in supply_locations_metadata["attraction_col"])
            elif isinstance(supply_locations_metadata["attraction_col"], str):
                attrac_cols = ', '.join([supply_locations_metadata["attraction_col"]])
            else:
                attrac_cols = supply_locations_metadata["attraction_col"]
        else:
            attrac_cols = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_ATTRAC} column(s)",
            attrac_cols
        )

        print("--------------------------------------")

        helper.print_summary_row(
            config.DEFAULT_NAME_CUSTOMER_ORIGINS,
            customer_origins_metadata["no_points"]
        )

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_MARKETSIZE} column",
            customer_origins_metadata["marketsize_col"]
        )
        
        helper.print_interaction_matrix_info(self)

        helper.print_weightings(self)

        print("======================================")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata
            ]

    def show_log(self):
        
        """
        Shows all timestamp logs of the InteractionMatrix object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp  

    def transport_costs(
        self,
        network: bool = True,
        range_type: str = "time",
        profile: str = "driving-car",
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

        """
        Compute transport costs between customer origins and supply locations.

        Transport costs are calculated either using the OpenRouteService (network-based)
        or as Euclidean distances. Results are added to the interaction matrix as a
        numeric transport cost column.
        
        If using ORS:
        See the ORS API documentation: https://openrouteservice.org/dev/#/api-docs
        See the current API restrictions: https://openrouteservice.org/restrictions/
                

        Parameters
        ----------
        network : bool, optional
            If True, compute network-based travel costs using OpenRouteService.
            If False, compute Euclidean distances.
        range_type : str, optional
            Cost type to compute ("time" or "distance").
        profile : str, optional
            Mode of travel: "driving-car", "cycling-regular", etc. (default: "driving-car").
        time_unit : str, optional
            Unit for time-based costs ("minutes" or "hours").
        distance_unit : str, optional
            Unit for distance-based costs ("kilometers" or "meters").
        ors_server : str, optional
            OpenRouteService API endpoint.
            Necessary if network-based transport costs are desired (network = True).
        ors_auth : str, optional
            API key for OpenRouteService.
            Necessary if network-based transport costs are desired (network = True).
        save_output : bool, optional
            If True, save the transport cost matrix to file.
        remove_duplicates : bool, optional
            If True, remove duplicate origin and destination locations.
        output_filepath : str, optional
            File path for saving the transport cost matrix.
        shp_save_output : bool, optional
            If True, save Euclidean distance lines as a shapefile.
        shp_output_filepath : str, optional
            Output file path for shapefile.
        shp_output_crs : str, optional
            CRS for shapefile output.
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        InteractionMatrix
            Updated InteractionMatrix object with transport costs added.

        Example
        -------
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
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        """

        if not network and range_type == "time":
            
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

        error_message = ""

        if network:

            ors_client = Client(
                server = ors_server,
                auth = ors_auth
                )
            time_distance_matrix = ors_client.matrix(
                locations = locations_coords,                
                sources = customer_origins_index,
                destinations = locations_coords_index,
                range_type = range_type,
                profile = profile,
                save_output = save_output,
                output_filepath = output_filepath,
                verbose = verbose
                )
            
            transport_costs_matrix = time_distance_matrix.get_matrix()            

            if transport_costs_matrix is None:

                print("WARNING: No transport costs matrix was built. Probably ORS server error. Check output above and try again later.")

                error_message = "Error: No transport costs matrix was built. Probably ORS server error."
            
            else:

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

        interaction_matrix_metadata["transport_costs_col"] = config.DEFAULT_COLNAME_TC
        
        self.interaction_matrix_df = interaction_matrix_df
        self.metadata = interaction_matrix_metadata

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.transport_costs",
            process=f"Calculated transport costs ({range_type}, {time_unit})",
            status="OK" if error_message == "" else error_message
            )

        return self
    
    def set_attraction_constant(
        self,
        constant_value: int = 1,
        verbose: bool = config.VERBOSE
        ):

        """
        Sets attraction values of all supply locations in the interaction matrix
        to an constant attraction value.

        Parameters
        ----------
        constant_value : int
            Integer value of constant attraction (default: 1).
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        self : InteractionMatrix
            Updated object with new attraction values.

        Example
        -------
        >>> Freiburg_Stadtbezirke_SHP = gp.read_file("data/Freiburg_Stadtbezirke_Point.shp")
        >>> Freiburg_Stadtbezirke_Einwohner = pd.read_excel("data/Freiburg_Stadtbezirke_Einwohner.xlsx")
        >>> Freiburg_Stadtbezirke_Einwohner["nr"] = Freiburg_Stadtbezirke_Einwohner["nr"].astype(str)
        >>> Freiburg_Stadtbezirke = Freiburg_Stadtbezirke_SHP.merge(
        ...     Freiburg_Stadtbezirke_Einwohner[["nr", "EWU18"]],
        ...     left_on="nr",
        ...     right_on="nr"
        ... )
        >>> Freiburg_Stadtbezirke = load_geodata(
        ...     Freiburg_Stadtbezirke,
        ...     location_type="origins",
        ...     unique_id="name"
        ... )
        >>> Freiburg_Stadtbezirke.define_marketsize("EWU18")
        >>> Freiburg_KuJAerzte = load_geodata(
        ...     "data/Freiburg_KuJAerzte_Point.shp",
        ...     location_type="destinations",
        ...     unique_id="LfdNr"
        ... )
        >>> pediatricians_interactionmatrix = create_interaction_matrix(
        ...     Freiburg_Stadtbezirke,
        ...     Freiburg_KuJAerzte,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.transport_costs(
        ...     network=False,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.define_weightings(
        ...     vars_funcs={
        ...         0: {
        ...             "name": "A_j",
        ...             "func": "power",
        ...             "param": 1
        ...         },
        ...         1: {
        ...             "name": "t_ij",
        ...             "func": "power",
        ...             "param": -1
        ...             }
        ...         }
        ... )
        >>> pediatricians_interactionmatrix.set_attraction_constant()
        """
        
        if verbose:
            print(f"Setting default attraction value '{config.DEFAULT_COLNAME_ATTRAC}' equal to {constant_value} with {config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0]} function weighting and param = 1", end = " ... ")
        
        supply_locations_metadata = self.supply_locations.metadata
        
        supply_locations_metadata["attraction_col"] = [config.DEFAULT_COLNAME_ATTRAC]
        
        supply_locations_metadata["weighting"][0]["name"] = config.DEFAULT_COLNAME_ATTRAC
        supply_locations_metadata["weighting"][0]["func"] = "power"
        supply_locations_metadata["weighting"][0]["param"] = 1
        
        helper.add_timestamp(
            self.supply_locations,
            function="models.InteractionMatrix.set_attraction_default",
            process=f"Set default attraction value '{config.DEFAULT_COLNAME_ATTRAC}' equal to {constant_value} with {config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0]} function weighting and param = 1"
            )
        
        self.supply_locations.metadata = supply_locations_metadata    
    
        self.interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC] = constant_value
        
        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.set_attraction_constant",
            process=f"Set default attraction value '{config.DEFAULT_COLNAME_ATTRAC}' equal to {constant_value} with {config.PERMITTED_WEIGHTING_FUNCTIONS_LIST[0]} function weighting and param = 1"
            )
        
        if verbose:
            print("OK")

        return self
            
    def define_weightings(
        self,
        vars_funcs: dict
        ):

        """
        Sets attraction values of all supply locations in the interaction matrix
        to an constant attraction value.

        Parameters
        ----------
        vars_funcs : dict
            Dictionary with weightings functions and parameters for each variable.

        Returns
        -------
        self : InteractionMatrix
            Updated object with updated metadata of CustomerOrigins and SupplyLocations objects.

        Example
        -------
        >>> Freiburg_Stadtbezirke_SHP = gp.read_file("data/Freiburg_Stadtbezirke_Point.shp")
        >>> Freiburg_Stadtbezirke_Einwohner = pd.read_excel("data/Freiburg_Stadtbezirke_Einwohner.xlsx")
        >>> Freiburg_Stadtbezirke_Einwohner["nr"] = Freiburg_Stadtbezirke_Einwohner["nr"].astype(str)
        >>> Freiburg_Stadtbezirke = Freiburg_Stadtbezirke_SHP.merge(
        ...     Freiburg_Stadtbezirke_Einwohner[["nr", "EWU18"]],
        ...     left_on="nr",
        ...     right_on="nr"
        ... )
        >>> Freiburg_Stadtbezirke = load_geodata(
        ...     Freiburg_Stadtbezirke,
        ...     location_type="origins",
        ...     unique_id="name"
        ... )
        >>> Freiburg_Stadtbezirke.define_marketsize("EWU18")
        >>> Freiburg_KuJAerzte = load_geodata(
        ...     "data/Freiburg_KuJAerzte_Point.shp",
        ...     location_type="destinations",
        ...     unique_id="LfdNr"
        ... )
        >>> pediatricians_interactionmatrix = create_interaction_matrix(
        ...     Freiburg_Stadtbezirke,
        ...     Freiburg_KuJAerzte,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.transport_costs(
        ...     network=False,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.define_weightings(
        ...     vars_funcs={
        ...         0: {
        ...             "name": "A_j",
        ...             "func": "power",
        ...             "param": 1
        ...         },
        ...         1: {
        ...             "name": "t_ij",
        ...             "func": "power",
        ...             "param": -1
        ...             }
        ...         }
        ... )
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

        helper.add_timestamp(
            self.supply_locations,
            function="models.InteractionMatrix.define_weightings",
            process=f"Defined weightings for {len(vars_funcs)} variables"
            )

        helper.add_timestamp(
            self.customer_origins,
            function="models.InteractionMatrix.define_weightings",
            process=f"Defined weightings for {len(vars_funcs)} variables"
            )
        
        self.supply_locations.metadata = supply_locations_metadata
        self.customer_origins.metadata = customer_origins_metadata       

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.define_weightings",
            process=f"Defined weightings for {len(vars_funcs)} variables"
            )
        
        return self

    def utility(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        """
        Calculate utilities for all combinations of supply locations and customer origins.

        All partial utilies have their own specifing weighting function and parameters,
        as defined in the CustomerOrigins and SupplyLocations metadata.
        See Huff (1962) for the derivation of the Huff model utility function.
        See Wieland (2017) for the usage of different weighting functions.

        Parameters
        ----------
        check_df_vars : bool, default=True
            If True, the relevant variables (attraction and transport costs) are checked
            whether they are numeric, strictly positive (non-zero), and non-constant.
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        self
            Returns the instance with an updated utility column.

        Raises
        ------
        Exception
            If attraction and transport costs columns do not meet processing criteria.
        InteractionMatrixError
            If relevant variables are not defined or NaN.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> haslach_interactionmatrix.utility()
        """
        
        interaction_matrix_df = self.interaction_matrix_df

        interaction_matrix_metadata = self.get_metadata()

        error_messages = []

        if config.DEFAULT_COLNAME_TC not in interaction_matrix_df.columns:
            error_messages.append("No transport costs variable in interaction matrix.")
        if config.DEFAULT_COLNAME_ATTRAC not in interaction_matrix_df.columns:
            error_messages.append("No attraction variable in interaction matrix.")
        if interaction_matrix_df[config.DEFAULT_COLNAME_TC].isna().all():
            error_messages.append(f"{config.DEFAULT_NAME_TC} variable is not defined (nan).")
        if interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC].isna().all():
            error_messages.append(f"{config.DEFAULT_NAME_ATTRAC} variable is not defined (nan).")
            
        if len(error_messages) > 0:
            raise InteractionMatrixError(f"Utility calculation is not possible because of the following error(s): {' '.join(error_messages)}")

        if check_df_vars:
            try:
                helper.check_vars(
                    df = interaction_matrix_df,
                    cols = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC],
                    check_constant = False
                    )
            except Exception as e:
                raise InteractionMatrixError(f"Utility calculation is not possible because of the following error(s): {str(e)}")
        
        if verbose:
            print("Calculating utility", end = " ... ")
            
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

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.utility",
            process=f"Calculated utilities based on {attrac_vars_no} attraction variable(s) and transport costs"
            )
        
        self.interaction_matrix_df = interaction_matrix_df
        self.metadata = interaction_matrix_metadata
        
        if verbose:
            print("OK")

        return self
    
    def probabilities(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE        
        ):

        """
        Calculate interaction probabilities for all combinations of supply locations and customer origins.

        If no utility values have been calculated before, the method uses InteractionMatrix.utility()
        to calculate utility values.
        All partial utilies have their own specifing weighting function and parameters,
        as defined in the CustomerOrigins and SupplyLocations metadata.
        See Huff (1962) for the derivation of the Huff model utility function and the
        calculation of probabilties.

        Parameters
        ----------
        check_df_vars : bool, default=True
            If True, the relevant variables (attraction and transport costs) are checked
            whether they are numeric, strictly positive (non-zero), and non-constant.
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        self
            Returns the instance with an updated probabilities column.

        Raises
        ------
        Exception
            If utility column does not meet processing criteria.
        InteractionMatrixError
            If relevant variables are not defined or NaN.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> haslach_interactionmatrix.probabilities()
        """

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
               
        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.probabilities",
            process="Calculated probabilities"
            )
                
        if verbose:
            print("OK")

        return self
        
    def flows(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        """
        Calculate expected customer/expenditure flows for all combinations of supply locations and customer origins.

        The method requires a defined market size variable for the origins. 
        If no probabilities have been calculated before, the method uses 
        InteractionMatrix.probabilities() to calculate probabilities. 
        See Huff (1962) for the calculation of probabilties and expected customer flows.

        Parameters
        ----------
        check_df_vars : bool, default=True
            If True, the relevant variables (attraction and transport costs) are checked
            whether they are numeric, strictly positive (non-zero), and non-constant.
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        self
            Returns the instance with an updated flows column.

        Raises
        ------
        Exception
            If market size variable column does not meet processing criteria.
        InteractionMatrixError
            If no market size variable is defined or all values are NaN.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> haslach_interactionmatrix.flows()
        """

        interaction_matrix_df = self.interaction_matrix_df
        
        interaction_matrix_metadata = self.get_metadata()

        if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
            raise InteractionMatrixError("Error in flows calculation: No market size variable in interaction matrix")
        if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
            raise InteractionMatrixError(f"Error in flows calculation: {config.DEFAULT_NAME_MARKETSIZE} column in customer origins not defined. Use CustomerOrigins.define_marketsize()")

        if check_df_vars:
            helper.check_vars(
                df = interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_MARKETSIZE]
                )

        if interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY].isna().all():
            self.probabilities()
            interaction_matrix_df = self.interaction_matrix_df

        if verbose:
            print("Calculating expected customer flows", end = " ... ")

        interaction_matrix_df[config.DEFAULT_COLNAME_FLOWS] = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY] * interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE]

        self.interaction_matrix_df = interaction_matrix_df

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.flows",
            process="Calculated expected customer flows"
            )
        
        self.metadata = interaction_matrix_metadata
        
        if verbose:
            print("OK")

        return self

    def marketareas(
        self,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        """
        Calculate total market areas for the supply locations as the sum
        of the expected customer flows.
        See Huff (1964) for the calculation of total market areas.

        Parameters
        ----------
        check_df_vars : bool, default=True
            If True, the relevant variables (attraction and transport costs) are checked
            whether they are numeric, strictly positive (non-zero), and non-constant.
        verbose : bool, optional
            If True, print progress messages (default: False).

        Returns
        -------
        HuffModel
        An instance of class HuffModel including customer origins, supply locations,
        the interaction matrix and a pandas.DataFrame with total market areas.

        Raises
        ------
        Exception
            If market size variable column does not meet processing criteria.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> haslach_interactionmatrix.flows()
        >>> huff_model = haslach_interactionmatrix.marketareas()
        """

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

        metadata = {}
        
        if "fit" in self.metadata:
            metadata["fit"] = self.metadata["fit"]

        huff_model = HuffModel(
            self,
            market_areas_df,
            metadata = metadata
            )
        
        helper.add_timestamp(
            huff_model,
            function="models.InteractionMatrix.marketareas",
            process=f"Creation and calculation of total market areas for {len(market_areas_df)} locations"
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

        """
        Compute Hansen accessibility from an interaction matrix.

        This method calculates a Hansen accessibility index based on an
        interaction matrix, either from the perspective of customer origins
        (origin-based accessibility) or supply locations (destination-based
        accessibility). Depending on the chosen mode, the method may internally 
        compute utilities, apply transport-cost weighting functions, and exclude 
        self-interactions. See Hansen (1959) for the original formulation.

        Parameters
        ----------
        from_origins : bool, default=True
            If True, compute accessibility for customer origins.
            If False, compute accessibility for supply locations.
        exclude_self : bool, default=True
            If True, exclude self-interactions where customer origins and supply
            locations refer to the same spatial unit.
        check_df_vars : bool, default=True
            If True, validate required variables in the interaction matrix before
            computing utilities.
        verbose : bool, default=False
            If True, print progress information during the calculation.

        Returns
        -------
        HansenAccessibility
            An object containing the computed Hansen accessibility values (pandas.DataFrame),
            the interaction matrix, and calculation metadata.

        Raises
        ------
        InteractionMatrixError
            If required variables (e.g., market size or transport cost weighting)
            are missing or undefined for destination-based accessibility.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> print(haslach_interactionmatrix.hansen())
        """

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
            hansen_df = hansen_df.rename(columns = {config.DEFAULT_COLNAME_UTILITY: config.DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS})

        else:
            
            if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
                raise InteractionMatrixError(f"Error in {config.MODELS['Hansen']['description']} calculation: Interaction matrix does not contain market size variable")
            if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
                raise InteractionMatrixError(f"Error in {config.MODELS['Hansen']['description']} calculation: Customer origins market size is not available")
            
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
                raise InteractionMatrixError(f"Error in {config.MODELS['Hansen']['description']} calculation: {config.DEFAULT_NAME_TC} weighting is not defined.")
                        
            interaction_matrix_df[config.DEFAULT_COLNAME_UTILITY_SUPPLY] = interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE]*interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED]           
            hansen_df = pd.DataFrame(interaction_matrix_df.groupby(config.DEFAULT_COLNAME_SUPPLY_LOCATIONS)[config.DEFAULT_COLNAME_UTILITY_SUPPLY].sum()).reset_index()
            hansen_df = hansen_df.rename(columns = {config.DEFAULT_COLNAME_UTILITY_SUPPLY: config.DEFAULT_COLNAME_ATTRAC})
            
            interaction_matrix_df = self.interaction_matrix_df
            
            helper.add_timestamp(
                self,
                function = "models.InteractionMatrix.hansen",
                process = f"Calculated utilities from destinations in column '{config.DEFAULT_COLNAME_UTILITY_SUPPLY}'"
                )

        metadata = {
            "calculation": {
                "from_origins": from_origins,
                "exclude_self": exclude_self
            },
        }

        hansen_accessibility = HansenAccessibility(
            interaction_matrix = self,
            hansen_df = hansen_df,
            metadata = metadata
            )
        
        helper.add_timestamp(
            hansen_accessibility,
            function = "models.InteractionMatrix.hansen",
            process = f"Creation and calculation of {config.MODELS['Hansen']['description']} from customer origins" if from_origins else f"Creation and calculation of {config.MODELS['Hansen']['description']} from supply locations"
            )

        if verbose:
            print("OK")

        return hansen_accessibility
    
    def supply_to_demand_ratio(
        self,
        threshold: float,
        use_attraction: bool = True,
        use_weightings: bool = True,
        demand_factor: int = 1,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        """
        Compute a supply-to-demand ratio using a floating catchment approach.

        This method calculates a supply-to-demand ratio (SD ratio) for each
        supply location based on a distance or travel-cost threshold, following
        the Two-Step Floating Catchment Area (2SFCA) method (Luo and Wang 2003).
        Demand is represented by market size at customer origins, while supply
        is represented by attraction at supply locations. Optional weighting 
        functions for transport costs and attraction variables can be applied. 
        If attraction is set constant and no weighting is applied, the result
        equals the first step of the original 2SFCA method by Luo and Wang (2003).

        Parameters
        ----------
        threshold : float
            Maximum transport cost (e.g., distance or travel time).
        use_attraction : bool, default=True
            If True, use an attraction variable for supply. If no valid attraction
            values are available, attraction is set to a constant value of 1.
        use_weightings : bool, default=True
            If True, apply weighting functions for transport costs and attraction
            variables as defined in the customer origins and supply locations
            metadata. If False, all weightings are set to 1.
        demand_factor : int, default=1
            Scaling factor applied to demand when computing the supply-to-demand
            ratio. For example, set `demand_factor` to 1000 for physicians per
            1,000 inhabitants.
        check_df_vars : bool, default=True
            If True, validate the presence of required variables in the interaction
            matrix before calculation.
        verbose : bool, default=False
            If True, print progress information during the calculation.

        Returns
        -------
        InteractionMatrix
            The modified interaction matrix instance with the computed
            supply-to-demand ratio as new column.

        Raises
        ------
        InteractionMatrixError
            If required variables such as market size or attraction are missing
            or undefined in the interaction matrix.

        Example
        --------
        >>> Freiburg_Stadtbezirke_SHP = gp.read_file("data/Freiburg_Stadtbezirke_Point.shp")
        >>> Freiburg_Stadtbezirke_Einwohner = pd.read_excel("data/Freiburg_Stadtbezirke_Einwohner.xlsx")
        >>> Freiburg_Stadtbezirke_Einwohner["nr"] = Freiburg_Stadtbezirke_Einwohner["nr"].astype(str)
        >>> Freiburg_Stadtbezirke = Freiburg_Stadtbezirke_SHP.merge(
        ...    Freiburg_Stadtbezirke_Einwohner[["nr", "EWU18"]],
        ...     left_on="nr",
        ...     right_on="nr"
        ... )
        >>> Freiburg_Stadtbezirke = load_geodata(
        ...     Freiburg_Stadtbezirke,
        ...     location_type="origins",
        ...     unique_id="name"
        ... )
        >>> Freiburg_Stadtbezirke.define_marketsize("EWU18")
        >>> Freiburg_KuJAerzte = load_geodata(
        ...     "data/Freiburg_KuJAerzte_Point.shp",
        ...     location_type="destinations",
        ...     unique_id="LfdNr"
        ... )
        >>> pediatricians_interactionmatrix = create_interaction_matrix(
        ...     Freiburg_Stadtbezirke,
        ...     Freiburg_KuJAerzte,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.transport_costs(
        ...     network=False,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.define_weightings(
        ...     vars_funcs={
        ...     0: {
        ...             "name": "A_j",
        ...             "func": "power",
        ...             "param": 1
        ...         },
        ...     1: {
        ...             "name": "t_ij",
        ...             "func": "power",
        ...             "param": -1
        ...         }
        ...     }
        ... )
        >>> pediatricians_interactionmatrix.set_attraction_constant()
        >>> pediatricians_interactionmatrix.summary()
        >>> sd_ratio_calculation = pediatricians_interactionmatrix.supply_to_demand_ratio(
        ...     threshold=1,
        ...     demand_factor=1000
        ... )
        """

        interaction_matrix_df = self.interaction_matrix_df

        if config.DEFAULT_COLNAME_MARKETSIZE not in interaction_matrix_df.columns:
            raise InteractionMatrixError(f"Error in {config.MODELS['2SFCA']['description']}: Interaction matrix does not contain market size variable")
        if interaction_matrix_df[config.DEFAULT_COLNAME_MARKETSIZE].isna().all():
            raise InteractionMatrixError(f"Error in {config.MODELS['2SFCA']['description']}: Customer origins market size is not available")
        
        if check_df_vars:
            helper.check_vars(
                interaction_matrix_df,
                cols = [config.DEFAULT_COLNAME_MARKETSIZE]
            )

        interaction_matrix_df[config.DEFAULT_COLNAME_TC_WEIGHTED] = 1
        interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED] = 1
        
        if use_attraction:
            
            if config.DEFAULT_COLNAME_ATTRAC not in interaction_matrix_df.columns:
                raise InteractionMatrixError(f"Error in {config.MODELS['2SFCA']['description']}: Interaction matrix does not contain attraction variable {config.DEFAULT_COLNAME_ATTRAC}")
            if interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC].isna().all():
                print(f"NOTE: No attraction definition in {config.MODELS['2SFCA']['description']}. Set attraction as constant = 1.")
                
            else:
                
                interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED] = interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC]
                
                helper.add_timestamp(
                    self,
                    function = "models.InteractionMatrix.supply_to_demand_ratio",
                    process = f"Set {config.DEFAULT_COLNAME_ATTRAC_WEIGHTED} as attraction variable"
                    )
                
                if verbose and not use_weightings:
                    print(f"NOTE: Using {config.DEFAULT_COLNAME_ATTRAC} as attraction variable in {config.MODELS['2SFCA']['description']}.")

        attrac_process = f"Set attraction weighting as constant = 1"
        tc_process = "Set transport costs weighting as constant = 1"

        if use_weightings:

            customer_origins_metadata = self.customer_origins.get_metadata()            
            
            tc_weighting = customer_origins_metadata["weighting"][0]    

            if tc_weighting["func"] in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
                
                tc_process = f"Calculated weighted transport costs with {tc_weighting['func']} function weighting"
                
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
                    
            else:
                
                print (f"NOTE: No {config.DEFAULT_NAME_TC} weighting in {config.MODELS['2SFCA']['description']}. Set transport costs weighting as constant = 1.")
                tc_process = "Set transport costs weighting as constant = 1"
            
            supply_locations_metadata = self.supply_locations.get_metadata()
            
            attraction_weighting = supply_locations_metadata["weighting"][0]

            if attraction_weighting is None:
                
                print (f"NOTE: No {config.DEFAULT_NAME_ATTRAC} weighting in {config.MODELS['2SFCA']['description']}. Set {config.DEFAULT_NAME_ATTRAC} as constant = 1.")            
                attrac_process = f"Set {config.DEFAULT_NAME_ATTRAC} as constant = 1"

            else:

                attrac_vars = supply_locations_metadata["attraction_col"]
                attrac_vars_no = len(attrac_vars)
                attrac_var_key = 0

                if attrac_vars_no >= 1:
                    
                    for key, attrac_var in enumerate(attrac_vars):
                        
                        attrac_var_key = key
                        
                        name = supply_locations_metadata["weighting"][attrac_var_key]["name"]
                        param = supply_locations_metadata["weighting"][attrac_var_key]["param"]
                        func = supply_locations_metadata["weighting"][attrac_var_key]["func"]

                        interaction_matrix_df[name+config.DEFAULT_WEIGHTED_SUFFIX] = weighting(
                            values = interaction_matrix_df[name],
                            func = func,
                            b = param
                            )
                        
                        interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED] = interaction_matrix_df[config.DEFAULT_COLNAME_ATTRAC_WEIGHTED]*interaction_matrix_df[name+config.DEFAULT_WEIGHTED_SUFFIX]

                        if name != config.DEFAULT_COLNAME_ATTRAC:
                            interaction_matrix_df = interaction_matrix_df.drop(columns=[name+config.DEFAULT_WEIGHTED_SUFFIX])
                
                attrac_process = f"Calculated weighted attraction with {attrac_vars_no} attraction variable(s): {', '.join(attrac_vars)}"
        
        else:

            if verbose:
                print(f"No weightings for attraction and transport costs. Both are set constant = 1")
                
        helper.add_timestamp(
            self,
            function = "models.InteractionMatrix.supply_to_demand_ratio",
            process = attrac_process
            )
        
        helper.add_timestamp(
            self,
            function = "models.InteractionMatrix.supply_to_demand_ratio",
            process = tc_process
            )
        
        if verbose:
            print(f"Calculating {config.DEFAULT_NAME_SD_RATIO}", end = " ... ")

        interaction_matrix_df_max = interaction_matrix_df[interaction_matrix_df[config.DEFAULT_COLNAME_TC] <= threshold].copy()

        supply_locations_R_j = interaction_matrix_df[[config.DEFAULT_COLNAME_SUPPLY_LOCATIONS, config.DEFAULT_COLNAME_ATTRAC]].drop_duplicates()

        P_i = interaction_matrix_df_max.groupby(config.DEFAULT_COLNAME_SUPPLY_LOCATIONS)[config.DEFAULT_COLNAME_MARKETSIZE].sum()

        supply_locations_R_j = supply_locations_R_j.merge(
            P_i,
            left_on=config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
            right_on=config.DEFAULT_COLNAME_SUPPLY_LOCATIONS
        )

        supply_locations_R_j[config.DEFAULT_COLNAME_SD_RATIO] = supply_locations_R_j[config.DEFAULT_COLNAME_ATTRAC]/supply_locations_R_j[config.DEFAULT_COLNAME_MARKETSIZE]*demand_factor

        supply_locations_R_j = supply_locations_R_j.drop(
            columns = {
                config.DEFAULT_COLNAME_MARKETSIZE,
                config.DEFAULT_COLNAME_ATTRAC
            }
        )

        interaction_matrix_df = interaction_matrix_df.merge(
            supply_locations_R_j,
            left_on = config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
            right_on=config.DEFAULT_COLNAME_SUPPLY_LOCATIONS,
            how="left"
        )

        interaction_matrix_df[config.DEFAULT_COLNAME_SD_RATIO] = interaction_matrix_df[config.DEFAULT_COLNAME_SD_RATIO].fillna(0)

        self.interaction_matrix_df = interaction_matrix_df

        if verbose:
            print("OK")

        helper.add_timestamp(
            self,
            function = "models.InteractionMatrix.supply_to_demand_ratio",
            process = f"Calculated {config.DEFAULT_NAME_SD_RATIO} with threshold = {threshold} and weightings" if use_weightings else f"Calculated {config.DEFAULT_NAME_SD_RATIO} with threshold = {threshold} without weightings"
            )
        
        return self

    def floating_catchment(
        self,
        threshold: float,
        use_attraction: bool = True,
        use_weightings: bool = True,
        demand_factor: int = 1,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):

        """
        Compute a Two-Step Floating Catchment Area (2SFCA) analysis.

        This method calculates 2SFCA accessibility for each customer origin
        based on a distance or travel-cost threshold, following the Two-Step 
        Floating Catchment Area (2SFCA) method (Luo and Wang 2003).
        Demand is represented by market size at customer origins, while supply
        is represented by attraction at supply locations. Optional weighting 
        functions for transport costs and attraction variables can be applied. 
        If attraction is set constant and no weighting is applied, the result
        equals the the original 2SFCA method by Luo and Wang (2003).

        Parameters
        ----------
        threshold : float
            Maximum transport cost (e.g., distance or travel time).
        use_attraction : bool, default=True
            If True, use an attraction variable for supply. If no valid attraction
            values are available, attraction is set to a constant value of 1.
        use_weightings : bool, default=True
            If True, apply weighting functions for transport costs and attraction
            variables as defined in the customer origins and supply locations
            metadata. If False, all weightings are set to 1.
        demand_factor : int, default=1
            Scaling factor applied to demand when computing the supply-to-demand
            ratio. For example, set `demand_factor` to 1000 for physicians per
            1,000 inhabitants.
        check_df_vars : bool, default=True
            If True, validate the presence of required variables in the interaction
            matrix before calculation.
        verbose : bool, default=False
            If True, print progress information during the calculation.

        Returns
        -------
        FloatingCatchment
            An object containing the computed accessibility values (pandas.DataFrame),
            the modified interaction matrix, and calculation metadata.

        Raises
        ------
        InteractionMatrixError
            If required variables such as market size or attraction are missing
            or undefined in the interaction matrix.

        Example
        --------
        >>> Freiburg_Stadtbezirke_SHP = gp.read_file("data/Freiburg_Stadtbezirke_Point.shp")
        >>> Freiburg_Stadtbezirke_Einwohner = pd.read_excel("data/Freiburg_Stadtbezirke_Einwohner.xlsx")
        >>> Freiburg_Stadtbezirke_Einwohner["nr"] = Freiburg_Stadtbezirke_Einwohner["nr"].astype(str)
        >>> Freiburg_Stadtbezirke = Freiburg_Stadtbezirke_SHP.merge(
        ...    Freiburg_Stadtbezirke_Einwohner[["nr", "EWU18"]],
        ...     left_on="nr",
        ...     right_on="nr"
        ... )
        >>> Freiburg_Stadtbezirke = load_geodata(
        ...     Freiburg_Stadtbezirke,
        ...     location_type="origins",
        ...     unique_id="name"
        ... )
        >>> Freiburg_Stadtbezirke.define_marketsize("EWU18")
        >>> Freiburg_KuJAerzte = load_geodata(
        ...     "data/Freiburg_KuJAerzte_Point.shp",
        ...     location_type="destinations",
        ...     unique_id="LfdNr"
        ... )
        >>> pediatricians_interactionmatrix = create_interaction_matrix(
        ...     Freiburg_Stadtbezirke,
        ...     Freiburg_KuJAerzte,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.transport_costs(
        ...     network=False,
        ...     verbose=True
        ... )
        >>> pediatricians_interactionmatrix.define_weightings(
        ...     vars_funcs={
        ...     0: {
        ...             "name": "A_j",
        ...             "func": "power",
        ...             "param": 1
        ...         },
        ...     1: {
        ...             "name": "t_ij",
        ...             "func": "power",
        ...             "param": -1
        ...         }
        ...     }
        ... )
        >>> pediatricians_interactionmatrix.set_attraction_constant()
        >>> pediatricians_interactionmatrix.summary()
        >>> sd_ratio_calculation = pediatricians_interactionmatrix.floating_catchment(
        ...     threshold=1,
        ...     demand_factor=1000
        ... )
        """
        
        self.supply_to_demand_ratio(
            threshold=threshold,
            use_attraction=use_attraction,
            use_weightings=use_weightings,
            demand_factor=demand_factor,
            verbose=verbose,
            check_df_vars=check_df_vars
        )
        
        if verbose:
            print(f"Calculating {config.MODELS['2SFCA']['description']}", end = " ... ")

        interaction_matrix_df = self.interaction_matrix_df

        interaction_matrix_df_max = interaction_matrix_df[interaction_matrix_df[config.DEFAULT_COLNAME_TC] <= threshold].copy()

        customer_origins_A_i = interaction_matrix_df_max.groupby(config.DEFAULT_COLNAME_CUSTOMER_ORIGINS)[config.DEFAULT_COLNAME_SD_RATIO].sum().reset_index(name=config.DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS)      

        interaction_matrix_df = interaction_matrix_df.merge(
            customer_origins_A_i,
            left_on = config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            right_on=config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            how="left"
        )

        interaction_matrix_df[config.DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS] = interaction_matrix_df[config.DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS].fillna(0)
        
        self.interaction_matrix_df = interaction_matrix_df

        helper.add_timestamp(
            self,
            function = "models.InteractionMatrix.floating_catchment",
            process = f"Calculated {config.MODELS['2SFCA']['description']} with threshold = {threshold} and weightings" if use_weightings else f"Calculated {config.MODELS['2SFCA']['description']} with threshold = {threshold} without weightings"
            )
        
        if customer_origins_A_i[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].any() not in interaction_matrix_df[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].unique():
            
            customer_origins_apart_threshold = interaction_matrix_df[~interaction_matrix_df[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].isin(customer_origins_A_i[config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].unique())][config.DEFAULT_COLNAME_CUSTOMER_ORIGINS].unique()            
            
            customer_origins_A_i_0 = pd.DataFrame(
                {
                    config.DEFAULT_COLNAME_CUSTOMER_ORIGINS: customer_origins_apart_threshold,
                    config.DEFAULT_COLNAME_ACCESSIBILITY_ORIGINS: 0
                }
            )
            
            customer_origins_A_i = pd.concat(
                [
                    customer_origins_A_i, 
                    customer_origins_A_i_0
                    ],
                ignore_index=True
                )
            
        customer_origins_A_i = customer_origins_A_i.sort_values(by = config.DEFAULT_COLNAME_CUSTOMER_ORIGINS)

        metadata = {
            "calculation": {
                "threshold": threshold,
                "use_weightings": use_weightings
            },
        }

        floating_catchment_analysis = FloatingCatchment(
            interaction_matrix = self,
            fca_df = customer_origins_A_i,
            metadata = metadata
            )
        
        if verbose:        
            print("OK")
            
        if len(customer_origins_apart_threshold) > 0:
            print(f"WARNING: There are {len(customer_origins_apart_threshold)} customer origins apart from the threshold value of {threshold}: {', '.join(str(customer_origin) for customer_origin in customer_origins_apart_threshold)}.")
            
        helper.add_timestamp(
            floating_catchment_analysis,
            function = "models.InteractionMatrix.floating_catchment",
            process = f"Creation and calculation of {config.MODELS['2SFCA']['description']} with threshold = {threshold} and weightings" if use_weightings else f"Creation and calculation of {config.MODELS['2SFCA']['description']} with threshold = {threshold} without weightings"
            )            

        return floating_catchment_analysis

    def mci_transformation(
        self,
        cols: list = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC],
        verbose: bool = config.VERBOSE
        ):

        """
        Apply a log-centering transformation to selected columns of the interaction
        matrix.

        This method performs a log-centering transformation on the specified metric
        columns and on the probability column. See Nakanishi and Cooper (1974)
        for the derivation of the log-centering transformation. Internally, the method
        uses the `models.log_centering_transformation()` function.

        Parameters
        ----------
        cols : list of str, optional
            Names of the columns to which the log-centering transformation is
            applied. By default, the attractiveness and transport cost columns
            defined in the configuration are used.
        verbose : bool, optional
            If True, print progress information during the calculation.

        Returns
        -------
        InteractionMatrix
            The current instance with the updated interaction matrix.

        Raises
        ------
        KeyError
            If any stated column is not in the interaction matrix DataFrame.

        Example
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
        >>> Wieland2015_interaction_matrix.mci_transformation(
        ...     cols = [
        ...         "VF", 
        ...         "K", 
        ...         "K_KKr",
        ...         "Dist_Min2"
        ...     ]
        ... )
        """

        if verbose:
            print(f"Processing log-centering transformation of columns: {', '.join(cols)}", end = " ... ")

        cols = cols + [config.DEFAULT_COLNAME_PROBABILITY]

        interaction_matrix_df = self.interaction_matrix_df

        interaction_matrix_df = log_centering_transformation(
            df = interaction_matrix_df,
            ref_col = config.DEFAULT_COLNAME_CUSTOMER_ORIGINS,
            cols = cols
            )
        
        self.interaction_matrix_df = interaction_matrix_df       

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.mci_transformation",
            process=f"Log-centering transformation of columns: {', '.join(cols)}"
            )

        if verbose:
            print("OK")
            
        return self

    def mci_fit(
        self,
        cols: list = None,
        alpha = 0.05,
        verbose: bool = config.VERBOSE
        ):

        """
        Estimate a Multiplicative Competitive Interaction (MCI) model.

        This method fits an MCI model on the interaction matrix, using the
        log-centered explanatory variables. If the required log-centered 
        variables are not yet available, the corresponding transformation 
        is performed automatically. See Nakanishi and Cooper (1974, 1982),
        Huff and McCallum (2008), or Wieland (2017) for the steps of
        fitting the MCI model.

        Parameters
        ----------
        cols : list of str, optional
            Names of the explanatory variables to be included in the MCI model.
            By default, the attractiveness and transport cost columns defined in
            the configuration are used.
        alpha : float, optional
            Significance level used to compute confidence intervals for the
            estimated coefficients.
        verbose : bool, optional
            If True, print progress information during the estimation.

        Returns
        -------
        MCIModel
            An object containing the fitted MCI model, estimated coefficients,
            confidence intervals, and the updated interaction matrix with
            associated metadata.

        Example
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
        >>> mci_fit = Wieland2015_interaction_matrix.mci_fit(
        ...     cols=[
        ...         "A_j", 
        ...         "t_ij", 
        ...         "K", 
        ...         "K_KKr"
        ...         ]
        ...     )
        >>> mci_fit.summary()
        """

        if cols is None:
            cols = [config.DEFAULT_COLNAME_ATTRAC, config.DEFAULT_COLNAME_TC]

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
            print(f"Performing {config.MODELS['MCI']['description']} estimation with variables: {', '.join(cols)}", end = " ... ")
                
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
                "Coefficient": col[:-len(config.DEFAULT_LCT_SUFFIX)],
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
        
        if self.metadata is None:
        
            interaction_matrix_metadata = {
                "fit": {
                    "function": "mci_fit",
                    "fit_by": "probabilities",
                    "method": "OLS"
                    }
                }
            
        else:
            
            interaction_matrix_metadata = self.metadata

            interaction_matrix_metadata["fit"] = {
                "function": "mci_fit",
                "fit_by": "probabilities",
                "method": "OLS"
                }
               
        interaction_matrix = InteractionMatrix(
            interaction_matrix_df,
            customer_origins,
            supply_locations,
            metadata=interaction_matrix_metadata
            )
        
        helper.add_timestamp(
            interaction_matrix,
            function="models.InteractionMatrix.mci_fit",
            process=f"Performed {config.MODELS['MCI']['description']} estimation with variables: {', '.join(cols)}"
            )
        
        metadata = {}
        
        mci_model = MCIModel(
            interaction_matrix,
            coefs,
            mci_ols_model,
            None,
            metadata
            )
        
        helper.add_timestamp(
            mci_model,
            function="models.InteractionMatrix.mci_fit",
            process=f"Creation and {config.MODELS['MCI']['description']} estimation with variables: {', '.join(cols)}"
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

        """
        Compute the negative log-likelihood of the Huff model.

        This method evaluates the Huff model fit by computing the (negative)
        log-likelihood based on either observed probabilities or observed flows.
        Model parameters are supplied as a vector and mapped to the corresponding
        weighting functions of customer origins and supply locations. The method
        internally recomputes utilities, probabilities, and flows using the given
        parameters before evaluating goodness-of-fit. See Orpana and Lampinen
        (2003) for the Maximum Likelihood Estimation of the Huff model.
        This method does not modify the interaction matrix.

        Parameters
        ----------
        params : list of float or numpy.ndarray
            Vector of model parameters. The first parameter corresponds to the
            attraction weighting parameter, the second to the transport cost
            parameter. Additional parameters are interpreted as location-
            specific weighting parameters. When logistic transport cost weighting
            is used, at least three parameters are required.
        fit_by : {"probabilities", "flows"}, optional
            Specifies whether the log-likelihood is computed using probabilities
            or flows.
        check_df_vars : bool, optional
            If True, checks for the presence and consistency of required DataFrame
            variables before computing utilities, probabilities, and flows.

        Returns
        -------
        float
            The negative log-likelihood value of the model.

        Raises
        ------
        ValueError
            If `fit_by` is not one of the supported options or if an insufficient
            number of parameters is provided for the selected weighting functions.
        TypeError
            If `params` is not provided as a list or NumPy array.

        Example
        --------
        >>> ll = haslach_interactionmatrix.loglik([1, -2], fit_by="probabilities")
        >>> print(ll)
        """
        
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

        """
        Estimate a Huff model via Maximum Likelihood optimization.

        This method performs Maximum Likelihood Estimation of the Huff model
        parameters by minimizing the negative log-likelihood using
        ``scipy.optimize.minimize``. The optimization is carried out with respect
        to attraction and transport cost weighting parameters, optionally
        including location-specific attraction parameters. The fitted parameters
        are written back to the metadata of customer origins and supply locations.
        Optionally, utilities, probabilities, and flows are recomputed using the
        estimated parameters. See Orpana and Lampinen (2003) for the Maximum 
        Likelihood Estimation of the Huff model.

        Parameters
        ----------
        initial_params : list of float, optional
            Initial values for the optimization parameters. The length of this
            list must match the number of free parameters implied by the selected
            weighting functions for customer origins and supply locations.
        method : str, optional
            Optimization algorithm passed to ``scipy.optimize.minimize`` (e.g.,
            "L-BFGS-B"). See the documentation of ``scipy.optimize.minimize``
            with respect to available solvers.
        bounds : list of tuple, optional
            Bounds on the parameters for the optimization. Must have the same
            length as ``initial_params``.
        constraints : list, optional
            Constraints passed to ``scipy.optimize.minimize``.
        fit_by : {"probabilities", "flows"}, optional
            Specifies whether the likelihood is evaluated using probabilities or
            flows.
        update_estimates : bool, optional
            If True, recomputes utilities, probabilities, and flows using the
            fitted parameters after successful optimization.
        check_df_vars : bool, optional
            If True, checks for the presence and consistency of required DataFrame
            variables during likelihood evaluation.
        verbose : bool, optional
            If True, prints progress messages and optimization status to the
            console.

        Returns
        -------
        InteractionMatrix
            The current instance with updated model parameters, metadata, and,
            optionally (if update_estimates=True), updated utilities, 
            probabilities, and flows in the interaction matrix DataFrame.

        Raises
        ------
        ValueError
            If the length of ``initial_params`` or ``bounds`` does not match the
            number of parameters implied by the model specification.

        Example
        --------
        >>> Wieland2015_interaction_matrix2 = load_interaction_matrix(
        ...     data="data/Wieland2015.xlsx",
        ...     customer_origins_col="Quellort",
        ...     supply_locations_col="Zielort",
        ...     attraction_col=[
        ...         "VF", 
        ...         "K", 
        ...         "K_KKr"
        ...         ],
        ...     market_size_col="Sum_Ek",
        ...     flows_col="Anb_Eink",
        ...     transport_costs_col="Dist_Min2",
        ...     probabilities_col="MA_Anb",
        ...     data_type="xlsx",
        ...     xlsx_sheet="interactionmatrix",
        ...     check_df_vars=False
        ...     )
        >>> Wieland2015_interaction_matrix2.define_weightings(
        ...     vars_funcs = {
        ...         0: {
        ...             "name": "A_j",
        ...             "func": "power"
        ...         },
        ...         1: {
        ...             "name": "t_ij",
        ...             "func": "power",              
        ...         },
        ...         2: {
        ...             "name": "K",
        ...             "func": "power"
        ...         },
        ...         3: {
        ...             "name": "K_KKr",
        ...             "func": "power"
        ...         }
        ...         }
        ...     )
        >>> Wieland2015_interaction_matrix2.huff_ml_fit(
        ...     initial_params=[0.9, -1.5, 0.5, 0.3],
        ...     bounds=[(0.5, 1), (-2, -1), (0.2, 0.7), (0.2, 0.7)],
        ...     fit_by="probabilities",
        ...     method="trust-constr",        
        ... )
        >>> Wieland2015_interaction_matrix2.summary()
        """
        
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
                    
            helper.add_timestamp(
                self.supply_locations,
                function="models.InteractionMatrix.huff_ml_fit"
                )
            
            helper.add_timestamp(
                self.customer_origins,
                function="models.InteractionMatrix.huff_ml_fit"
                )
            
            if verbose:
                print("OK")
                print(f"Optimization via {method} algorithm succeeded with parameters: {', '.join(str(round(par, 3)) for par in param_results)}.")
   
        else:
            if verbose:
                print("OK")
                
            helper.add_timestamp(
                self.supply_locations,
                function="models.InteractionMatrix.huff_ml_fit",
                status = f"Error: Optimiziation via {method} algorithm failed"
                )
            
            helper.add_timestamp(
                self.customer_origins,
                function="models.InteractionMatrix.huff_ml_fit",
                status = f"Error: Optimiziation via {method} algorithm failed"
                )
            
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

                helper.add_timestamp(
                    self,
                    function="models.InteractionMatrix.huff_ml_fit",
                    status = "Error: No update of estimates because fit parameters contain NaN"
                    )

            else:

                self = self.utility(verbose=False)
                self = self.probabilities(verbose=False)
                self = self.flows(verbose=False)            

                helper.add_timestamp(
                    self,
                    function="models.InteractionMatrix.huff_ml_fit",
                    process="Calculcated utilities, probabilities, expected customer flows"
                    )
        
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

        """
        Update attraction values for selected supply locations in the 
        interaction matrix.

        This method replaces attraction values for one or more supply locations by
        updating the specified attraction columns in the interaction matrix
        DataFrame. Each update is defined by a location identifier, the name of
        the attraction column, and the new attraction value.

        Parameters
        ----------
        new_attraction_values : dict
            Dictionary defining the attraction updates. Each entry must contain
            the keys ``"location"``, ``"attraction_col"``, and ``"new_value"``,
            specifying the supply location identifier, the attraction column to be
            modified, and the new attraction value, respectively.
        verbose : bool, optional
            If True, prints a summary of the performed updates to the console.

        Returns
        -------
        InteractionMatrix
            The current instance with updated attraction values, enabling method
            chaining.

        Raises
        ------
        KeyError
            If required keys are missing from an entry in ``new_attraction_values``
            or if the specified attraction column does not exist in the
            interaction matrix.

        Example
        --------
        >>> updates = {
        ...     "new_val": {
        ...         "location": "LFDNR", 
        ...         "attraction_col": "VKF_qm", 
        ...         "new_value": 12345
        ...         }
        ... }
        >>> haslach_interactionmatrix.change_attraction_values(updates)
        """

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

        helper.add_timestamp(
            self,
            function="models.InteractionMatrix.change_attraction_values",
            process=f"Set new attraction values for {len(new_attraction_values)} locations"
            )
               
        if verbose:
            print(f"Set new attraction values for {len(new_attraction_values)} locations")

        return self

    def update(
        self,
        verbose: bool = config.VERBOSE
        ):

        """
        Update the interaction matrix with newly added supply locations.

        The method checks for supply locations marked as new (update flag),
        builds an interaction matrix for those, merges them into the existing
        interaction matrix, and recomputes transport costs and derived
        quantities where appropriate.

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress messages (default: config.VERBOSE).

        Returns
        -------
        InteractionMatrix
            The updated InteractionMatrix instance.

        Raises
        ------
        ValueError
            If no new destinations are present when an update is requested.
        """

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
  
            helper.add_timestamp(
                self,
                function="models.InteractionMatrix.update",
                process="Update of interaction matrix, no transport costs calculation"
                )        
            
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
            
            helper.add_timestamp(
                self,
                function="models.InteractionMatrix.update",
                process="Update of interaction matrix"
                )
            
            if verbose:
                print("Interaction matrix was updated")

        return self
    
    def plot(
        self,
        origin_point_style: dict = {},
        location_point_style: dict = {},
        line_color: str = "black",
        line_alpha: float = 0.7,
        line_size_by: str = "flows",
        log_line_size: bool = True,
        save_output: bool = True,
        output_filepath: str = "interaction_matrix.png",
        output_dpi: int = 300,
        zoom: int = 15,
        legend: bool = True,
        map_title: str = "Map of interaction matrix with OSM basemap",
        verbose: bool = False
        ):

        """
        Plot interaction flows from the interaction matrix on an OSM basemap.

        Creates a map with customer origin points, supply location points and
        flow lines whose width is scaled by either predicted probabilities or
        flows. The function returns the rendered map together with the layer
        objects and the styles used for plotting.

        Parameters
        ----------
        origin_point_style : dict, optional
            Style dictionary for origin points (name, color, alpha, size).
        location_point_style : dict, optional
            Style dictionary for supply location points (name, color, alpha, size).
        line_color : str, optional
            Color used for flow lines.
        line_alpha : float, optional
            Alpha transparency for flow lines.
        line_size_by : {"probabilities", "flows"}, optional
            Field used to scale line widths. Must be either "probabilities" or
            "flows".
        log_line_size : bool, optional
            If True, apply log-transform to the chosen line-size field.
        save_output : bool, optional
            If True, save the map to `output_filepath`.
        output_filepath : str, optional
            File path to save the rendered map image.
        output_dpi : int, optional
            DPI to use when saving the map.
        zoom : int, optional
            Initial zoom level for the basemap.
        legend : bool, optional
            If True, include a legend on the map (where supported).
        map_title : str, optional
            Title for the map output.
        verbose : bool, optional
            If True, print progress messages.

        Returns
        -------
        list
            A list containing `[map_osm, layers_to_plot, layer_styles]` where
            `map_osm` is the map object returned by `map_with_basemap`,
            `layers_to_plot` is a list of GeoDataFrames/layers, and
            `layer_styles` is the dictionary of styles used.

        Raises
        ------
        ValueError
            If `line_size_by` is not one of `"probabilities"` or `"flows"`.

        Example
        --------
        >>> Haslach = load_geodata(
        ...     "data/Haslach.shp",
        ...     location_type="origins",
        ...     unique_id="BEZEICHN"
        ... )
        >>> Haslach.define_transportcosts_weighting(
        ...     func = "power",
        ...     param_lambda = -2.2
        ... )
        >>> Haslach_supermarkets = load_geodata(
        ...     "data/Haslach_supermarkets.shp",
        ...     location_type="destinations",
        ...     unique_id="LFDNR"
        ... )
        >>> Haslach_supermarkets.define_attraction("VKF_qm")
        ... Haslach_supermarkets.define_attraction_weighting(param_gamma=0.9)
        >>> haslach_interactionmatrix = create_interaction_matrix(
        ...     Haslach,
        ...     Haslach_supermarkets
        ... )
        >>> haslach_interactionmatrix.transport_costs(
        ...     network=False,
        ...     distance_unit="meters"
        ... )
        >>> haslach_interactionmatrix.flows()
        >>> haslach_interactionmatrix.plot(
        ...     origin_point_style = {
        ...         "name": "Districts",
        ...         "color": "black",
        ...         "alpha": 1,
        ...         "size": 100,
        ...         },
        ...     location_point_style = {
        ...         "name": "Supermarket chains",
        ...         "color": {
        ...             "Name": {
        ...                 "Aldi Süd": "blue",
        ...                 "Edeka": "yellow",
        ...                 "Lidl": "red",
        ...                 "Netto": "orange",
        ...                 "Real": "darkblue",
        ...                 "Treff 3000": "fuchsia"
        ...                 }
        ...             },
        ...         "alpha": 1,
        ...         "size": 100
        ...         },
        ... )
        """

        if line_size_by not in ["probabilities", "flows"]:
            raise ValueError(f"Parameter 'line_size_by' must be one of the following: probabilities, flows.")
        
        if verbose:
            print("Checking customer origins and supply locations for geometry", end = " ... ")
            
        customer_origins = self.get_customer_origins()
        customer_origins_gdf = customer_origins.get_geodata_gpd_original() 
        customer_origins_metadata = customer_origins.get_metadata()
        customer_origins_uid = customer_origins_metadata["unique_id"]

        supply_locations = self.get_supply_locations()
        supply_locations_gdf = supply_locations.get_geodata_gpd_original()
        supply_locations_metadata = supply_locations.get_metadata()
        supply_locations_uid = supply_locations_metadata["unique_id"]

        interaction_matrix_df = self.get_interaction_matrix_df()

        geometry_errors = []       

        if "geometry" not in customer_origins_gdf.columns:
            geometry_errors.append("Customer origins lack geometry.")
        if "geometry" not in supply_locations_gdf.columns:
            geometry_errors.append("Supply locations lack geometry.")

        customer_origins_crs = customer_origins_metadata["crs_input"]
            
        if verbose:
            print("OK")
            
        if len(geometry_errors) > 0:
            
            print(f"No map plot of InteractionMatrix object possible because of the following error(s): {' '.join(geometry_errors)}")            
            
            return None
        
        if verbose:
            print(f"Constructing flow lines for {len(customer_origins_gdf)} customer origins and {len(supply_locations_gdf)} supply locations", end = " ... ")

        distance_matrix_results = distance_matrix_from_gdf(
            sources_points_gdf = customer_origins_gdf,
            sources_uid_col = customer_origins_uid,
            destinations_points_gdf = supply_locations_gdf,
            destinations_uid_col = supply_locations_uid,
            distance_type = "euclidean",
            unit = "m",
            remove_duplicates = True,
            save_output = False,
            output_crs = customer_origins_crs,
            verbose = False
            )
        
        flows_line_layer = distance_matrix_results[1]

        if line_size_by == "probabilities":
            line_size_col = config.DEFAULT_COLNAME_PROBABILITY
        else:
            line_size_col = config.DEFAULT_COLNAME_FLOWS
            
        if log_line_size:
            interaction_matrix_df[line_size_col] = np.log(interaction_matrix_df[line_size_col]+0.01)

        flows_line_layer = flows_line_layer.merge(
            interaction_matrix_df[[config.DEFAULT_COLNAME_INTERACTION, line_size_col]],
            left_on = f"{config.MATRIX_COL_SOURCE}{config.MATRIX_OD_SEPARATOR}{config.MATRIX_COL_DESTINATION}{config.DEFAULT_UNIQUE_ID_SUFFIX}",
            right_on = config.DEFAULT_COLNAME_INTERACTION
        )

        if verbose:
            print("OK")
            print("Compiling layers and layer styles", end = " ... ")

        flows_line_style = {
            "name": "Customer flows",
            "color": line_color,
            "alpha": line_alpha,
            "linewidth": {
                "width_col": line_size_col
            }
        }

        layers_to_plot = [            
            customer_origins_gdf, 
            supply_locations_gdf,
            flows_line_layer,
            ]
                
        layers_to_plot_included = [            
            "Customer origins points",
            "Supply locations points",
            "Expected flows",
            ] 

        if verbose:
            print("OK")
            print(f"NOTE: InteractionMatrix object contains {len(layers_to_plot)} layers to plot: {', '.join(layers_to_plot_included)}.")
      
        layer_styles = {            
            0: origin_point_style,
            1: location_point_style,
            2: flows_line_style,
            }

        assert len(layers_to_plot) == len(layer_styles), f"Error while trying to plot customer origins: There are {len(layers_to_plot)} layers to plot but {len(layer_styles)} plot styles were stated." 

        map_osm = map_with_basemap(
            layers = layers_to_plot,
            styles = layer_styles,
            save_output = save_output,
            output_filepath = output_filepath,
            output_dpi = output_dpi,
            legend = legend,
            map_title = map_title,
            zoom = zoom,
            verbose = verbose
            )
        
        return [
            map_osm,
            layers_to_plot,
            layer_styles
            ]     

class MarketAreas:
    
    """
    Container for total market area results.

    Computed market areas per supply location and associated metadata.
    Can be integrated into Huff or MCI model objects.

    Parameters
    ----------
    market_areas_df : pandas.DataFrame
        DataFrame containing total market areas for each supply location.
    metadata : dict
        Metadata.
    """

    def __init__(
        self, 
        market_areas_df,        
        metadata
        ):

        self.market_areas_df = market_areas_df
        self.metadata = metadata

    def get_market_areas_df(self):

        """
        Get the market areas DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing total market areas for each supply location.
        """

        return self.market_areas_df
    
    def get_metadata(self):

        """
        Get the metadata dictionary.
        
        Returns
        -------
        dict
            Metadata associated with the market areas calculation.
        """

        return self.metadata
    
    def add_to_model(
        self,
        model_object,
        output_model = "Huff",
        verbose: bool = config.VERBOSE
        ):
        """
        Attach market areas to a Huff or MCI model.
        
        Integrates computed market areas into an existing Huff or MCI model object,
        or wraps an interaction matrix in a Huff/MCI model with market areas.
        
        Parameters
        ----------
        model_object : HuffModel, MCIModel, or InteractionMatrix
            The model or interaction matrix to which market areas are to be added.
        output_model : {"Huff", "MCI"}, default="Huff"
            Type of model to return when input is an InteractionMatrix.
        verbose : bool, default=True
            If True, print progress information.
        
        Returns
        -------
        HuffModel or MCIModel
            A new model object with market areas attached and updated metadata.
        
        Raises
        ------
        TypeError
            If model_object is not a HuffModel, MCIModel, or InteractionMatrix.
        ValueError
            If output_model is not "Huff" or "MCI" when input is InteractionMatrix.
        
        Example
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
        
        if not isinstance(model_object, (HuffModel, MCIModel, InteractionMatrix)):
            raise TypeError("Error while adding market areas to model: Parameter 'interaction_matrix' must be of class HuffModel, MCIModel, or InteractionMatrix")
        
        if isinstance(model_object, MCIModel):
            
            model_object_type = config.MODELS["MCI"]["description"]
            
            metadata = model_object.metadata

            model = MCIModel(
                interaction_matrix = model_object.interaction_matrix,
                coefs = model_object.get_coefs_dict(),
                mci_ols_model = model_object.get_mci_ols_model(),
                market_areas_df = self.market_areas_df,
                metadata = metadata
                )            
    
        elif isinstance(model_object, HuffModel):
            
            model_object_type = config.MODELS["Huff"]["description"]
            
            metadata = model_object.metadata
            
            model = HuffModel(
                interaction_matrix = model_object.interaction_matrix,
                market_areas_df = self.market_areas_df,
                metadata = metadata
            )

        elif isinstance(model_object, InteractionMatrix):
            
            model_object_type = "Interaction Matrix"

            if output_model not in [config.MODELS_LIST[0], config.MODELS_LIST[1]]:
                raise ValueError("Error while adding MarketAreas to model: Parameter 'output_model' must be either 'Huff' or 'MCI'")
            
            if output_model == config.MODELS_LIST[0]:

                metadata = {}

                model = HuffModel(
                    interaction_matrix=model_object,
                    market_areas_df=self.market_areas_df,
                    metadata=metadata
                )

            if output_model == config.MODELS_LIST[1]:

                metadata = {}
                
                model = MCIModel(
                    interaction_matrix=model_object,
                    coefs=model_object.coefs,
                    mci_ols_model=model_object.mci_ols_model,
                    market_areas_df=self.market_areas_df,
                    metadata=metadata
                )

        helper.add_timestamp(
            model,
            function="models.MarketAreas.add_to_model",
            process=f"{model_object_type} was added to {output_model} model object"
            )

        if verbose:
            print(f"{model_object_type} was added to {output_model} model object")
            
        return model

class HuffModel:

    """
    Container for a Huff model built from an interaction matrix and market areas.

    Parameters
    ----------
    interaction_matrix : InteractionMatrix
        The interaction matrix object used to calculate utilities, probabilities
        and flows for the Huff model.
    market_areas_df : pandas.DataFrame
        DataFrame containing total market areas per supply location.
    metadata : dict
        Metadata for the model (e.g., fit information, timestamps, notes).
    """

    def __init__(
        self,
        interaction_matrix, 
        market_areas_df,
        metadata
        ):

        self.interaction_matrix = interaction_matrix
        self.market_areas_df = market_areas_df
        self.metadata = metadata

    def get_interaction_matrix_df(self):

        """
        Return the interaction matrix DataFrame from the wrapped InteractionMatrix.

        Returns
        -------
        pandas.DataFrame
            The interaction matrix DataFrame contained in the underlying
            `InteractionMatrix` object.
        """

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        return interaction_matrix_df
    
    def get_supply_locations(self):

        """
        Return the `SupplyLocations` object used by this model.

        Returns
        -------
        SupplyLocations
            The supply locations instance from the underlying
            `InteractionMatrix`.
        """

        interaction_matrix = self.interaction_matrix
        supply_locations = interaction_matrix.get_supply_locations()

        return supply_locations

    def get_customer_origins(self):

        """
        Return the `CustomerOrigins` object used by this model.

        Returns
        -------
        CustomerOrigins
            The customer origins instance from the underlying
            `InteractionMatrix`.
        """

        interaction_matrix = self.interaction_matrix
        customer_origins = interaction_matrix.get_customer_origins()

        return customer_origins

    def get_market_areas_df(self):

        """
        Return the DataFrame containing total market areas for supply locations.

        Returns
        -------
        pandas.DataFrame
            DataFrame with total market areas stored in `self.market_areas_df`.
        """

        return self.market_areas_df
        
    def summary(self):
        """
        Print a concise summary of the Huff model and return related metadata.

        The summary prints information about supply locations, customer origins,
        defined utility weightings and (if present) model fit details. The
        function also returns a list with customer origins metadata, supply
        locations metadata, interaction matrix metadata and goodness-of-fit
        results (if available).

        Returns
        -------
        list
            [customer_origins_metadata, supply_locations_metadata,
            interaction_matrix_metadata, huff_modelfit]
        """

        interaction_matrix = self.interaction_matrix
        customer_origins = self.interaction_matrix.get_customer_origins()
        supply_locations = self.interaction_matrix.get_supply_locations()

        customer_origins_metadata = customer_origins.get_metadata()
        supply_locations_metadata = supply_locations.get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()

        print(config.MODELS["Huff"]["description"])        
        print("======================================")

        helper.print_summary_row(
            config.DEFAULT_NAME_SUPPLY_LOCATIONS,
            supply_locations_metadata["no_points"]
        )

        if supply_locations_metadata["attraction_col"][0] is not None:
            if isinstance(supply_locations_metadata["attraction_col"], list):
                attrac_cols = ', '.join(str(x) for x in supply_locations_metadata["attraction_col"])
            elif isinstance(supply_locations_metadata["attraction_col"], str):
                attrac_cols = ', '.join([supply_locations_metadata["attraction_col"]])
            else:
                attrac_cols = supply_locations_metadata["attraction_col"]
        else:
            attrac_cols = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_ATTRAC} column(s)",
            attrac_cols
        )

        helper.print_interaction_matrix_info(interaction_matrix)
        
        helper.print_weightings(interaction_matrix)
        
        huff_modelfit = None

        if interaction_matrix_metadata != {} and "fit" in interaction_matrix_metadata and interaction_matrix_metadata["fit"]["function"] is not None:

            print("--------------------------------------")

            print("Parameter estimation")

            helper.print_summary_row(
                "Fit function",
                interaction_matrix_metadata["fit"]["function"]
            )
            helper.print_summary_row(
                "Fit by",
                interaction_matrix_metadata["fit"]["fit_by"]
            )
            if interaction_matrix_metadata["fit"]["function"] == "huff_ml_fit":
                helper.print_summary_row(
                    "Fit method",
                    f"{interaction_matrix_metadata['fit']['method']} (Converged: {interaction_matrix_metadata['fit']['minimize_success']})"
                    )
            if interaction_matrix_metadata["fit"]["function"] == "mci_fit":
                helper.print_summary_row(
                    "Fit method",
                    interaction_matrix_metadata['fit']['method']
                    )

            print("--------------------------------------")            

            if "fit" in interaction_matrix_metadata:
                huff_modelfit = self.modelfit(by = interaction_matrix_metadata["fit"]["fit_by"])
            
            if huff_modelfit is not None:
                                
                print (f"Goodness-of-fit for {interaction_matrix_metadata['fit']['fit_by']}")

                helper.print_modelfit(huff_modelfit)
                
        print("======================================")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            huff_modelfit
            ]

    def show_log(self):
        
        """
        Shows all timestamp logs of the HuffModel object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp 

    def plot(
        self,
        origin_point_style: dict = None,
        location_point_style: dict = None,
        line_color: str = "black",
        line_alpha: float = 0.7,
        line_size_by: str = "flows",
        log_line_size: bool = True,
        save_output: bool = True,
        output_filepath: str = "Huff_model_results.png",
        output_dpi: int = 300,
        zoom: int = 15,
        legend: bool = True,
        map_title: str = "Map of Huff model results with OSM basemap",
        verbose: bool = False
        ):
        
        """
        Plot Huff model results by delegating to the underlying InteractionMatrix.

        This convenience method calls the wrapped `InteractionMatrix.plot()` to
        render a map of origins, supply locations and flow lines. See
        `InteractionMatrix.plot` for parameter details and returned values.

        Returns
        -------
        list
            The return value from `InteractionMatrix.plot()`:
            `[map_osm, layers_to_plot, layer_styles]`.

        Example
        --------
        >>> huff_model.plot(
        ...     origin_point_style={
        ...         "name": "Districts", 
        ...         "color": "black", 
        ...         "size": 100
        ...         },
        ...     location_point_style={
        ...         "name": "Supermarket chains", 
        ...         "color": {
        ...             "Name": {
        ...                 "Aldi Süd": "blue"
        ...                 }
        ...             }, 
        ...             "size": 100
        ...         }
        ... )
        """
        
        if origin_point_style is None:
            origin_point_style = {}
        if location_point_style is None:
            location_point_style = {}

        interaction_matrix = self.interaction_matrix        

        interaction_matrix_plot = interaction_matrix.plot(
            origin_point_style = origin_point_style,
            location_point_style = location_point_style,
            line_color = line_color,
            line_alpha = line_alpha,
            line_size_by = line_size_by,
            log_line_size = log_line_size,
            save_output = save_output,
            output_filepath = output_filepath,
            output_dpi = output_dpi,
            zoom = zoom,
            legend = legend,
            map_title = map_title,
            verbose = verbose
            )
        
        return interaction_matrix_plot

    def mci_fit(
        self,
        cols: list = None,
        alpha = 0.05,
        verbose: bool = config.VERBOSE
        ):
        
        """
        Fit a Multiplicative Competitive Interaction (MCI) model for this Huff model.

        This convenience method delegates to the underlying interaction matrix
        to perform the log-centering transformation (if needed) and to estimate
        the MCI model via OLS. Returns an `MCIModel` object containing the
        fitted coefficients and diagnostics. See Nakanishi and Cooper (1974, 1982),
        Huff and McCallum (2008), or Wieland (2017) for the steps of
        fitting the MCI model.

        Parameters
        ----------
        cols : list of str, optional
            Explanatory variables to include (default: attraction and transport cost).
        alpha : float, optional
            Significance level for confidence intervals (default: 0.05).
        verbose : bool, optional
            If True, print progress messages.

        Returns
        -------
        MCIModel
            A fitted MCIModel instance.

        Example
        --------
        >>> huff_model.mci_fit(cols=["A_j", "t_ij"])
        """

        if cols is None:
            cols = [
                config.DEFAULT_COLNAME_ATTRAC, 
                config.DEFAULT_COLNAME_TC
                ]

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
                "Coefficient": col[:-len(config.DEFAULT_LCT_SUFFIX)],
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
        
        metadata = {}   
        
        mci_model = MCIModel(
            interaction_matrix,
            coefs,
            mci_ols_model,
            None,
            metadata
            )
        
        helper.add_timestamp(
            mci_model,
            function="models.HuffModel.mci_fit",
            process=f"Estimated {config.MODELS['MCI']['description']} with {len(cols)} utility variables"
            )
    
        if verbose:
            print("OK")
        
        return mci_model

    def loglik(
        self,
        params,
        check_df_vars: bool = True,
        ):
        """
        Compute the negative log-likelihood for the Huff model based on totals.

        This method evaluates the negative log-likelihood of the Huff model
        using the current `market_areas_df` and model parameter vector `params`.
        It is used by optimization routines to fit model parameters to observed
        totals when `HuffModel.ml_fit` is invoked with `fit_by='totals'`.

        Parameters
        ----------
        params : list or numpy.ndarray
            Parameter vector: first element is attraction parameter, second is
            transport-cost parameter; additional parameters are location-specific.
        check_df_vars : bool, optional
            If True, validate required DataFrame variables before computing.

        Returns
        -------
        float
            Negative log-likelihood value.

        Raises
        ------
        ValueError
            If `params` does not have the required length or format.
        """

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
        fit_by: str = "probabilities",
        update_estimates: bool = True,
        check_numbers: bool = True,
        check_df_vars: bool = True,
        verbose: bool = config.VERBOSE
        ):
        """
        Fit model parameters for the Huff model.

        Supports three fitting modes: "probabilities"/"flows" (delegates to
        interaction-matrix likelihood optimization) and "totals" (fits to
        observed total market areas). On success, model parameters in the
        underlying interaction matrix and market areas are updated.

        Parameters
        ----------
        initial_params : list, optional
            Initial parameter guesses.
        method : str, optional
            Optimization method for `scipy.optimize.minimize`.
        bounds : list of tuple, optional
            Bounds for the parameters.
        constraints : list, optional
            Constraints for the optimizer.
        fit_by : {"probabilities", "flows", "totals"}, optional
            Which data to fit (default: "probabilities").
        update_estimates : bool, optional
            If True, recompute utilities/probabilities/flows after fitting.
        check_numbers : bool, optional
            If True, validate numeric totals when fitting by "totals".
        check_df_vars : bool, optional
            If True, check required DataFrame variables during fitting.
        verbose : bool, optional
            If True, print progress messages.

        Returns
        -------
        HuffModel
            The current HuffModel instance with updated parameters.

        Raises
        ------
        ValueError
            If `fit_by` is not one of the supported options or parameter shapes
            do not match model configuration.
        """

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

            helper.add_timestamp(
                self.interaction_matrix.supply_locations,
                function="models.HuffModel.ml_fit",
                process=f"Update of weighting parameters for {attrac_vars} attraction variables"
                )
            helper.add_timestamp(
                self.interaction_matrix.customer_origins,
                function="models.HuffModel.ml_fit",
                process="Update of weighting parameters for transport costs"
                )

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
                    
                    helper.add_timestamp(
                        self.interaction_matrix,
                        function="models.HuffModel.ml_fit",
                        status = "Error: No update of estimates because fit parameters contain NaN"
                        )
                    
                    helper.add_timestamp(
                        self,
                        function="models.HuffModel.ml_fit",
                        status = "Error: No update of estimates because fit parameters contain NaN"
                        )            

                else:
                                       
                    self.interaction_matrix.utility(check_df_vars = check_df_vars)
                    self.interaction_matrix.probabilities(check_df_vars = check_df_vars)
                    self.interaction_matrix.flows(check_df_vars = check_df_vars)
  
                    huff_model_new_marketareas = self.interaction_matrix.marketareas(check_df_vars = check_df_vars)
                    self.market_areas_df[config.DEFAULT_COLNAME_TOTAL_MARKETAREA] = huff_model_new_marketareas.get_market_areas_df()[config.DEFAULT_COLNAME_TOTAL_MARKETAREA]

                    helper.add_timestamp(
                        self.interaction_matrix,
                        function="models.HuffModel.ml_fit",
                        process="Update of interaction matrix with empirically estimated parameters"
                        )
                    
                    helper.add_timestamp(
                        self,
                        function="models.HuffModel.ml_fit",
                        process="Update of interaction matrix and total market areas with empirically estimated parameters"
                        )
            
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

        helper.add_timestamp(
            self,
            function="models.HuffModel.update",
            process="Update of interaction matrix and market areas"
            )
        
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
    """
    Container for a fitted Multiplicative Competitive Interaction (MCI) model.

    Parameters
    ----------
    interaction_matrix : InteractionMatrix
        The interaction matrix used to fit the MCI model.
    coefs : dict
        Estimated coefficients from the OLS fit (structured as in `mci_fit`).
    mci_ols_model : statsmodels.regression.linear_model.RegressionResults
        The fitted OLS model object.
    market_areas_df : pandas.DataFrame or None
        Optional DataFrame with total market areas produced from the model.
    metadata : dict
        Metadata related to the estimation and model configuration.
    """

    def __init__(
        self,
        interaction_matrix: InteractionMatrix,
        coefs: dict,
        mci_ols_model,
        market_areas_df,
        metadata
        ):

        self.interaction_matrix = interaction_matrix
        self.coefs = coefs
        self.mci_ols_model = mci_ols_model
        self.market_areas_df = market_areas_df
        self.metadata = metadata

    def get_interaction_matrix_df(self):

        """
        Return the interaction matrix DataFrame used by the MCI model.

        Returns
        -------
        pandas.DataFrame
            The interaction matrix DataFrame from the wrapped
            `InteractionMatrix`.
        """

        interaction_matrix = self.interaction_matrix
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()

        return interaction_matrix_df
    
    def get_supply_locations(self):

        """
        Return the `SupplyLocations` object used by the MCI model.

        Returns
        -------
        SupplyLocations
            Supply locations instance from the underlying interaction matrix.
        """

        interaction_matrix = self.interaction_matrix
        supply_locations = interaction_matrix.get_supply_locations()

        return supply_locations

    def get_customer_origins(self):

        """
        Return the `CustomerOrigins` object used by the MCI model.

        Returns
        -------
        CustomerOrigins
            Customer origins instance from the underlying interaction matrix.
        """

        interaction_matrix = self.interaction_matrix
        customer_origins = interaction_matrix.get_customer_origins()

        return customer_origins
    
    def get_mci_ols_model(self):

        """
        Return the underlying statsmodels OLS results object.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResults
            The fitted OLS model used for MCI estimation.
        """

        return self.mci_ols_model
    
    def get_coefs_dict(self):

        """
        Return the dictionary of estimated coefficients.

        Returns
        -------
        dict
            Coefficients structured as produced by `InteractionMatrix.mci_fit()`.
        """

        return self.coefs
    
    def get_market_areas_df(self):

        """
        Return the DataFrame containing total market areas (if available).

        Returns
        -------
        pandas.DataFrame or None
            Market areas DataFrame stored in `self.market_areas_df` or None.
        """

        return self.market_areas_df

    def modelfit(self):
        """
        Compute goodness-of-fit metrics for the MCI model probabilities.

        Returns
        -------
        dict or None
            Goodness-of-fit metrics as returned by `gof.modelfit`, or None if
            empirical probability values are not present or cannot be computed.
        """

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
        """
        Print a concise summary of the MCI model and return related metadata.

        Returns
        -------
        list
            [customer_origins_metadata, supply_locations_metadata,
            interaction_matrix_metadata, mci_modelfit]
        """

        interaction_matrix = self.interaction_matrix
        coefs = self.coefs

        customer_origins_metadata = interaction_matrix.get_customer_origins().get_metadata()
        supply_locations_metadata = interaction_matrix.get_supply_locations().get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()

        print(config.MODELS["MCI"]["description"])
        print("============================================")

        helper.print_summary_row(
            config.DEFAULT_NAME_SUPPLY_LOCATIONS,
            supply_locations_metadata["no_points"]
        )
        helper.print_summary_row(
            config.DEFAULT_NAME_CUSTOMER_ORIGINS,
            customer_origins_metadata["no_points"]
        )

        helper.print_interaction_matrix_info(interaction_matrix)
        
        print("--------------------------------------------")
        
        print("Weighting estimates")
 
        coefficients_rows = []

        for key, value in coefs.items():

            coefficient_name = value["Coefficient"]
            if coefficient_name == config.DEFAULT_COLNAME_ATTRAC:
                coefficient_name = config.DEFAULT_NAME_ATTRAC
            if coefficient_name == config.DEFAULT_COLNAME_TC:
                coefficient_name = config.DEFAULT_NAME_TC

            coefficients_rows.append({
                "": coefficient_name,
                "Estimate": round(value["Estimate"], config.FLOAT_ROUND),
                "SE": round(value["SE"], config.FLOAT_ROUND),
                "t": round(value["t"], config.FLOAT_ROUND),
                "p": round(value["p"], config.FLOAT_ROUND),
                "CI lower": round(value["CI_lower"], config.FLOAT_ROUND),
                "CI upper": round(value["CI_upper"], config.FLOAT_ROUND)
            })

        coefficients_df = pd.DataFrame(coefficients_rows)
        
        print(coefficients_df.to_string(index=False))
        
        mci_modelfit = None

        mci_modelfit = self.modelfit()

        if mci_modelfit is not None:

            print("--------------------------------------------")
            
            print ("Goodness-of-fit for probabilities")

            helper.print_modelfit(mci_modelfit)            

        print("============================================")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            mci_modelfit
            ]
    
    def show_log(self):
        
        """
        Shows all timestamp logs of the MCIModel object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp 
               
    def plot(
        self,
        origin_point_style: dict = {},
        location_point_style: dict = {},
        line_color: str = "black",
        line_alpha: float = 0.7,
        line_size_by: str = "flows",
        log_line_size: bool = True,
        save_output: bool = True,
        output_filepath: str = "MCI_model_results.png",
        output_dpi: int = 300,
        zoom: int = 15,
        legend: bool = True,
        map_title: str = "Map of MCI model results with OSM basemap",
        verbose: bool = False
        ):
        
        """
        Plot MCI model results by delegating to the underlying InteractionMatrix.

        See `InteractionMatrix.plot` for detailed parameter descriptions and
        returned values. This is a convenience wrapper that forwards all
        plotting arguments to the interaction matrix.

        Returns
        -------
        list
            The return value from `InteractionMatrix.plot()`.
        """

        interaction_matrix = self.interaction_matrix        

        interaction_matrix_plot = interaction_matrix.plot(
            origin_point_style = origin_point_style,
            location_point_style = location_point_style,
            line_color = line_color,
            line_alpha = line_alpha,
            line_size_by = line_size_by,
            log_line_size = log_line_size,
            save_output = save_output,
            output_filepath = output_filepath,
            output_dpi = output_dpi,
            zoom = zoom,
            legend = legend,
            map_title = map_title,
            verbose = verbose
            )
        
        return interaction_matrix_plot
                  
    def utility(
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        check_df_vars: bool = True
        ):
        """
        Compute utilities for the MCI model according to the selected transformation.
        Using the original formulation or inverse log-centering transformation
        are available. See Nakanishi and Cooper (1982) or Wieland (2017) for details 
        on the inverse log-centering transformation (ILCT).

        Parameters
        ----------
        transformation : str, optional
            Transformation to use for MCI (`LCT` or `ILCT`).
        check_df_vars : bool, optional
            If True, validate required DataFrame variables before computing.

        Returns
        -------
        self
            The `MCIModel` instance with updated `interaction_matrix` utilities.

        Raises
        ------
        ValueError
            If `transformation` is not in the supported list or required columns
            are missing from the interaction matrix.
        """

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

        helper.add_timestamp(
            self,
            function="models.MCIModel.utility",
            process="Calculated utilities"
            )

        return self
    
    def probabilities (
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        verbose: bool = config.VERBOSE
        ):
        """
        Calculate probabilities for the MCI model based on computed utilities.

        Parameters
        ----------
        transformation : str, optional
            Transformation used when computing utilities (passed to `utility`).
        verbose : bool, optional
            If True, print informational messages.

        Returns
        -------
        self
            The `MCIModel` instance with updated probabilities in its
            underlying interaction matrix.
        """

        interaction_matrix = self.interaction_matrix        
        interaction_matrix_df = interaction_matrix.get_interaction_matrix_df()        
       
        if config.DEFAULT_COLNAME_PROBABILITY in interaction_matrix_df.columns:
            
            if config.DEFAULT_COLNAME_PROBABILITY_OBSERVED not in interaction_matrix_df.columns:
            
                if verbose:
                    print("NOTE: Probabilities in interaction matrix are treated as empirical probabilities")
                
                interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY_OBSERVED] = interaction_matrix_df[config.DEFAULT_COLNAME_PROBABILITY]
                
                self.interaction_matrix.interaction_matrix_df = interaction_matrix_df
                
                helper.add_timestamp(
                    self.interaction_matrix,
                    function="models.MCIModel.probabilities",
                    process=f"Saved observed market shares in column '{config.DEFAULT_COLNAME_PROBABILITY_OBSERVED}'"
                    )
                
            else:
                
                raise InteractionMatrixError(f"Error in {config.MODELS['MCI']['description']} analysis: Interaction matrix does not contain probabilities.")

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

        helper.add_timestamp(
            self.interaction_matrix,
            function="models.MCIModel.probabilities",
            process="Calculated probabilities"
            )

        return self
        
    def flows (
        self,
        transformation = config.DEFAULT_MCI_TRANSFORMATION,
        check_df_vars: bool = True
        ):
        """
        Compute expected flows for the MCI model.

        Parameters
        ----------
        transformation : str, optional
            Transformation used to compute utilities (passed to `probabilities`).
        check_df_vars : bool, optional
            If True, validate required DataFrame variables before computing.

        Returns
        -------
        self
            The `MCIModel` instance with flows computed and stored in
            `self.interaction_matrix_df`.
        """

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

        helper.add_timestamp(
            self,
            function="models.MCIModel.flows",
            process="Calculated expected customer flows"
            )

        return self

    def marketareas(
        self,
        check_df_vars: bool = True
        ):
        """
        Aggregate expected flows into total market areas and return an MCIModel.

        Parameters
        ----------
        check_df_vars : bool, optional
            If True, validate required DataFrame variables before aggregating.

        Returns
        -------
        MCIModel
            A new `MCIModel` instance containing the computed market areas.
        """

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
            market_areas_df = market_areas_df,
            metadata = self.metadata
            )
        
        helper.add_timestamp(
            mci_model,
            function="models.MCIModel.marketareas",
            process="Calculated total market areas"
            )

        return mci_model
    
class HansenAccessibility:

    def __init__(
        self,
        interaction_matrix, 
        hansen_df,
        metadata
        ):

        self.interaction_matrix = interaction_matrix
        self.hansen_df = hansen_df
        self.metadata = metadata

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

    def get_hansen_df(self):

        """
        Return the Hansen accessibility DataFrame computed by this object.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with Hansen accessibility values stored in `self.hansen_df`.
        """

        return self.hansen_df
    
    def summary(self):
        """
        Print a concise summary of the Hansen accessibility calculation.

        Returns
        -------
        list
            [customer_origins_metadata, supply_locations_metadata,
            interaction_matrix_metadata, metadata]
        """

        interaction_matrix = self.interaction_matrix
        customer_origins = self.get_customer_origins()
        supply_locations = self.get_supply_locations()

        customer_origins_metadata = customer_origins.get_metadata()
        supply_locations_metadata = supply_locations.get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()
        metadata = self.metadata

        print(config.MODELS["Hansen"]["description"])
        print("======================================")

        helper.print_summary_row(
            config.DEFAULT_NAME_SUPPLY_LOCATIONS,
            supply_locations_metadata["no_points"]
        )

        if supply_locations_metadata["attraction_col"][0] is not None:

            if isinstance(supply_locations_metadata["attraction_col"], list):
                attrac_cols = ', '.join(str(x) for x in supply_locations_metadata["attraction_col"])
            elif isinstance(supply_locations_metadata["attraction_col"], str):
                attrac_cols = ', '.join([supply_locations_metadata["attraction_col"]])
            else:
                attrac_cols = supply_locations_metadata["attraction_col"]

        else:

            attrac_cols = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_ATTRAC} column(s)",
            attrac_cols
        )

        print("--------------------------------------")

        helper.print_summary_row(
            config.DEFAULT_NAME_CUSTOMER_ORIGINS,
            customer_origins_metadata["no_points"]
        )

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_MARKETSIZE} column",
            customer_origins_metadata["marketsize_col"]
        )

        helper.print_interaction_matrix_info(interaction_matrix)

        helper.print_weightings(interaction_matrix)

        print("--------------------------------------")
        
        if metadata["calculation"]["from_origins"]:

            helper.print_summary_row(
                "Calculated from",
                config.DEFAULT_NAME_CUSTOMER_ORIGINS
                )
            
        else:
            
            helper.print_summary_row(
                "Calculated from",
                config.DEFAULT_NAME_SUPPLY_LOCATIONS
                )
            
        print("======================================")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            metadata
            ]

    def show_log(self):
        
        """
        Shows all timestamp logs of the HansenAccessibility object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp
    

class FloatingCatchment:

    def __init__(
        self,
        interaction_matrix, 
        fca_df,
        metadata
        ):

        self.interaction_matrix = interaction_matrix
        self.fca_df = fca_df
        self.metadata = metadata

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

    def get_fca_df(self):

        """
        Return the floating catchment accessibility DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with accessibility values stored in `self.fca_df`.
        """

        return self.fca_df
    
    def summary(self):
        """
        Print a concise summary of the floating catchment (2SFCA) analysis.

        Returns
        -------
        list
            [customer_origins_metadata, supply_locations_metadata,
            interaction_matrix_metadata, metadata]
        """

        interaction_matrix = self.interaction_matrix
        customer_origins = self.get_customer_origins()
        supply_locations = self.get_supply_locations()

        customer_origins_metadata = customer_origins.get_metadata()
        supply_locations_metadata = supply_locations.get_metadata()
        interaction_matrix_metadata = interaction_matrix.get_metadata()
        metadata = self.metadata

        print(config.MODELS["2SFCA"]["description"])
        print("======================================")
 
        helper.print_summary_row(
            config.DEFAULT_NAME_SUPPLY_LOCATIONS,
            supply_locations_metadata["no_points"]
        )

        if supply_locations_metadata["attraction_col"][0] is not None:

            if isinstance(supply_locations_metadata["attraction_col"], list):
                attrac_cols = ', '.join(str(x) for x in supply_locations_metadata["attraction_col"])
            elif isinstance(supply_locations_metadata["attraction_col"], str):
                attrac_cols = ', '.join([supply_locations_metadata["attraction_col"]])
            else:
                attrac_cols = supply_locations_metadata["attraction_col"]

        else:

            attrac_cols = None

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_ATTRAC} column(s)",
            attrac_cols
        )

        print("--------------------------------------")

        helper.print_summary_row(
            config.DEFAULT_NAME_CUSTOMER_ORIGINS,
            customer_origins_metadata["no_points"]
        )

        helper.print_summary_row(
            f"{config.DEFAULT_NAME_MARKETSIZE} column",
            customer_origins_metadata["marketsize_col"]
        )

        helper.print_interaction_matrix_info(interaction_matrix)

        if metadata["calculation"]["use_weightings"]:

            helper.print_weightings(interaction_matrix)
        
        else:
            
            print("Calculation without weightings")

        print("--------------------------------------")

        helper.print_summary_row(
            "Threshold",
            metadata['calculation']['threshold']
        )
        
        print("======================================")

        return [
            customer_origins_metadata,
            supply_locations_metadata,
            interaction_matrix_metadata,
            metadata
            ]

    def show_log(self):
        
        """
        Shows all timestamp logs of the FloatingCatchment object
        """

        timestamp = helper.print_timestamp(self)
        return timestamp

def create_interaction_matrix(
    customer_origins,
    supply_locations,
    requiring_attributes: bool = True,
    remove_duplicates: bool = True,
    verbose: bool = False
    ):

    """
    Create an :class:`InteractionMatrix` from origin and destination objects.

    Parameters
    ----------
    customer_origins : CustomerOrigins
        A :class:`CustomerOrigins` instance containing origin locations and
        optional attributes (e.g. market size).
    supply_locations : SupplyLocations
        A :class:`SupplyLocations` instance containing destination locations
        and attraction attributes.
    requiring_attributes : bool, optional
        If True, require market size and attraction attributes (default True).
    remove_duplicates : bool, optional
        If True drop duplicate locations based on the provided unique id
        columns (default True).
    verbose : bool, optional
        If True, print progress messages (default False).

    Returns
    -------
    InteractionMatrix
        The constructed interaction matrix object combining all origin-
        destination pairs and placeholder columns for transport costs,
        utilities, probabilities and flows.

    Raises
    ------
    ValueError
        If `customer_origins` is not a :class:`CustomerOrigins` or
        `supply_locations` is not a :class:`SupplyLocations`.

    Example
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

    if not isinstance(customer_origins, CustomerOrigins):
        raise ValueError ("Error while creating interaction matrix: customer_origins must be of class CustomerOrigins")
    if not isinstance(supply_locations, SupplyLocations):
        raise ValueError ("Error while creating interaction matrix: supply_locations must be of class SupplyLocations")

    if verbose:
        print("Loading customer origins and supply locations", end = " ... ")

    customer_origins_metadata = customer_origins.get_metadata()
    supply_locations_metadata = supply_locations.get_metadata()
    
    customer_origins_unique_id = customer_origins_metadata["unique_id"]
    customer_origins_marketsize = customer_origins_metadata["marketsize_col"]
    
    supply_locations_unique_id = supply_locations_metadata["unique_id"]
    supply_locations_attraction = supply_locations_metadata["attraction_col"][0]
    
    if verbose:
        print("OK")

    if customer_origins_marketsize is None and requiring_attributes:
        print(f"WARNING: {config.DEFAULT_NAME_MARKETSIZE} column in customer origins not defined and is set to {config.DEFAULT_COLNAME_MARKETSIZE} = np.nan. Use CustomerOrigins.define_marketsize().")  
    
    if verbose:
        print("Arranging customer origins data", end = " ... ")
    
    customer_origins_geodata_gpd = pd.DataFrame(customer_origins.get_geodata_gpd())
    customer_origins_geodata_gpd_original = pd.DataFrame(customer_origins.get_geodata_gpd_original())
    
    if remove_duplicates:
        customer_origins_geodata_gpd = customer_origins_geodata_gpd.drop_duplicates(subset=customer_origins_unique_id)
        customer_origins_geodata_gpd_original = customer_origins_geodata_gpd_original.drop_duplicates(subset=customer_origins_unique_id)
    
    if customer_origins_marketsize is None:
        customer_origins_marketsize = config.DEFAULT_COLNAME_MARKETSIZE
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
    
    if verbose:
        print("OK")        
            
    if supply_locations_attraction is None and requiring_attributes:
        print(f"WARNING: {config.DEFAULT_NAME_ATTRAC} column in supply locations not defined and is set to {config.DEFAULT_COLNAME_ATTRAC} = np.nan. Use SupplyLocations.define_attraction().")

    if verbose:
        print("Arranging supply locations data", end = " ... ")

    supply_locations_geodata_gpd = pd.DataFrame(supply_locations.get_geodata_gpd())
    supply_locations_geodata_gpd_original = pd.DataFrame(supply_locations.get_geodata_gpd_original())
    
    if remove_duplicates:
        supply_locations_geodata_gpd = supply_locations_geodata_gpd.drop_duplicates(subset=supply_locations_unique_id)
        supply_locations_geodata_gpd_original = supply_locations_geodata_gpd_original.drop_duplicates(subset=supply_locations_unique_id)
        
    if supply_locations_metadata["attraction_col"][0] is None:
        supply_locations_attraction = config.DEFAULT_COLNAME_ATTRAC
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
    
    if verbose:
        print("OK")
        print("Constructing interaction matrix", end = " ... ")

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
    
    helper.add_timestamp(
        interaction_matrix,
        function="models.create_interaction_matrix",
        process="Created interaction matrix" if requiring_attributes else "Created interaction matrix without attributes required"
        )
    
    if verbose:
        print("OK")
    
    return interaction_matrix

def market_shares(
    df: pd.DataFrame,
    turnover_col: str,
    ref_col: str = None,
    marketshares_col: str = config.DEFAULT_COLNAME_PROBABILITY,
    drop_total_col: bool = True,
    check_df_vars: bool = True
    ):
    """
    Calculate market shares for turnover within a reference area.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the turnover/use column and optional
        reference grouping column.
    turnover_col : str
        Name of the column holding turnover (numerator for the market share).
    ref_col : str or None, optional
        Optional reference column (e.g., region). If provided market shares
        are calculated within each group; otherwise across the whole
        dataframe (default None).
    marketshares_col : str, optional
        Column name to write market shares to (default
        `config.DEFAULT_COLNAME_PROBABILITY`).
    drop_total_col : bool, optional
        If True, drop the temporary total-turnover column used in the
        calculation (default True).
    check_df_vars : bool, optional
        If True, call :func:`helper.check_vars` to validate input columns
        (default True).

    Returns
    -------
    pandas.DataFrame
        The input dataframe extended with a column containing market shares.

    Raises
    ------
    KeyError
        If `ref_col` is provided but not present in `df`.

    Example
    --------
    >>> df = market_shares(df, turnover_col="turnover", ref_col="region")

    """

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

    """
    Retrieve isochrones for a set of points using the OpenRouteService API.

    Parameters
    ----------
    geodata_gpd : geopandas.GeoDataFrame
        GeoDataFrame with point geometries for which isochrones are required.
    unique_id_col : str
        Column name with unique identifiers for the points.
    segments : list, optional
        List of time (minutes) or distance (km) segments to request
        (default [5, 10, 15]). The units depend on `range_type`.
    range_type : {'time', 'distance'}, optional
        Whether segments refer to travel time ('time') or distance
        ('distance'). Default is 'time'.
    intersections : str, optional
        ORS `intersections` parameter (default 'true').
    profile : str, optional
        ORS routing profile (default 'driving-car').
    donut : bool, optional
        If True, remove overlapping rings to create donut polygons
        (default True).
    ors_server : str, optional
        ORS server base URL.
    ors_auth : str or None, optional
        ORS API key or auth token (default None).
    timeout : int, optional
        Request timeout in seconds.
    delay : float, optional
        Delay in seconds between API calls (throttling).
    save_output : bool, optional
        If True save result to `output_filepath`.
    output_filepath : str, optional
        Filepath where results are saved if `save_output` is True.
    output_crs : str, optional
        Coordinate reference system of the returned GeoDataFrame
        (default 'EPSG:4326').
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing isochrone polygons with a column for the
        unique id and the segment value.

    Raises
    ------
    ValueError
        If no isochrones were retrieved from ORS (likely server error).

    Example
    --------
    >>> isos = get_isochrones(Haslach.get_geodata_gpd(), unique_id_col='BEZEICHN', segments=[5,10])

    """

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
   
    """
    Apply a log-centering transformation to specified columns grouped by a reference column.

    The transformation computes log(x) minus the group geometric mean log, which
    is suitable for multiplicative modelling approaches.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the columns to transform.
    ref_col : str
        Column name to group by (e.g., market or region).
    cols : list
        List of column names to transform.
    suffix : str, optional
        Suffix appended to column names for the transformed variables
        (default `config.DEFAULT_LCT_SUFFIX`).

    Returns
    -------
    pandas.DataFrame
        The dataframe extended with new columns named `<col><suffix>`
        containing the transformed values.

    Raises
    ------
    KeyError
        If `ref_col` is not present in `df`.

    Example
    --------
    >>> df2 = log_centering_transformation(df, ref_col='region', cols=['pop','income'])

    """

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
    """
    Apply a weighting function to a vector of values.

    The function selected via `func` is evaluated with parameters `a`, `b`
    and optionally `c`. The permitted functions and their formulas are
    defined in :mod:`config` under `PERMITTED_WEIGHTING_FUNCTIONS`.

    Parameters
    ----------
    values : pandas.Series
        Numeric series providing the input values to be weighted (e.g.
        transport costs).
    func : str
        Key identifying the weighting function to use (must be one of
        `config.PERMITTED_WEIGHTING_FUNCTIONS_LIST`).
    b : float
        Primary parameter for the weighting function (e.g. lambda).
    c : float, optional
        Additional parameter required by some weighting functions.
    a : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Result of applying the weighting function to `values`.

    Raises
    ------
    InteractionMatrixError
        If `func` is None or not a permitted function.
    TypeError
        If `values` is not numeric.
    ValueError
        If a required parameter `c` is missing for the chosen function.

    Example
    --------
    >>> w = weighting(df['t_ij'], func='power', b=-2.2)

    """
    
    if func is None:
        raise InteractionMatrixError(f"Parameter 'func' is None must be one of {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")
    
    if func not in config.PERMITTED_WEIGHTING_FUNCTIONS_LIST:
        raise InteractionMatrixError(f"Parameter 'func' is equal to {func} must be one of {', '.join(config.PERMITTED_WEIGHTING_FUNCTIONS_LIST)}")
    
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