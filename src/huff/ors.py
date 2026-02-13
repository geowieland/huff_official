#-----------------------------------------------------------------------
# Name:        ors (huff package)
# Purpose:     OpenRouteService client
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.5.2
# Last update: 2026-02-11 20:49
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import requests
import geopandas as gp
from shapely.geometry import shape
import huff.config as config
import huff.helper as helper


class Isochrone:

    """
    Container class for isochrone results from OpenRouteService and related metadata.

    Attributes
    ----------
    isochrones_gdf : geopandas.GeoDataFrame
        Geometries of computed isochrones.
    metadata : dict
        Metadata associated with the isochrone computation (from ORS server).
    status_code : int
        Status code of the isochrone request (from ORS server).
    save_config : dict
        Configuration used for saving outputs.
    error_message : str or None
        Error message if the computation failed.
    """

    def __init__(
        self, 
        isochrones_gdf, 
        metadata,
        status_code,
        save_config,
        error_message
        ):

        self.isochrones_gdf = isochrones_gdf
        self.metadata = metadata
        self.status_code = status_code
        self.save_config = save_config
        self.error_message = error_message

    def get_isochrones_gdf(self):
    
        """
        Return the geopandas.GeoDataFrame containing the computed isochrones.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Isochrones GeoDataFrame if available, otherwise None.
        """

        isochrones_gdf = self.isochrones_gdf
        return isochrones_gdf

    def summary(
        self,
        ors_info: bool = False
        ):

        """
        Print a summary of the isochrone query and status.

        Parameters
        ----------
        ors_info : bool
            Whether to include detailed ORS server output in the summary (default: False).

        Returns
        -------
        dict
            Metadata associated with this Isochrone object. Keys may include:
            - 'attribution': str
            - 'service': str        
            - 'query': dict
            - 'engine': dict
            - 'ors_timestamp': str
            - 'timestamp': dict

        Examples
        --------
        >>> ors_client = Client(auth = "5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f")
        >>> x, y = 7.84117, 47.997697
        >>> Freiburg_main_station_iso = ors_client.isochrone(
        ...     locations = [[x,y]],
        ...     segments = [900, 300, 600],
        ...     save_output = True,
        ...     output_filepath = "Freiburg_main_station_iso.shp",
        ...     output_crs = "EPSG:4326",
        ...     verbose = True
        ... )
        >>> Freiburg_main_station_iso.summary(ors_info=False)
        """

        metadata = self.metadata
        status_code = self.status_code
        save_config = self.save_config
        error_message = self.error_message

        print("Isochrones")
        print("===========================================")

        print("Server respone")
        
        helper.print_summary_row(
            "ORS Status code",
            status_code
            )
        if error_message != "":
            helper.print_summary_row(
                "Error message",
                error_message
                )

        print("-------------------------------------------")

        if metadata is not None and len(metadata) > 0 and "query" in metadata:
            
            range_str = [str(range) for range in metadata["query"]["range"]]
            profile = metadata["query"]["profile"]
            range_type = metadata["query"][config.ORS_ENDPOINTS["Isochrones"]["Parameters"]["unit"]["param"]]
            no_locations = len(metadata["query"]["locations"])

            helper.print_summary_row(
                "Locations",
                no_locations
            )
            helper.print_summary_row(
                "Segments",
                ", ".join(range_str)
            )
            helper.print_summary_row(
                "Range type",
                range_type
            )
            helper.print_summary_row(
                "Profile",
                profile
            )

            if ors_info:

                attribution = metadata["attribution"]
                engine_version = metadata["engine"]["version"]
                engine_build_date = metadata["engine"]["build_date"]
                engine_graph_date = metadata["engine"]["graph_date"]
                engine_osm_date = metadata["engine"]["osm_date"]

                ors_url = save_config["ors_url"]
                auth = save_config["auth"]

                print("-------------------------------------------")

                helper.print_summary_row(
                    "Attribution",
                    attribution
                )
                helper.print_summary_row(
                    "Engine version",
                    engine_version
                )
                helper.print_summary_row(
                    "Engine build date",
                    engine_build_date
                )
                helper.print_summary_row(
                    "Graph date",
                    engine_graph_date
                )
                helper.print_summary_row(
                    "OSM date",
                    engine_osm_date
                )
                helper.print_summary_row(
                    "ORS URL",
                    ors_url
                )
                helper.print_summary_row(
                    "ORS API Token",
                    auth
                )

        else:

            print("No isochrones were built.")
        
        print("===========================================")
        
        return metadata

class TimeDistanceMatrix:

    """
    Container class for time/distance matrix results from OpenRouteService and related metadata.

    Attributes
    ----------
    matrix_df : pandas.DataFrame
        Table with all distances/travel times (I origins x J destinations).
    metadata : dict
        Metadata associated with the matrix computation (from ORS server).
    status_code : int
        Status code of the matrix request (from ORS server).
    save_config : dict
        Configuration used for saving outputs.
    error_message : str or None
        Error message if the computation failed.
    """

    def __init__(
        self, 
        matrix_df, 
        metadata,
        status_code,
        save_config,
        error_message
        ):

        self.matrix_df = matrix_df
        self.metadata = metadata
        self.status_code = status_code
        self.save_config = save_config
        self.error_message = error_message

    def get_matrix(self):

        """
        Return the pandas.DataFrame containing the computed distance/travel time matrix.

        Returns
        -------
        pandas.DataFrame or None
            Matrix DataFrame if available, otherwise None.
        """

        return self.matrix_df
    
    def get_metadata(self):

        """
        Return the metadata (dict) associated with the matrix computation (from ORS server).

        Returns
        -------
        dict
            Metadata associated with this Isochrone object. Keys may include:
            - 'attribution': str
            - 'service': str        
            - 'query': dict
            - 'engine': dict
            - 'ors_timestamp': str
            - 'timestamp': dict
        """

        return self.metadata
    
    def get_config(self):

        """
        Return the configuration used for saving outputs (dict).

        Returns
        -------
        dict
            Configuration details associated with this Isochrone object. Keys may include:
            - 'range_type': range_type
            - 'save_output': bool
            - 'output_filepath' : str
            - 'output_crs': str
            - 'ors_url': str
            - 'auth': str
        """

        return self.save_config
    
    def summary(
        self,
        ors_info: bool = False
        ):

        """
        Print a summary of the distance/time matrix query and status.

        Parameters
        ----------
        ors_info : bool
            Whether to include detailed ORS server output in the summary (default: False).

        Returns
        -------
        dict
            Metadata associated with this Isochrone object. Keys may include:
            - 'attribution': str
            - 'service': str        
            - 'query': dict
            - 'engine': dict
            - 'ors_timestamp': str
            - 'timestamp': dict

        Examples
        --------
        >>> coords = [
        ...     [7.84117, 47.997697],
        ...     [7.945725, 48.476014],
        ...     [8.400558, 48.993997],
        ...     [8.41080, 49.01090] 
        ... ]
        >>> travel_time_matrix = ors_client.matrix(
        ...     locations=coords,
        ...     sources=[0,1],
        ...     destinations=[2,3],
        ...     verbose=True
        ... )
        >>> travel_time_matrix.summary(ors_info=False)
        """

        metadata = self.metadata
        status_code = self.status_code
        save_config = self.save_config
        error_message = self.error_message

        print("Matrix")
        print("===========================================")

        print("Server response")
        
        helper.print_summary_row(
            "ORS Status code",
            status_code
            )
        if error_message != "":
            helper.print_summary_row(
                "Error message",
                error_message
                )
       
        if metadata is not None and len(metadata) > 0 and "query" in metadata:

            print("-------------------------------------------")

            profile = metadata["query"]["profile"]
            no_locations = len(metadata["query"]["locations"])
            range_type = ', '.join(metadata["query"][config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]])
     
            helper.print_summary_row(
                "Locations",
                no_locations
            )
            if save_config["sources"] is not None:
                helper.print_summary_row(
                    "Sources",
                    save_config["sources"]
                )
            if save_config["destinations"] is not None:
                helper.print_summary_row(
                    "Destinations",
                    save_config["destinations"]
                )

            helper.print_summary_row(
                "Range type",
                range_type
            )
            helper.print_summary_row(
                "Profile",
                profile
            )

            if ors_info:

                attribution = metadata["attribution"]
                engine_version = metadata["engine"]["version"]
                engine_build_date = metadata["engine"]["build_date"]
                engine_graph_date = metadata["engine"]["graph_date"]
                engine_osm_date = metadata["engine"]["osm_date"]

                ors_url = save_config["ors_url"]
                auth = save_config["auth"]

                print("-------------------------------------------")

                helper.print_summary_row(
                    "Attribution",
                    attribution
                )
                helper.print_summary_row(
                    "Engine version",
                    engine_version
                )
                helper.print_summary_row(
                    "Engine build date",
                    engine_build_date
                )
                helper.print_summary_row(
                    "Graph date",
                    engine_graph_date
                )
                helper.print_summary_row(
                    "OSM date",
                    engine_osm_date
                )
                helper.print_summary_row(
                    "ORS URL",
                    ors_url
                )
                helper.print_summary_row(
                    "ORS API Token",
                    auth
                )

        else:

            print("No time/distance matrix was built.")

        print("===========================================")

class Client:

    """
    A client for accessing the OpenRouteService (ORS) API.

    Provides methods for retrieving distance/travel time matrices and isochrones 
    from ORS given locations and user parameters.
    See the ORS API documentation: https://openrouteservice.org/dev/#/api-docs
    See the current API restrictions: https://openrouteservice.org/restrictions/
    """
    
    def __init__(
        self,
        server = config.ORS_SERVER,
        auth: str = config.ORS_AUTH
        ):
        
        self.server = server
        self.auth = auth
            
    def isochrone(
        self,
        locations: list,
        segments: list = None,
        range_type: str = "time",
        intersections: str = "true",
        profile: str = "driving-car",
        timeout = 10,
        save_output: bool = True,
        output_filepath: str = "isochrones.shp",
        output_crs: str = "EPSG:4326",
        verbose: bool = False
        ):
        
        """
        Retrieve isochrones from the ORS API for given locations and user parameters.

        Parameters
        ----------
        locations : list
            List of coordinates (lon, lat) representing origin points.
            Example from ORS documentation: [[8.681495,49.41461],[8.686507,49.41943]].
        segments : list, optional
            Travel time or distance thresholds for isochrones (default: [900, 600, 300]).
            Note: Travel time is measured in seconds, and distance in meters.
        range_type : str, optional
            Type of range measurement: {"origins", "destinations"} (default: "time").
        intersections : str, optional
            Whether to return isochrones as intersections ("true"/"false") (default: "true").
        profile : str, optional
            Travel mode, e.g., "driving-car", "cycling", "walking" (default: "driving-car").
            See https://openrouteservice.org/dev/#/api-docs/v2/isochrones/{profile}/post for available profiles.
        timeout : int, optional
            Maximum request time in seconds (default: 10).
        save_output : bool, optional
            Whether to save the resulting GeoDataFrame as a file (default: True).
        output_filepath : str, optional
            Filepath to save the output if save_output=True (default: "isochrones.shp").
        output_crs : str, optional
            Coordinate reference system of the output (default: "EPSG:4326").
        verbose : bool, optional
            If True, print additional information during execution (default: False).

        Returns
        -------
        Isochrone
            An `Isochrone` object containing the geopandas.GeoDataFrame, metadata, status, and error message.

        Examples
        --------
        >>> ors_client = Client(auth = "5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f")
        >>> x, y = 7.84117, 47.997697
        >>> Freiburg_main_station_iso = ors_client.isochrone(
        ...     locations = [[x,y]],
        ...     segments = [900, 300, 600],
        ...     save_output = True,
        ...     output_filepath = "Freiburg_main_station_iso.shp",
        ...     output_crs = "EPSG:4326",
        ...     verbose = True
        ... )
        >>> Freiburg_main_station_iso.summary(ors_info=False)
        """

        if segments is None:
            segments = [900, 600, 300]
        
        check_params(
            range_type,
            profile,
        )

        assert len(segments) <= config.ORS_ENDPOINTS["Isochrones"]["Restrictions"]["Intervals"], f"ORS client does not allow >{config.ORS_ENDPOINTS['Isochrones']['Restrictions']['Intervals']} intervals in an Isochrones query. See {config.ORS_URL_RESTRICTIONS}."
        assert len(locations) <= config.ORS_ENDPOINTS["Isochrones"]["Restrictions"]["Locations"], f"ORS client does not allow >{config.ORS_ENDPOINTS['Isochrones']['Restrictions']['Locations']} locations in an Isochrones query. See {config.ORS_URL_RESTRICTIONS}."
    
        ors_url = self.server + config.ORS_ENDPOINTS["Isochrones"]["endpoint"] + profile
        auth = self.auth

        headers = {
            **config.ORS_HEADERS,
            "Authorization": auth
        }

        body = {
            "locations": locations,
            "range": segments,
            "intersections": intersections,
            config.ORS_ENDPOINTS["Isochrones"]["Parameters"]["unit"]["param"]: range_type
        }
        
        save_config = {
            config.ORS_ENDPOINTS["Isochrones"]["Parameters"]["unit"]["param"]: range_type,
            "save_output": save_output,
            "output_filepath" : output_filepath,
            "output_crs": output_crs,
            "ors_url": ors_url,
            "auth": auth
            }

        try:

            response = requests.post(
                ors_url, 
                headers=headers, 
                json=body,
                timeout=timeout
                )
            
        except Exception as e:

            error_message = f"Error while accessing ORS server: {str(e)}"
            
            print (error_message)
            
            status_code = 99999
            isochrones_gdf = None 
            metadata = {}

            isochrone_output = Isochrone(
                isochrones_gdf, 
                metadata,
                status_code,
                save_config,
                error_message
                )
            
            return isochrone_output

        status_code = response.status_code

        if status_code == 200:

            if verbose:
                print ("Accessing ORS server successful")

            response_json = response.json()
            
            metadata = response_json["metadata"]
            
            features = response_json["features"]
            geometries = [shape(feature["geometry"]) for feature in features]

            isochrones_gdf = gp.GeoDataFrame(
                features, 
                geometry=geometries, 
                crs=config.WGS84_CRS
                )

            isochrones_gdf[config.ORS_SEGMENT_COL] = 0
            isochrones_gdf_properties_dict = dict(isochrones_gdf["properties"])
            
            for i in range(len(isochrones_gdf_properties_dict)):
                isochrones_gdf.iloc[i,3] = isochrones_gdf_properties_dict[i]["value"]

            isochrones_gdf = isochrones_gdf.drop(columns=["properties"])
            isochrones_gdf = isochrones_gdf.to_crs(output_crs)

            if save_output:
                
                isochrones_gdf.to_file(output_filepath)
                
                if verbose:
                    print (f"Saved as {output_filepath}")
                    
            error_message = ""

        else:
            
            error_message = f"Error while accessing ORS server. Status code: {status_code} - {response.reason}"
            
            print(error_message)
            
            isochrones_gdf = None
            metadata = {}
        
        if "timestamp" in metadata:
            metadata["ors_timestamp"] = metadata.pop("timestamp")

        isochrone_output = Isochrone(
            isochrones_gdf, 
            metadata,
            status_code,
            save_config,
            error_message
            )
               
        helper.add_timestamp(
            isochrone_output,
            function="ors.Client.isochrone",
            process=f"Retrieved isochrones ({range_type}, {profile}) with {len(segments)} segments for {len(locations)} locations",
            status = "OK" if error_message == "" else error_message
            )
        
        return isochrone_output

    def matrix(
        self,
        locations: list,
        sources: list = None,
        destinations: list = None,
        id: str = None,
        range_type: str = "time",
        profile: str = "driving-car",
        resolve_locations: bool = False,    
        units: str = "mi",
        timeout: int = 10,
        save_output: bool = False,
        output_filepath: str = "matrix.csv",
        csv_sep: str = ";",
        csv_decimal: str = ",",
        csv_encoding: str = None,
        verbose: bool = config.VERBOSE
        ):

        """
        Retrieves a travel time or distance matrix from OpenRouteService (ORS) for a set of locations.

        Parameters
        ----------
        locations : list
            Coordinates of locations for the matrix query, e.g., [[lon1, lat1], [lon2, lat2]].
            ORS documentation example: [[9.70093,48.477473],[9.207916,49.153868],[37.573242,55.801281],[115.663757,38.106467]].
        sources : list, optional
            Indices of source locations in `locations` to include (if available).
        destinations : list, optional
            Indices of destination locations in `locations` to include (if available).
        id : str, optional
            Optional identifier for the request.
        range_type : str, optional
            Type of range measurement: {"time", "distance"} (default: "time").
        profile : str, optional
            Mode of travel: "driving-car", "cycling-regular", etc. (default: "driving-car").
        resolve_locations : bool, optional
            Whether ORS should snap coordinates to the road network (default: False).
        units : str, optional
            Output units, e.g., "mi" or "km" (default: "mi").
        timeout : int, optional
            Request timeout in seconds (default: 10).
        save_output : bool, optional
            If True, saves the result as a CSV file (default: False).
        output_filepath : str, optional
            Filepath to save CSV if `save_output=True` (default: "matrix.csv").
        csv_sep : str, optional
            Separator for CSV file (default: ";").
        csv_decimal : str, optional
            Decimal symbol for CSV file (default: ",").
        csv_encoding : str, optional
            Encoding for CSV file (default: None, system default).
        verbose : bool, optional
            If True, print additional information during execution (default: False).

        Returns
        -------
        TimeDistanceMatrix
            Object containing the DataFrame (`matrix_df`), metadata, status code, save config,
            and error message. Provides additional methods and timestamps.

        Examples
        --------
        >>> coords = [
        ...     [7.84117, 47.997697],
        ...     [7.945725, 48.476014],
        ...     [8.400558, 48.993997],
        ...     [8.41080, 49.01090] 
        ... ]
        >>> travel_time_matrix = ors_client.matrix(
        ...     locations=coords,
        ...     sources=[0,1],
        ...     destinations=[2,3],
        ...     verbose=True
        ... )
        >>> travel_time_matrix.summary(ors_info=False)
        """

        if sources is None:
            sources = []
        if destinations is None:
            destinations = []

        check_params(
            range_type,
            profile,
        )
        
        ors_url = self.server + config.ORS_ENDPOINTS["Matrix"]["endpoint"] + profile
        auth = self.auth
        
        headers = {
            **config.ORS_HEADERS,
            "Authorization": auth
        }

        body = {
            "locations": locations,
            config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]: [config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["options"][range_type]["param"]],
            "resolve_locations": resolve_locations
        }
        if id is not None:
            body["id"] = id
        if sources != []:
            body["sources"] = sources
        if destinations != []:
            body["destinations"] = destinations
        if units is not None:
            body["units"] = units

        save_config = {
            config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]: range_type,
            "save_output": save_output,
            "output_filepath": output_filepath,
            "ors_url": ors_url,
            "auth": auth,
            "sources": len(sources) if len(sources) > 0 else None,
            "destinations": len(destinations) if len(destinations) > 0 else None,
        }

        try:

            response = requests.post(
                ors_url, 
                headers=headers, 
                json=body,
                timeout=timeout
                )
            
        except Exception as e:
            
            error_message = f"Error while accessing ORS server: {str(e)}"

            print (error_message)
            
            status_code = 99999
            matrix_df = None
            metadata = {}

        status_code = response.status_code

        if status_code == 200:

            if verbose:
                print ("Accessing ORS server successful")

            response_json = response.json()

            metadata = response_json["metadata"]
            
            matrix_df = pd.DataFrame(
                {
                    config.MATRIX_COL_SOURCE: pd.Series(dtype="string"),
                    f"{config.MATRIX_COL_SOURCE}_lat": pd.Series(dtype="float"),
                    f"{config.MATRIX_COL_SOURCE}_lon": pd.Series(dtype="float"),
                    f"{config.MATRIX_COL_SOURCE}_snapped_distance": pd.Series(dtype="float"),
                    config.MATRIX_COL_DESTINATION: pd.Series(dtype="string"),
                    f"{config.MATRIX_COL_DESTINATION}_lat": pd.Series(dtype="float"),
                    f"{config.MATRIX_COL_DESTINATION}_lon": pd.Series(dtype="float"),
                    f"{config.MATRIX_COL_DESTINATION}_snapped_distance": pd.Series(dtype="float"),
                    f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}": pd.Series(dtype="string"),
                    range_type: pd.Series(dtype="string")
                }
            )

            for i, value in enumerate(response_json["durations"]):

                source_lat = response_json["sources"][i]["location"][1]
                source_lon = response_json["sources"][i]["location"][0]
                source_snapped_distance = response_json["sources"][i]["snapped_distance"]
                
                for j, entry in enumerate(value):

                    destination_lat = response_json["destinations"][j]["location"][1]
                    destination_lon = response_json["destinations"][j]["location"][0]
                    destination_snapped_distance = response_json["destinations"][j]["snapped_distance"]

                    matrix_row = pd.DataFrame(
                        [
                            {
                                config.MATRIX_COL_SOURCE: str(i),
                                f"{config.MATRIX_COL_SOURCE}_lat": source_lat,
                                f"{config.MATRIX_COL_SOURCE}_lon": source_lon,
                                f"{config.MATRIX_COL_SOURCE}_snapped_distance": source_snapped_distance,
                                config.MATRIX_COL_DESTINATION: str(j),
                                f"{config.MATRIX_COL_DESTINATION}_lat": destination_lat,
                                f"{config.MATRIX_COL_DESTINATION}_lon": destination_lon,
                                f"{config.MATRIX_COL_DESTINATION}_snapped_distance": destination_snapped_distance,
                                f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}": f"{i}{config.MATRIX_OD_SEPARATOR}{j}",
                                range_type: entry
                                }
                            ],
                        columns=matrix_df.columns
                        )
                    
                    if matrix_df.empty:
                        
                        matrix_df = pd.DataFrame(
                            [
                                {
                                    config.MATRIX_COL_SOURCE: str(i),
                                    f"{config.MATRIX_COL_SOURCE}_lat": source_lat,
                                    f"{config.MATRIX_COL_SOURCE}_lon": source_lon,
                                    f"{config.MATRIX_COL_SOURCE}_snapped_distance": source_snapped_distance,
                                    config.MATRIX_COL_DESTINATION: str(j),
                                    f"{config.MATRIX_COL_DESTINATION}_lat": destination_lat,
                                    f"{config.MATRIX_COL_DESTINATION}_lon": destination_lon,
                                    f"{config.MATRIX_COL_DESTINATION}_snapped_distance": destination_snapped_distance,
                                    f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}": f"{i}{config.MATRIX_OD_SEPARATOR}{j}",
                                    range_type: entry
                                    }
                                ], 
                            columns=matrix_df.columns
                            )
                        
                    else:
                    
                        matrix_df = pd.concat(
                            [
                                matrix_df, 
                                matrix_row
                                ], 
                            ignore_index=True
                            )

            if save_output:
                
                matrix_df.to_csv(
                    output_filepath, 
                    decimal = csv_decimal, 
                    sep = csv_sep, 
                    encoding = csv_encoding
                    )
                
                if verbose:
                    print ("Saved as", output_filepath)

            error_message = ""

        else:

            error_message = f"Error while accessing ORS server. Status code: {status_code} - {response.reason}"
            
            print(error_message)

            matrix_df = None
            metadata = {}

        if "timestamp" in metadata:
            metadata["ors_metadata"] = metadata.pop("timestamp")
            
        matrix_output = TimeDistanceMatrix(
            matrix_df, 
            metadata,
            status_code,
            save_config,
            error_message
            )
        
        helper.add_timestamp(
            matrix_output,
            function="ors.Client.matrix",
            process=f"Retrieved transport costs matrix ({range_type}, {profile}) for {len(locations)} locations",
            status = "OK" if error_message == "" else error_message
            )
             
        return matrix_output    
    
def check_params(
    range_type,
    profile
    ):

    """
    Validate `range_type` and `profile` against allowed ORS API values, definded in the `config` module.
    This function is implemented in the Client methods.
    """
        
    assert range_type in config.ORS_RANGE_TYPES_LIST_API, f"Parameter 'range_type' must be one of these: {', '.join(config.ORS_RANGE_TYPES_LIST_API)}."
    assert profile in config.ORS_PROFILES_LIST_API, f"Parameter 'profile' must be one of these: {', '.join(config.ORS_PROFILES_LIST_API)}."