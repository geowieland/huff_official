#-----------------------------------------------------------------------
# Name:        ors (huff package)
# Purpose:     OpenRouteService client
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.5.0
# Last update: 2026-02-05 16:18
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import requests
import geopandas as gp
from shapely.geometry import shape
import huff.config as config
import huff.helper as helper

class Isochrone:

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

    def get_isochrones_gdf(self):
    
        """
        Return the geopandas.GeoDataFrame containing the computed isochrones.
        """

        isochrones_gdf = self.isochrones_gdf
        return isochrones_gdf

    def summary(self):

        """
        Print a summary of the isochrone query and status.
        """

        metadata = self.metadata
        status_code = self.status_code

        if metadata is not None:
            range_str = [str(range) for range in metadata["query"]["range"]]
            profile = metadata["query"]["profile"]
            range_type = metadata["query"][config.ORS_ENDPOINTS["Isochrones"]["Parameters"]["unit"]["param"]]
            no_locations = len(metadata["query"]["locations"])

            print("Locations    " + str(no_locations))
            print("Segments     " + ", ".join(range_str))
            print("Range type   " + range_type)
            print("Profile      " + profile)
            
        else:
            print("No isochrones were built.")
        
        print("Status code  " + str(status_code))

class TimeDistanceMatrix:

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

    def get_matrix(self):

        """
        Return the pandas.DataFrame containing the computed distance/travel time matrix.
        """

        return self.matrix_df
    
    def get_metadata(self):

        """
        Return the metadata (dict) associated with the matrix computation (from ORS server).
        """

        return self.metadata
    
    def get_config(self):

        """
        Return the configuration used for saving outputs (dict).
        """

        return self.save_config
    
    def summary(self):

        """
        Print a summary of the distance/time matrix query and status.
        """

        metadata = self.metadata
        status_code = self.status_code

        if metadata is not None:

            profile = metadata["query"]["profile"]
            no_locations = len(metadata["query"]["locations"])
            range_type = config[config.ORS_ENDPOINTS["Matrix"]["Parameters"]["unit"]["param"]]

            print("Locations    " + str(no_locations))
            print("Range type   " + range_type)
            print("Profile      " + profile)

        else:

            print("No time/distance matrix was built.")

        print("Status code  " + str(status_code))

class Client:

    def __init__(
        self,
        server = config.ORS_SERVER,
        auth: str = config.ORS_AUTH
        ):
        
        self.server = server
        self.auth = auth

        """
        A client for accessing the OpenRouteService (ORS) API.

        Provides methods for retrieving distance/travel time matrices and isochrones 
        from ORS given locations and user parameters.
        See the ORS API documentation: https://openrouteservice.org/dev/#/api-docs
        See the current API restrictions: https://openrouteservice.org/restrictions/
        """
            
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
        >>> client = Client(auth="your_api_key")
        >>> locations = [[8.004, 48.013], [8.005, 48.014]]
        >>> result = client.isochrone(
        ...     locations=locations,
        ...     segments=[600, 300],
        ...     range_type="time",
        ...     profile="driving-car",
        ...     verbose=True
        ... )

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
            "output_crs": output_crs
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
            metadata["ors_metadata"] = metadata.pop("timestamp")

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
            process=f"Retrieved isochrones ({range_type}, {profile}) with {len(segments)} for {len(locations)} locations",
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
            Type of range measurement: {"origins", "destinations"} (default: "time").
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
        >>> client = Client(auth="your_api_key")
        >>> locations = [[8.004, 48.013], [8.005, 48.014]]
        >>> result = client.matrix(
        ...     locations=locations,
        ...     range_type="time",
        ...     profile="driving-car",
        ...     verbose=True
        ... )
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
            "output_filepath": output_filepath
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
                columns=[
                    config.MATRIX_COL_SOURCE,
                    f"{config.MATRIX_COL_SOURCE}_lat",
                    f"{config.MATRIX_COL_SOURCE}_lon",
                    f"{config.MATRIX_COL_SOURCE}_snapped_distance",
                    config.MATRIX_COL_DESTINATION,
                    f"{config.MATRIX_COL_DESTINATION}_lat",
                    f"{config.MATRIX_COL_DESTINATION}_lon", 
                    f"{config.MATRIX_COL_DESTINATION}_snapped_distance",
                    f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}", 
                    range_type
                    ]
                )

            for i, value in enumerate(response_json["durations"]):

                source_lat = response_json["sources"][i]["location"][1]
                source_lon = response_json["sources"][i]["location"][0]
                source_snapped_distance = response_json["sources"][i]["snapped_distance"]
                
                for j, entry in enumerate(value):

                    destination_lat = response_json["destinations"][j]["location"][1]
                    destination_lon = response_json["destinations"][j]["location"][0]
                    destination_snapped_distance = response_json["destinations"][j]["snapped_distance"]

                    matrix_row = pd.Series(
                        {
                            config.MATRIX_COL_SOURCE: str(i),
                            f"{config.MATRIX_COL_SOURCE}_lat": source_lat,
                            f"{config.MATRIX_COL_SOURCE}_lon": source_lon,
                            f"{config.MATRIX_COL_SOURCE}_snapped_distance": source_snapped_distance,
                            config.MATRIX_COL_DESTINATION: str(j),
                            f"{config.MATRIX_COL_DESTINATION}_lat": destination_lat,
                            f"{config.MATRIX_COL_DESTINATION}_lon": destination_lon,
                            f"{config.MATRIX_COL_DESTINATION}_snapped_distance": destination_snapped_distance,
                            f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}": str(i)+config.MATRIX_OD_SEPARATOR+str(j), 
                            range_type: entry
                            }
                            )

                    matrix_df = pd.concat([
                        matrix_df, 
                        pd.DataFrame([matrix_row])])

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