#-----------------------------------------------------------------------
# Name:        osm (huff package)
# Purpose:     Helper functions for OpenStreetMap API
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.4.17
# Last update: 2026-09-05 12:44
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import math
import requests
import geopandas as gp
from shapely.geometry import box
import tempfile
import time
from PIL import Image
import huff.config as config


OSM_TILES_SERVER = config.OSM_TILES_SERVER
OSM_USER_AGENT = config.OSM_USER_AGENT
OSM_REFERER = config.GITHUB_HUFF_URL
   
class Client:

    """
    Client for downloading OpenStreetMap tiles.

    See the OSM documentation: https://wiki.openstreetmap.org/wiki/Raster_tile_providers
    """

    def __init__(
        self, 
        server=None, 
        headers=None
        ):
        
        self.server = server if server is not None else OSM_TILES_SERVER
        self.headers = headers if headers is not None else {
            "User-Agent": OSM_USER_AGENT,
            "Referer": OSM_REFERER,
        }

    def download_tile(
        self,
        zoom, 
        x, 
        y,
        timeout = 10
        ):

        """
        Download a single OSM tile as a PIL.Image object.

        Parameters
        ----------
        zoom : int
            Zoom level of the tile.
            See the OSM documentation with respect to zoom: https://wiki.openstreetmap.org/wiki/Zoom_levels
        x : int
            X coordinate of the tile.
        y : int
            Y coordinate of the tile.
        timeout : int, optional
            Request timeout in seconds (default 10).

        Returns
        -------
        PIL.Image.Image or None
            Tile image if successful, None on error.

        Example
        -------
        >>> client = Client()
        >>> img = client.download_tile(12, 1205, 1532)
        >>> img.show()
        """

        osm_url = self.server + f"{zoom}/{x}/{y}.png"
       
        try:
        
            response = requests.get(
                osm_url, 
                headers = self.headers,
                timeout = timeout,
                verify=config.REQUESTS_VERIFY
                )

            if response.status_code == 200:

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
                return Image.open(tmp_file_path)
            
            else:

                print(f"Error while accessing OSM server with URL {osm_url}. Status code: {response.status_code} - {response.reason}")

                return None
            
        except Exception as e:
            
            print(f"Error while accessing OSM server with URL {osm_url}. Error message: {e}")
            
            return None

def get_basemap(
    sw_lat,
    sw_lon,
    ne_lat,
    ne_lon,
    zoom=15,
    tile_delay=config.OSM_DELAY,
    verbose: bool = False
    ):

    """
    Retrieve and stitch OSM tiles to create a basemap for a bounding box.

    Returns
    -------
    tuple
        (stitched_image, extent_3857)
        stitched_image : PIL.Image.Image
        extent_3857 : tuple
            (min_x, min_y, max_x, max_y) in EPSG:3857.
    """

    def lat_lon_to_tile(lat, lon, zoom):

        n = 2 ** zoom

        x = int(
            n * ((lon + 180) / 360)
        )

        y = int(
            n
            * (
                1
                - (
                    math.log(
                        math.tan(math.radians(lat))
                        + 1 / math.cos(math.radians(lat))
                    )
                    / math.pi
                )
            )
            / 2
        )

        return x, y


    def tile_to_lat_lon(x, y, zoom):

        n = 2 ** zoom

        lon = x / n * 360.0 - 180.0

        lat = math.degrees(
            math.atan(
                math.sinh(
                    math.pi * (1 - 2 * y / n)
                )
            )
        )

        return lat, lon


    def stitch_tiles(
        zoom,
        sw_lat,
        sw_lon,
        ne_lat,
        ne_lon,
        delay=tile_delay
        ):

        osm_client = Client()  

        sw_x_tile, sw_y_tile = lat_lon_to_tile(
            sw_lat,
            sw_lon,
            zoom
        )

        ne_x_tile, ne_y_tile = lat_lon_to_tile(
            ne_lat,
            ne_lon,
            zoom
        )

        tile_size = 256

        width = (
            ne_x_tile - sw_x_tile + 1
        ) * tile_size

        height = (
            sw_y_tile - ne_y_tile + 1
        ) * tile_size

        stitched_image = Image.new(
            "RGB",
            (width, height)
        )

        for x in range(
            sw_x_tile,
            ne_x_tile + 1
        ):

            for y in range(
                ne_y_tile,
                sw_y_tile + 1
            ):

                tile = osm_client.download_tile(
                    zoom=zoom,
                    x=x,
                    y=y
                )

                if tile is not None:

                    tile = tile.convert("RGB")

                    stitched_image.paste(
                        tile,
                        (
                            (x - sw_x_tile) * tile_size,
                            (y - ne_y_tile) * tile_size
                        )
                    )

                else:

                    print(f"WARNING: Error while retrieving tile {x}, {y}.")

                time.sleep(delay)

        min_lat, min_lon = tile_to_lat_lon(
            sw_x_tile,
            sw_y_tile + 1,
            zoom
        )

        max_lat, max_lon = tile_to_lat_lon(
            ne_x_tile + 1,
            ne_y_tile,
            zoom
        )

        bbox = box(
            min_lon,
            min_lat,
            max_lon,
            max_lat
        )

        extent_3857 = (
            gp.GeoSeries(
                [bbox],
                crs=config.WGS84_CRS
            )
            .to_crs(
                config.PSEUDO_MERCATOR_CRS
            )
            .total_bounds
        )

        extent_3857 = tuple(extent_3857)

        return stitched_image, extent_3857

    stitched_image, extent_3857 = stitch_tiles(
        zoom,
        sw_lat,
        sw_lon,
        ne_lat,
        ne_lon
    )

    stitched_image_path = config.DEFAULT_FILENAME_ORS_TMP
    stitched_image.save(stitched_image_path)

    if verbose:
        print(f"Temporary file saved as '{stitched_image_path}'.")

    return stitched_image, extent_3857
        
def define_tiles_server(
    server_url: str    
    ):

    """
    Define the OpenStreetMap tiles server URL.

    Parameters
    ----------
    server_url : str
        The URL of the OSM tiles server.

    Example
    -------
    >>> define_tiles_server("https://tile.openstreetmap.org/")
    """

    global OSM_TILES_SERVER
    OSM_TILES_SERVER = server_url
    
    print(f"OSM tiles server set to: {OSM_TILES_SERVER}")
    print(config.OSM_ATTRIBUTION)
    
def define_headers(
    user_agent: str,
    referer: str
    ):

    """
    Define the headers for OSM requests.

    Parameters
    ----------
    user_agent : str
        The User-Agent string to be used in requests.
    referer : str
        The Referer string to be used in requests.

    Example
    -------
    >>> define_headers("MyApp/1.0", "https://myapp.example.com")
    """

    global OSM_USER_AGENT, OSM_REFERER
    OSM_USER_AGENT = user_agent
    OSM_REFERER = referer
    
    print(f"OSM User-Agent set to: {OSM_USER_AGENT}")
    print(f"OSM Referer set to: {OSM_REFERER}")
    
define_tiles_server(config.OSM_TILES_SERVER)

define_headers(
    OSM_USER_AGENT, 
    OSM_REFERER
    )