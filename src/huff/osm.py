#-----------------------------------------------------------------------
# Name:        osm (huff package)
# Purpose:     Helper functions for OpenStreetMap API
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.4.10
# Last update: 2026-02-11 17:44
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import math
import requests
import tempfile
import time
from PIL import Image
import huff.config as config


class Client:

    """
    Client for downloading OpenStreetMap tiles.

    See the OSM documentation: https://wiki.openstreetmap.org/wiki/Raster_tile_providers
    """

    def __init__(
        self,
        server = config.OSM_TILES_SERVER,
        headers = {
           'User-Agent': 'huff.osm/1.4.10 (your_name@your_email_provider.com)'
           }
        ):
        
        self.server = server
        self.headers = headers

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
                timeout = timeout
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
    zoom = 15,
    tile_delay = 0.1,
    verbose: bool = False
    ):

    """
    Retrieve and stitch OSM tiles to create a basemap for a bounding box.

    Parameters
    ----------
    sw_lat, sw_lon : float
        Latitude and longitude of the southwest corner.
    ne_lat, ne_lon : float
        Latitude and longitude of the northeast corner.
    zoom : int, optional
        Zoom level of the map (default 15).
        See the OSM documentation with respect to zoom: https://wiki.openstreetmap.org/wiki/Zoom_levels
    tile_delay : float, optional
        Delay in seconds between tile downloads (default 0.1).
    verbose : bool, optional
        If True, print progress messages (default False).

    Returns
    -------
    PIL.Image.Image
        Stitched basemap image.

    Example
    -------
    >>> img = get_basemap(40.712, -74.227, 40.774, -74.125, zoom=14)
    >>> img.show()
    """

    def lat_lon_to_tile(
        lat, 
        lon, 
        zoom
        ):
        
        n = 2 ** zoom
        x = int(n * ((lon + 180) / 360))
        y = int(n * (1 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)) / 2)
        return x, y

    def stitch_tiles(
        zoom, 
        sw_lat, 
        sw_lon, 
        ne_lat, 
        ne_lon,
        delay = tile_delay        
        ):

        osm_client = Client(
            server = config.OSM_TILES_SERVER
            )
        
        sw_x_tile, sw_y_tile = lat_lon_to_tile(sw_lat, sw_lon, zoom)
        ne_x_tile, ne_y_tile = lat_lon_to_tile(ne_lat, ne_lon, zoom)

        tile_size = 256
        width = (ne_x_tile - sw_x_tile + 1) * tile_size
        height = (sw_y_tile - ne_y_tile + 1) * tile_size

        stitched_image = Image.new('RGB', (width, height))
        
        for x in range(sw_x_tile, ne_x_tile + 1):
            for y in range(ne_y_tile, sw_y_tile + 1):
                tile = osm_client.download_tile(
                    zoom = zoom, 
                    x = x, 
                    y = y
                    )
                if tile:
                    
                    stitched_image.paste(tile, ((x - sw_x_tile) * tile_size, (sw_y_tile - y) * tile_size))
                else:
                    print(f"WARNING: Error while retrieving tile {x}, {y}.")

                time.sleep(delay)
        
        return stitched_image
    
    stitched_image = stitch_tiles(
        zoom, 
        sw_lat, 
        sw_lon, 
        ne_lat, 
        ne_lon
        )

    if stitched_image:

        stitched_image_path = config.DEFAULT_FILENAME_ORS_TMP
        stitched_image.save(stitched_image_path)
        
        if verbose:
            print(f"Saved as {stitched_image_path}.")

    else:
        print("Error while building stitched images")