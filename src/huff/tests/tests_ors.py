#-----------------------------------------------------------------------
# Name:        tests_ors (huff package)
# Purpose:     Tests for ors module in the Huff Model package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.1.1
# Last update: 2026-09-30 19:05
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import geopandas as gpd
from shapely.geometry import Point
from huff.ors import Client
from huff.gistools import overlay_difference, buffers


ors_client = Client(auth = "5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f")
# API token FOR TESTING PURPOSES ONLY


# Buffers vs. isochrones:

x, y = 7.84117, 47.997697
# WGS84 coordinates of Freiburg main station

Freiburg_main_station = gpd.GeoDataFrame(
    {"id": [1]},
    geometry = [Point(x,y)],
    crs = "EPSG:4326"
)
# Creating a geopandas GeoDataFrame from the point

Freiburg_main_station = Freiburg_main_station.to_crs("EPSG:25832")
# Set a projection

Freiburg_main_station_buf1 = buffers(
    point_gdf = Freiburg_main_station,
    unique_id_col = "id",
    distances = [5000, 10000, 15000],
    save_output = False,
    output_crs = "EPSG:25832",
    verbose=True
    )
# Buffers with 5, 10, and 15 km

Freiburg_main_station_buf2 = overlay_difference(
    Freiburg_main_station_buf1,
    sort_col = "segment"
)
# Buffers as rings

Freiburg_main_station_buf2.to_file("Freiburg_main_station_buf2.shp")
# Saving as shapefile

Freiburg_main_station_iso1 = ors_client.isochrone(
    locations = [[x,y]],
    segments = [300, 600, 900],
    output_crs = "EPSG:25832",    
    verbose = True
)
# Retrieve isochrones of 5, 10, and 15 minutes

Freiburg_main_station_iso1_gdf = Freiburg_main_station_iso1.isochrones_gdf
# Extract geodataframe

Freiburg_main_station_iso1.summary(ors_info=True)
# Summary of isochrones

Freiburg_main_station_iso2_gdf = overlay_difference(
    Freiburg_main_station_iso1_gdf,
    sort_col = "segment"
)
# Isochrones as rings

Freiburg_main_station_iso2_gdf.to_file("Freiburg_main_station_iso2_gdf.shp")
# Saving as shapefile

Freiburg_main_station_iso3 = ors_client.isochrone(
    locations = [[x,y]],
    range_type="distance",
    segments = [5000, 10000, 15000],
    output_crs = "EPSG:25832",
    save_output=False,    
    verbose = True
)
# Retrieve isoareas of 5, 10, and 15 km

Freiburg_main_station_iso3_gdf = Freiburg_main_station_iso3.isochrones_gdf
# Extract geodataframe

Freiburg_main_station_iso4_gdf = overlay_difference(
    Freiburg_main_station_iso3_gdf,
    sort_col = "segment"
)
# Isoareas as rings

Freiburg_main_station_iso4_gdf.to_file("Freiburg_main_station_iso4_gdf.shp")
# Saving as shapefile


# Travel time matrix:

coords = [
    [7.84117, 47.997697],
    [7.945725, 48.476014],
    [8.400558, 48.993997],
    [8.41080, 49.01090] 
]
# 4 Locations in (Freiburg, Offenburg, Karlsruhe)

travel_time_matrix = ors_client.matrix(
    locations=coords,
    sources=[0,1],
    destinations=[2,3],
    verbose=True
)
# Travel time matrix

travel_time_matrix.summary(ors_info=False)
# Summary of travel time matrix

print(travel_time_matrix.get_matrix())
# Show travel times (in seconds!)


Freiburg_main_station_iso3 = ors_client.isochrone(
    locations = [[x,y]],
    segments = [900, 300, 600],
    save_output = True,
    output_filepath = "Freiburg_main_station_iso3.shp",
    output_crs = "EPSG:4326",
    verbose = True
)
# Retrieve isochrones
# This MUST produce a ValueError