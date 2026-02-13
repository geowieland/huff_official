#-----------------------------------------------------------------------
# Name:        tests:ors (huff package)
# Purpose:     Tests for ors module in the Huff Model package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-02-13 18:44
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


from huff.ors import Client


ors_client = Client(auth = "5b3ce3597851110001cf62487536b5d6794a4521a7b44155998ff99f")
# API token FOR TESTING PURPOSES ONLY


# Isochrones:

x, y = 7.84117, 47.997697
# WGS 84 coordinates of Freiburg main station

Freiburg_main_station_iso = ors_client.isochrone(
    locations = [[x,y]],
    segments = [900, 300, 600],
    save_output = True,
    output_filepath = "Freiburg_main_station_iso.shp",
    output_crs = "EPSG:4326",
    verbose = True
)
# Retrieve isochrones

Freiburg_main_station_iso.summary(ors_info=False)
# Summary of isochrones


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