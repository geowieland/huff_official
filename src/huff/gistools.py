#-----------------------------------------------------------------------
# Name:        gistools (huff package)
# Purpose:     GIS tools
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.4.19
# Last update: 2025-12-23 14:37
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------


import os
from math import pi, sin, cos, acos, radians
import pandas as pd
import geopandas as gp
from pandas.api.types import is_numeric_dtype
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box, Point
import contextily as cx
from PIL import Image
from huff.osm import get_basemap
import huff.config as config


def manhattan_distance(
    source: list,
    destination: list,
    unit: str = "m"
):
    lat1, lat2, lon1, lon2 = lonlat_transform(
        source,
        destination,
        transform=False
    )

    distance = abs(lat1-lat2)*111.32+abs(lon1-lon2)*111.32*cos(radians(lat1))

    if unit == "m": 
        distance = distance*1000
    elif unit == "mile": 
        distance = distance/1.60934

    return distance

def euclidean_distance(
    source: list,
    destination: list,
    unit: str = "m"
    ):

    lat1_r, lat2_r, lon1_r, lon2_r = lonlat_transform(
        source,
        destination
        )

    distance = 6378 * (acos(sin(lat1_r) * sin(lat2_r) + cos(lat1_r) * cos(lat2_r) * cos(lon2_r - lon1_r)))

    if unit == "m": 
        distance = distance*1000
    elif unit == "mile": 
        distance = distance/1.60934

    return distance

def distance_matrix(
    sources: list,
    destinations: list,
    sources_uid: list = [],
    destinations_uid: list = [],
    distance_type: str = "euclidean",
    unit: str = "m",    
    save_output: bool = True,
    output_filepath: str = "lines.shp",
    output_crs: str = "EPSG:4326",    
    verbose: bool = False
    ):    

    if distance_type not in config.DISTANCE_TYPES_LIST_FUNC:
        raise ValueError(f"Distance type {distance_type} is unknown. Please choose one of the following: {', '.join(config.DISTANCE_TYPES_LIST_FUNC)}.")
    
    if verbose:
        print(f"Calculating {distance_type} distance matrix for {len(sources)} sources and {len(destinations)} destinations", end = " ... ")

    if len(sources_uid) > 0 and len(sources_uid) != len(sources):
        print(f"Source unique IDs were stated ({len(sources_uid)}), but do not correspond to the number of sources ({len(sources)}). No source IDs are added to the GeoDataFrame.")
    if len(destinations_uid) > 0 and len(destinations_uid) != len(destinations):
        print(f"Destination unique IDs were stated ({len(destinations_uid)}), but do not correspond to the number of destinations ({len(destinations)}). No source IDs are added to the GeoDataFrame.")
    
    matrix = []
    line_data = []
        
    for i, source in enumerate(sources):
        
        row = []
        for j, destination in enumerate(destinations):
            
            if distance_type == config.DISTANCE_TYPES_LIST_FUNC[0]:
                dist = euclidean_distance(
                    source, 
                    destination, 
                    unit
                    )
            elif distance_type == config.DISTANCE_TYPES_LIST_FUNC[1]:
                dist = manhattan_distance(
                    source, 
                    destination, 
                    unit
                    )
                
            row.append(dist) 
            
            line = LineString([source, destination])
                            
            if len(sources_uid) == len(sources) and len(destinations_uid) == len(destinations):
                
                line_data.append(
                    {
                        config.MATRIX_COL_SOURCE: source,
                        config.MATRIX_COL_DESTINATION: destination,
                        "distance": dist,
                        "geometry": line,
                        f"{config.MATRIX_COL_SOURCE}_uid": sources_uid[i],
                        f"{config.MATRIX_COL_DESTINATION}_uid": destinations_uid[j],
                        f"{config.MATRIX_COL_SOURCE}_{config.MATRIX_COL_DESTINATION}_uid": f"{sources_uid[i]}{config.MATRIX_OD_SEPARATOR}{destinations_uid[j]}"
                        }
                    )
            
            else:
                
                line_data.append(
                    {
                        config.MATRIX_COL_SOURCE: source,
                        config.MATRIX_COL_DESTINATION: destination,
                        "distance": dist,
                        "geometry": line
                        }
                    )
          
        matrix.append(row)

    if verbose:
        print("OK")

    line_data_gdf = gp.GeoDataFrame(line_data)
    line_data_gdf.set_crs(config.WGS84_CRS, inplace=True)
    line_data_gdf = line_data_gdf.to_crs(output_crs)
    
    if len(sources_uid) > 0 and len(sources_uid) == len(sources):
        sources_point_gpd = point_gpd_from_list(
            input_lonlat = sources,
            input_id = sources_uid,
            input_crs = "EPSG:4326",
            output_crs = output_crs,
            save_shapefile = None,
            verbose = False
            )
    else:
        sources_point_gpd = point_gpd_from_list(
            input_lonlat = sources,
            input_id = list(range(len(sources))),
            input_crs = "EPSG:4326",
            output_crs = output_crs,
            save_shapefile = None,
            verbose = False
            )
        
    if len(destinations_uid) > 0 and len(destinations_uid) == len(destinations):
        destinations_point_gpd = point_gpd_from_list(
            input_lonlat = destinations,
            input_id = destinations_uid,
            input_crs = "EPSG:4326",
            output_crs = output_crs,
            save_shapefile = None,
            verbose = False
            )
    else:
        destinations_point_gpd = point_gpd_from_list(
            input_lonlat = destinations,
            input_id = list(range(len(destinations))),
            input_crs = "EPSG:4326",
            output_crs = output_crs,
            save_shapefile = None,
            verbose = False
            )        

    if save_output:        
        try:
            line_data_gdf.to_file(output_filepath)            
            if verbose:      
                print (f"Saved as {output_filepath}")        
        except Exception as e:        
            print(f"WARNING: Saving line geometries as {output_filepath} failed. Error message: {str(e)}")
            
    return [
        matrix,
        line_data_gdf,
        sources_point_gpd,
        destinations_point_gpd
        ]

def distance_matrix_from_gdf(
    sources_points_gdf: gp.GeoDataFrame,
    sources_uid_col: str,
    destinations_points_gdf: gp.GeoDataFrame,
    destinations_uid_col: str,
    distance_type: str = "euclidean",
    unit: str = "m",
    remove_duplicates: bool = True,
    save_output: bool = True,
    output_filepath: str = "lines.shp",
    output_crs: str = "EPSG:4326",
    verbose: bool = False
    ):
    
    if sources_points_gdf.crs != destinations_points_gdf.crs:        
        print(f"NOTE: Sources and destinations have different CRS: {sources_points_gdf.crs}, {destinations_points_gdf.crs}")
            
    sources_points_gdf = sources_points_gdf.to_crs(config.WGS84_EPSG)        
    destinations_points_gdf = destinations_points_gdf.to_crs(config.WGS84_EPSG)
    
    if remove_duplicates:

        if verbose:
            print("Removing duplicates from sources and destinations", end = " ... ")

        sources_points_gdf = sources_points_gdf.drop_duplicates()
        destinations_points_gdf = destinations_points_gdf.drop_duplicates()
    
        if verbose:
            print("OK")

    sources = [[point.x, point.y] for point in sources_points_gdf["geometry"]]   
    sources_uid = list(sources_points_gdf[sources_uid_col])
    
    destinations = [[point.x, point.y] for point in destinations_points_gdf["geometry"]]   
    destinations_uid = list(destinations_points_gdf[destinations_uid_col])
    
    distance_matrix_results = distance_matrix(
        sources = sources,
        destinations = destinations,
        sources_uid = sources_uid,
        destinations_uid = destinations_uid,
        distance_type = distance_type,
        unit = unit,    
        save_output = save_output,
        output_filepath = output_filepath,
        output_crs = output_crs,
        verbose = verbose
    )
    
    return distance_matrix_results   
    
def buffers(
    point_gdf: gp.GeoDataFrame,
    unique_id_col: str,
    distances: list,
    donut: bool = True,
    merge_buffers: bool = False,
    save_output: bool = True,
    output_filepath: str = "buffers.shp",
    output_crs: str = "EPSG:4326",    
    verbose: bool = False   
    ):    
    
    if point_gdf.crs.is_geographic:
        print(f"WARNING: Point GeoDataFrame has geographic coordinate system {point_gdf.crs}. Results may be invalid.")
  
    if verbose:
        print(f"Calculating buffers for {len(point_gdf)} points", end = " ... ")

    if unique_id_col not in point_gdf.columns:
        raise KeyError(f"No column {unique_id_col} in input GeoDataFrame")
        
    all_buffers_gdf = gp.GeoDataFrame(
        columns=[
            unique_id_col, 
            config.DEFAULT_SEGMENTS_COL, 
            "geometry"
            ]
        )

    for idx, row in point_gdf.iterrows():

        point_buffers = []

        for distance in distances:

            point = row["geometry"] 
            point_buffer = point.buffer(distance)

            point_buffer_gdf = gp.GeoDataFrame(
            {
                unique_id_col: row[unique_id_col],
                "geometry": [point_buffer], 
                config.DEFAULT_SEGMENTS_COL: [distance]
                },
                crs=point_gdf.crs
            )
        
            point_buffers.append(point_buffer_gdf)

        point_buffers_gdf = pd.concat(
            point_buffers, 
            ignore_index = True
            )

        if donut:
            point_buffers_gdf = overlay_difference(
                polygon_gdf = point_buffers_gdf, 
                sort_col = config.DEFAULT_SEGMENTS_COL,
                verbose = verbose
                )
 
        all_buffers_gdf = pd.concat(
            [
                all_buffers_gdf,
                point_buffers_gdf
                ], 
            ignore_index = True)

    all_buffers_gdf = all_buffers_gdf.to_crs(output_crs)

    if verbose:
        print("OK")

    if merge_buffers:

        if verbose:
            print("Merging buffers", end = " ... ")

        merged_geom = all_buffers_gdf.union_all()
        all_buffers_gdf = gp.GeoDataFrame(
            {
                unique_id_col: ["all"],
                config.DEFAULT_SEGMENTS_COL: ["merged"],
                "geometry": [merged_geom]
            },
            crs=output_crs
        )

        if verbose:
            print("OK")

    if save_output:
        try:
            all_buffers_gdf.to_file(output_filepath)            
            if verbose:      
                print (f"Saved as {output_filepath}")        
        except Exception as e:        
            print(f"WARNING: Saving buffer geometries as {output_filepath} failed. Error message: {str(e)}")

    return all_buffers_gdf 

def polygon_select(
    gdf: gp.GeoDataFrame,
    gdf_unique_id_col: str,
    gdf_polygon_select: gp.GeoDataFrame,
    gdf_polygon_select_unique_id_col: str,
    distance: int,
    within: bool = False,
    save_output: bool = True,
    output_filepath: str = "polygon_select.shp",
    output_filepath_buffer = "gdf_buffer.shp",
    output_crs: str = "EPSG:4326",
    verbose: bool = False
    ):
    
    if gdf.crs != gdf_polygon_select.crs:
        raise ValueError(f"Coordinate reference systems of inputs do not match. Polygons: {str(gdf.crs)}, points: {str(gdf_polygon_select.crs)}")
        
    if gdf_unique_id_col not in gdf.columns:
        raise KeyError(f"No column {gdf_unique_id_col} in input GeoDataFrame")
    
    if gdf_polygon_select_unique_id_col not in gdf_polygon_select.columns:        
        raise KeyError(f"No column {gdf_polygon_select_unique_id_col} in input GeoDataFrame for selection")
    
    if gdf.crs.is_geographic:
        print(f"WARNING: Input GeoDataFrames have geographic coordinate system {gdf.crs}. Results may be invalid.")
    
    if len(gdf) > 1:
        print(f"WARNING: Input GeoDataFrame 'gdf' includes {len(gdf)} objects. Using the first only.")
        gdf = gdf[0]
    
    if verbose:
        print(f"Selecting from a set of {len(gdf_polygon_select)} polygons", end = " ... ")

    gdf_buffer = buffers(
        point_gdf = gdf,
        unique_id_col = gdf_unique_id_col,
        distances = [distance],        
        save_output = save_output,
        output_filepath = output_filepath_buffer,
        output_crs = output_crs,
        verbose = False
        )
    
    gdf_buffer = gdf_buffer.geometry.union_all()
    
    gdf_polygon_select = gdf_polygon_select.to_crs(output_crs)
    
    gdf_select_intersects = gdf_polygon_select[
        gdf_polygon_select.geometry.intersects(gdf_buffer)
        ]
    
    if within:
        gdf_select_intersects = gdf_select_intersects[gdf_select_intersects.geometry.within(gdf_buffer)]
               
    gdf_select_intersects_unique_ids = gdf_select_intersects[gdf_polygon_select_unique_id_col].unique()
    
    gdf_polygon_select_selection = gdf_polygon_select[gdf_polygon_select[gdf_polygon_select_unique_id_col].isin(gdf_select_intersects_unique_ids)]

    if verbose:
        print("OK")

    if save_output:        
        try:       
            gdf_polygon_select_selection.to_file(output_filepath)            
            if verbose:       
                print (f"Saved as {output_filepath}")                
        except Exception as e:            
            print(f"WARNING: Saving selection data as {output_filepath} failed. Error message: {str(e)}")   
        
    return gdf_polygon_select_selection

def overlay_difference(
    polygon_gdf: gp.GeoDataFrame, 
    sort_col: str = None,
    verbose: bool = False
    ):

    if verbose:
        print(f"Performing overlay difference on {len(polygon_gdf)} polygons", end = " ... ")

    if sort_col is not None:
        polygon_gdf = polygon_gdf.sort_values(by=sort_col).reset_index(drop=True)
    else:
        polygon_gdf = polygon_gdf.reset_index(drop=True)

    new_geometries = []
    new_data = []

    for i in range(len(polygon_gdf) - 1, 0, -1):
        
        current_polygon = polygon_gdf.iloc[i].geometry
        previous_polygon = polygon_gdf.iloc[i - 1].geometry
        difference_polygon = current_polygon.difference(previous_polygon)

        if difference_polygon.is_empty or not difference_polygon.is_valid:
            continue

        new_geometries.append(difference_polygon)
        new_data.append(polygon_gdf.iloc[i].drop("geometry"))

    inner_most_polygon = polygon_gdf.iloc[0].geometry

    if inner_most_polygon.is_valid:

        new_geometries.append(inner_most_polygon)
        new_data.append(polygon_gdf.iloc[0].drop("geometry"))

    polygon_gdf_difference = gp.GeoDataFrame(
        new_data, geometry=new_geometries, crs=polygon_gdf.crs
    )

    if verbose:
        print("OK")

    return polygon_gdf_difference

def point_spatial_join(
    polygon_gdf: gp.GeoDataFrame,
    point_gdf: gp.GeoDataFrame,
    join_type: str = "inner",
    polygon_ref_cols: list = [],
    point_stat_col: str = None,
    check_polygon_ref_cols: bool = False,
    save_output: bool = True,
    output_filepath_join: str = "shp_points_gdf_join.shp",
    output_filepath_stat: str = "spatial_join_stat.csv",
    output_crs: str = "EPSG:4326",
    verbose: bool = False
    ):
    
    if polygon_gdf is None:
        raise ValueError("Parameter 'polygon_gdf' is None")
    if point_gdf is None:
        raise ValueError("Parameter 'point_gdf' is None")
    
    if polygon_gdf.crs != point_gdf.crs:
        raise ValueError(f"Coordinate reference systems of polygon and point data do not match. Polygons: {str(polygon_gdf.crs)}, points: {str(point_gdf.crs)}")
    
    if polygon_ref_cols != [] and check_polygon_ref_cols:
        for polygon_ref_col in polygon_ref_cols:
            if polygon_ref_col not in polygon_gdf.columns:
                raise KeyError (f"Column {polygon_ref_col} not in polygon data")
        
    if point_stat_col is not None:
        if point_stat_col not in point_gdf.columns:
            raise KeyError (f"Column {point_stat_col} not in point data")
        if not is_numeric_dtype(point_gdf[point_stat_col]):
            raise TypeError (f"Column {point_stat_col} is not numeric")
    
    if verbose:
        print(f"Performing spatial join with {len(polygon_gdf)} polygons and {len(point_gdf)} points", end = " ... ")

    shp_points_gdf_join = point_gdf.sjoin(
        polygon_gdf, 
        how=join_type
        )
    
    if verbose:
        print("OK")

    spatial_join_stat = None

    if polygon_ref_cols != [] and point_stat_col is not None:

        if verbose:
            print("Calculation overlay statistics", end = " ... ")

        shp_points_gdf_join_count = shp_points_gdf_join.groupby(polygon_ref_cols)[point_stat_col].count()
        shp_points_gdf_join_sum = shp_points_gdf_join.groupby(polygon_ref_cols)[point_stat_col].sum()
        shp_points_gdf_join_min = shp_points_gdf_join.groupby(polygon_ref_cols)[point_stat_col].min()
        shp_points_gdf_join_max = shp_points_gdf_join.groupby(polygon_ref_cols)[point_stat_col].max()
        shp_points_gdf_join_mean = shp_points_gdf_join.groupby(polygon_ref_cols)[point_stat_col].mean()
        
        shp_points_gdf_join_count = shp_points_gdf_join_count.rename("count").to_frame()
        shp_points_gdf_join_sum = shp_points_gdf_join_sum.rename("sum").to_frame()
        shp_points_gdf_join_min = shp_points_gdf_join_min.rename("min").to_frame()
        shp_points_gdf_join_max = shp_points_gdf_join_max.rename("max").to_frame()
        shp_points_gdf_join_mean = shp_points_gdf_join_mean.rename("mean").to_frame()
        spatial_join_stat = shp_points_gdf_join_count.join(
            [
                shp_points_gdf_join_sum, 
                shp_points_gdf_join_min, 
                shp_points_gdf_join_max,
                shp_points_gdf_join_mean
                ]
            )
        
        if verbose:
            print("OK")

    if save_output:        
        
        shp_points_gdf_join = shp_points_gdf_join.to_crs(crs = output_crs)        
        
        try:
            shp_points_gdf_join.to_file(output_filepath_join)        
            if verbose:       
                print (f"Saved join data as {output_filepath_join}")
        except Exception as e:
            print(f"WARNING: Saving join data as {output_filepath_join} failed. Error message: {str(e)}")        
        
        if polygon_ref_cols != [] and point_stat_col is not None:                
            try:
                spatial_join_stat.to_csv(output_filepath_stat)
                if verbose:       
                    print (f"Saved statistics as {output_filepath_stat}")
            except Exception as e:
                print(f"WARNING: Saving statistics as {output_filepath_stat} failed. Error message: {str(e)}")

    return [
        shp_points_gdf_join,
        spatial_join_stat
        ]
      
def map_with_basemap(
    layers: list,
    layers_auto_crs: bool = True,
    osm_basemap: bool = True,
    zoom: int = 15,
    tile_delay = 0.1,
    figsize=(10, 10),
    bounds_factor = [0.9999, 0.9999, 1.0001, 1.0001],
    styles: dict = {},
    save_output: bool = True,
    output_filepath: str = "osm_map_with_basemap.png",
    output_dpi=300,
    legend: bool = True,
    map_title: str = "Map with OSM basemap",
    show_plot: bool = True,
    verbose: bool = False
    ):
    
    if not isinstance(layers, list):
        raise TypeError("Param 'layers' must be a list")
    
    if not layers:
        raise ValueError("List layers is empty")

    if verbose:
        print("Testing for different CRS", end = " ... ")
    
    crs_layer0 = layers[0].crs

    unique_crs = [layer.crs for layer in layers]
    
    if verbose:
        print("OK")
    
    if len(unique_crs) > 1:
        
        if layers_auto_crs:           
            
            for i, layer in enumerate(layers):
                
                layers[i].crs = crs_layer0
            
            if verbose:
                print(f"NOTE: Input layers have different CRS: {', '.join(unique_crs)}. All layers were automatically converted to CRS {crs_layer0}.")
                
        else:
            raise TypeError(f"The {len(layers)} layers have {len(unique_crs)} different CRS: {', '.join(unique_crs)}.")
        
    else:
        
        if verbose:
            print(f"NOTE: All input layers have the same CRS: {crs_layer0}.")

    if verbose:
        print("Combining layers ...", end = " ")

    layers_combined = gp.GeoDataFrame(
        pd.concat(
            layers, 
            ignore_index=True
            ),
        crs=crs_layer0
    )

    layers_combined_wgs84 = layers_combined.to_crs(crs = config.WGS84_CRS)
    
    if verbose:
        print("OK")        
        print("Retrieving total bounds ...", end = " ")
        
    bounds = layers_combined_wgs84.total_bounds

    sw_lon, sw_lat, ne_lon, ne_lat = bounds[0]*bounds_factor[0], bounds[1]*bounds_factor[1], bounds[2]*bounds_factor[2], bounds[3]*bounds_factor[3]

    if verbose:
        print("OK")        
    
    if osm_basemap:
        
        if verbose:
            print("Retrieving OSM basemap ...", end = " ")
            
        get_basemap(
            sw_lat, 
            sw_lon, 
            ne_lat, 
            ne_lon, 
            zoom=zoom,
            tile_delay=tile_delay,
            verbose=verbose
            )

    fig, ax = plt.subplots(figsize=figsize)

    if osm_basemap:
        
        img = Image.open(config.DEFAULT_FILENAME_ORS_TMP)
        extent_img = [sw_lon, ne_lon, sw_lat, ne_lat]
        ax.imshow(img, extent=extent_img, origin="upper")
        
        if verbose:
            print("OK")

    if verbose:
        print("Inserting layers and plotting map ...", end = " ")
            
    i = 0
    legend_handles = []

    for i, layer in enumerate(layers):
        
        layer_3857 = layer.to_crs(crs = config.PSEUDO_MERCATOR_CRS)

        if styles != {}:
            
            layer_style = styles[i]
            
            if "color" not in layer_style:
                raise KeyError(f"No 'color' key in definition of layer {i}")
            if "name" not in layer_style:
                raise KeyError(f"No 'name' key in definition of layer {i}")
            if "alpha" not in layer_style:
                raise KeyError(f"No 'alpha' key in definition of layer {i}")
            
            if all(layer_3857.geometry.geom_type == "Point"):
                if "size" not in layer_style:
                    raise KeyError(f"No 'size' key in definition of point layer {i}")
            
            layer_color = layer_style["color"]
            layer_alpha = layer_style["alpha"]
            layer_name = layer_style["name"]           

            if all(layer_3857.geometry.geom_type == "Point"):
                
                layer_markersize = layer_style["size"]
                
                if isinstance(layer_color, str):
                    layer_3857.plot(
                        ax=ax,
                        color=layer_color,
                        alpha=layer_alpha,
                        label=layer_name,
                        markersize=layer_markersize
                    )
                    if legend:
                        handle = Line2D(
                            [], 
                            [], 
                            marker='o', 
                            color='w', 
                            markerfacecolor=layer_color, 
                            markersize=config.DEFAULT_LEGEND_POINT_SIZE, 
                            alpha=layer_alpha, 
                            label=layer_name
                        )
                        legend_handles.append(handle)              

                elif isinstance(layer_color, dict):
                    color_key = list(layer_color.keys())[0]
                    color_mapping = layer_color[color_key]

                    if color_key not in layer_3857.columns:
                        raise KeyError(f"Column {color_key} not in layer.")

                    for value, color in color_mapping.items():
                        
                        subset = layer_3857[layer_3857[color_key].astype(str) == str(value)]
                        
                        if not subset.empty:
                            
                            subset.plot(
                                ax=ax,
                                color=color,
                                alpha=layer_alpha,
                                label=str(value),
                                markersize=layer_markersize
                            )
                            
                            if legend:
                                handle = Line2D(
                                    [], 
                                    [], 
                                    marker='o', 
                                    color='w', 
                                    markerfacecolor=color,
                                    markersize=config.DEFAULT_LEGEND_POINT_SIZE, 
                                    alpha=layer_alpha, 
                                    label=str(value)
                                )
                                legend_handles.append(handle)
                                                                
            else:
                
                if isinstance(layer_color, str):
                    layer_3857.plot(
                        ax=ax,
                        color=layer_color,
                        alpha=layer_alpha,
                        label=layer_name,
                    )
                    if legend:
                                               
                        patch = Patch(
                            facecolor=layer_color, 
                            alpha=layer_alpha, 
                            label=layer_name,
                            )
                        legend_handles.append(patch)

                elif isinstance(layer_color, dict):
                    color_key = list(layer_color.keys())[0]
                    color_mapping = layer_color[color_key]

                    if color_key not in layer_3857.columns:
                        raise KeyError(f"Column {color_key} not in layer.")

                    for value, color in color_mapping.items():
                        
                        subset = layer_3857[layer_3857[color_key].astype(str) == str(value)]
                        
                        if not subset.empty:
                            
                            subset.plot(
                                ax=ax,
                                color=color,
                                alpha=layer_alpha,
                                label=str(value),
                            )
                            
                            if legend:
                                
                                patch = Patch(
                                    facecolor=color, 
                                    alpha=layer_alpha, 
                                    label=str(value),
                                    )
                                legend_handles.append(patch)

        else:
            
            layer_3857.plot(
                ax=ax, 
                alpha=config.DEFAULT_LAYER_ALPHA, 
                label=f"{config.DEFAULT_LAYER_LABEL} {i+1}"
                )
            
            if legend:
                
                patch = Patch(
                    facecolor="gray", 
                    alpha=0.6, 
                    label=f"{config.DEFAULT_LAYER_LABEL} {i+1}"
                    )
                
                legend_handles.append(patch)

    bbox = box(sw_lon, sw_lat, ne_lon, ne_lat)
    extent_geom = gp.GeoSeries([bbox], crs = config.WGS84_CRS).to_crs(crs = config.PSEUDO_MERCATOR_CRS).total_bounds
    ax.set_xlim(extent_geom[0], extent_geom[2])
    ax.set_ylim(extent_geom[1], extent_geom[3])

    if osm_basemap:
        
        try:
            
            cx.add_basemap(
                ax,
                source=cx.providers.OpenStreetMap.Mapnik,
                zoom=zoom
            )
            
        except Exception as e:
            
            error_message = f"Error while retrieving basemap from OSM. Error message: {str(e)}"
            
            print(error_message)

    plt.axis('off')

    if legend and legend_handles:
        ax.legend(
            handles=legend_handles, 
            loc=config.DEFAULT_LEGEND_LOC, 
            fontsize=config.DEFAULT_LEGEND_FONTSIZE, 
            frameon=True
            )

    plt.title(map_title)

    if verbose:
        print("OK")
    
    if save_output:
        plt.savefig(
            output_filepath,
            dpi=output_dpi,
            bbox_inches="tight"
        )    

    if show_plot:
        plt.show()

    plt.close()
    
    if os.path.exists(config.DEFAULT_FILENAME_ORS_TMP):
        try:
            os.remove(config.DEFAULT_FILENAME_ORS_TMP)
        except Exception as e:
            error_message = f"Temporary file {config.DEFAULT_FILENAME_ORS_TMP} can not be removed. Error message: {str(e)}"
            print(error_message)
        
    return fig

def point_gpd_from_list(
    input_lonlat: list,
    input_id: list = [],
    input_crs = "EPSG:4326",
    output_crs = "EPSG:4326",
    save_shapefile = None,
    verbose: bool = False
    ):

    if verbose:
        print(f"Calucalating point gdf from list with {len(input_lonlat)} entries", end = " ... ")
    
    input_lonlat_gpd = gp.GeoDataFrame()
    
    for entry, coords in enumerate(input_lonlat):
    
        coords_Point = Point(coords)
    
        coords_gpd = gp.GeoDataFrame(
            [
                {
                    "geometry": coords_Point
                    }
                ], 
            crs=input_crs
        )
        
        if input_id != []:
            coords_gpd["ID"] = input_id[entry]
        else:
            coords_gpd["ID"] = entry
        
        input_lonlat_gpd = pd.concat([input_lonlat_gpd, coords_gpd], ignore_index=True)

    input_lonlat_gpd = input_lonlat_gpd.to_crs(crs=output_crs)

    if verbose:
        print("OK")

    if save_shapefile is not None:
        input_lonlat_gpd.to_file(save_shapefile)

    return input_lonlat_gpd

def lonlat_transform(
    source: list,
    destination: list,
    transform: bool = True
    ):

    lon1 = source[0]
    lat1 = source[1]
    lon2 = destination[0]
    lat2 = destination[1]

    if transform:
        lat1_r = lat1*pi/180
        lon1_r = lon1*pi/180
        lat2_r = lat2*pi/180
        lon2_r = lon2*pi/180

        return lat1_r, lat2_r, lon1_r, lon2_r
    
    else:

        return lat1, lat2, lon1, lon2