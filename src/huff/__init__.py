#-----------------------------------------------------------------------
# Name:        huff
# Purpose:     Market Area Analysis in Python
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.9.13
# Last update: 2026-10-05 19:21
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import requests


def info():

    from huff.config import PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_AUTHOR, PACKAGE_AUTHOR_EMAIL, PYPI_HUFF_URL, GITHUB_HUFF_URL, GITHUB_HANDBOOK_URL, HUFF_SOFTWAREPAPER_URL, OSM_TILES_SERVER, OSM_ATTRIBUTION, ORS_SERVER, ORS_ATTRIBUTION, MODELS, PERMITTED_WEIGHTING_FUNCTIONS, MODEL_WRAPPER_AVAILABLE, ORS_PROFILES

    print(f"{PACKAGE_NAME} v{PACKAGE_VERSION}")
    print(f"Author: {PACKAGE_AUTHOR} (EMail: {PACKAGE_AUTHOR_EMAIL})")
    print(f"GitHub: {GITHUB_HUFF_URL}")
    print(f"PyPI: {PYPI_HUFF_URL}")
    print(f"Software paper: {HUFF_SOFTWAREPAPER_URL}")
    print(f"Methodological handbook: {GITHUB_HANDBOOK_URL}")

    print("\nAvailable market area and accessibility models:")
    for model_name, model_info in MODELS.items():
        print(f"  - {model_name}: {model_info['description']}")

    print("\nAvailable weighting functions:")
    for weighting_function_name, weighting_function_info in PERMITTED_WEIGHTING_FUNCTIONS.items():
        print(f"  - {weighting_function_name}: {weighting_function_info['description']}")

    print("\nAvailable machine learning models:")
    for model_name, model_info in MODEL_WRAPPER_AVAILABLE.items():
        print(f"  - {model_name}: {model_info}")

    osm_responding = True
    try:
        requests.get(OSM_TILES_SERVER, timeout=10)
    except requests.RequestException:
        osm_responding = False

    print(f"\nOpenStreetMap tiles server set to: {OSM_TILES_SERVER}")
    print(f"Server responding: {osm_responding}")
    print(OSM_ATTRIBUTION)
    print("Use huff.osm.define_tiles_server() and huff.osm.define_headers() to change OSM configuration")

    ors_responding = True
    try:
        requests.get(ORS_SERVER, timeout=10)
    except requests.RequestException:
        ors_responding = False

    print(f"\nOpenRouteService server set to: {ORS_SERVER}")
    print(f"Server responding: {ors_responding}")
    print(ORS_ATTRIBUTION)
    print("Use huff.ors.define_ors_server(), huff.ors.define_headers(), and huff.ors.define_ors_auth() to change ORS configuration")
    print(f"Available ORS profiles:")
    for profile_info, profile_name in ORS_PROFILES.items():
        print(f"  - {profile_name}: {profile_info}")