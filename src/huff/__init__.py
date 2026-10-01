def info():

    from huff.config import PACKAGE_NAME, PACKAGE_VERSION, PYPI_HUFF_URL, GITHUB_HUFF_URL, GITHUB_HANDBOOK_URL, HUFF_SOFTWAREPAPER_URL, OSM_TILES_SERVER, OSM_ATTRIBUTION, ORS_SERVER, ORS_ATTRIBUTION

    print(f"{PACKAGE_NAME} v{PACKAGE_VERSION}")
    print(f"GitHub: {GITHUB_HUFF_URL}")
    print(f"PyPI: {PYPI_HUFF_URL}")
    print(f"Software paper: {HUFF_SOFTWAREPAPER_URL}")
    print(f"Methodological handbook: {GITHUB_HANDBOOK_URL}")

    print(f"\nOpenStreetMap tiles server set to: {OSM_TILES_SERVER}")
    print(OSM_ATTRIBUTION)
    print("Use huff.osm.define_tiles_server() and huff.osm.define_headers() to change OSM configuration")

    print(f"\nOpenRouteService server set to: {ORS_SERVER}")
    print(ORS_ATTRIBUTION)
    print("Use huff.ors.define_ors_server(), huff.ors.define_headers(), and huff.ors.define_ors_auth() to change ORS configuration")