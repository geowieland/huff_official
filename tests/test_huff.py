#-----------------------------------------------------------------------
# Name:        tests (huff package)
# Purpose:     Tests for Huff Model package functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-10-03 10:47
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------


def test_import_huff():
    """Test if huff package can be imported and info() function works."""

    import huff

    huff.info()


def test_huff():
    """Test if huff package functions work with test data."""

    from pathlib import Path

    from huff.models import create_interaction_matrix
    from huff.data_management import load_geodata

    # Dealing with customer origins (statistical districts):

    test_data_haslach_shp = str(
        Path(__file__).parent / "data" / "Haslach.shp"
    )

    Haslach = load_geodata(
        test_data_haslach_shp,
        location_type="origins",
        unique_id="BEZEICHN"
        )
    # Loading customer origins (shapefile)

    assert Haslach is not None

    Haslach_buf = Haslach.buffers(
        segments_distance=[500,1000,1500],
        save_output=False,
        output_crs="EPSG:31467"
        )
    # Buffers for customer origins

    assert Haslach_buf is not None

    Haslach.summary()
    # Summary of customer origins

    Haslach.define_marketsize("pop")
    # Definition of market size variable

    Haslach.define_transportcosts_weighting(
        param_lambda = -2.2,    
        # one weighting parameter for power function (default)
        # two weighting parameters for logistic function
        )
    # Definition of transport costs weighting (lambda)

    Haslach.summary()
    # Summary after update

    Haslach.show_log()
    # Show log of CustomerOrigins object

    # Dealing with supply locations (supermarkets):

    test_data_haslach_supermarkets_shp = str(
        Path(__file__).parent / "data" / "Haslach_supermarkets.shp"
    )

    Haslach_supermarkets = load_geodata(
        test_data_haslach_supermarkets_shp,
        location_type="destinations",
        unique_id="LFDNR"
        )
    # Loading supply locations (shapefile)

    assert Haslach_supermarkets is not None

    Haslach_supermarkets.summary()
    # Summary of supply locations

    Haslach_supermarkets.define_attraction("VKF_qm")
    # Defining attraction variable

    Haslach_supermarkets.define_attraction_weighting(
        attrac_var = "VKF_qm",
        param_gamma=0.9
        )
    # Define attraction weighting (gamma)

    print(Haslach_supermarkets.get_metadata())

    Haslach_supermarkets.summary()
    # Summary of supermarkets

    Haslach_supermarkets.show_log()
    # Log of supermarkets object

    # Using customer origins and supply locations for building interaction matrix:

    haslach_interactionmatrix = create_interaction_matrix(
        Haslach,
        Haslach_supermarkets
        )
    # Creating interaction matrix

    assert haslach_interactionmatrix is not None

    haslach_interactionmatrix.transport_costs(
        network=False,
        distance_unit="meters",
        )
    # Calculating transport costs (airline distance) for interaction matrix

    haslach_interactionmatrix.summary()
    # Summary of interaction matrix

    haslach_interactionmatrix.flows()
    # Calculating spatial flows for interaction matrix

    huff_model = haslach_interactionmatrix.marketareas()
    # Calculating total market areas
    # Result of class HuffModel

    assert huff_model is not None

    huff_model.summary()
    # Summary of Huff model

    huff_model.show_log()
    # Log of Huff model

    huff_model_interactionmatrix = huff_model.get_interaction_matrix_df()
    print(huff_model_interactionmatrix)
    # Showing interaction matrix

    huff_model_marketareas = huff_model.get_market_areas_df()
    print(huff_model_marketareas)
    # Showing total market areas


    # Maximum Likelihood fit for Huff Model:

    haslach_interactionmatrix.huff_ml_fit(
        initial_params=[1, -2],
        method="trust-constr",
        bounds = [(0.8, 0.9999),(-2.5, -1.5)]    
    )
    # Maximum Likelihood fit for Huff Model

    haslach_interactionmatrix.summary()
    # Summary of fitted ML-fitted interaction matrix (Huff model)

    huff_model_fit = haslach_interactionmatrix.marketareas()
    # Calculcation of total market areas
    # Result of class HuffModel

    assert huff_model_fit is not None

    huff_model_fit.summary()
    # Huff model summary


    # Multiplicative Competitive Interaction Model:

    mci_fit = huff_model.mci_fit(verbose=True)
    # Fitting via MCI

    mci_fit.show_log()

    mci_fit.summary()
    # Summary of MCI model

    mci_fit.marketareas()
    # MCI model market simulation

    mci_fit.get_market_areas_df()
    # MCI model market areas

    mci_fit.show_log()
    # Log of MCIModel object

    assert mci_fit is not None