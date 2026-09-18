# Two-Step Floating Catchment Area (2SFCA) Analysis

## Background

The *Two-step Floating Catchment Area (2SFCA) Analysis* was developed by Luo and Wang[1] in the context of health geography, more specifically: to measure the spatial accessibility of healthcare providers. The basic idea is to include both the capacity utilization of the providers (supply-to-demand ratio) and their accessibility from the demand locations. Furthermore, a catchment threshold is introduced, which is a maximum distance or travel time up to which supply and demand locations are considered.

## Model formulation

The basic 2SFCA Analysis is calculated as follows [1]:

Step 1: $$R_j = \frac{S_j}{\sum_{k \in \{d_{kj} \leq d_0\}} P_k}$$

where $$R_j$$ is the supply-to-demand ratio at location $$j$$ which is within the catchment threshold, $$P_k$$ is the population of origin $$k$$ which is within the catchment threshold, $$S_j$$ is the number of opportunities at location $$j$$, $$d_{kj}$$ is the distance or travel time between $$k$$ and $$j$$, and $$d_0$$ is the catchment threshold.

Step 2: $$A_i^F = \sum_{j \in \{d_{ij} \leq d_0\}} R_j$$

where $$A_i^F$$ is the accessibility at origin $$i$$ and $$d_{ij}$$ is the distance or travel time between $$i$$ and $$j$$.

## Empirical application

A basic 2SFCA Analysis involves the following steps:

1. Define a study area and divide it into $$I$$ customer origins (e.g., municipalities, ZIP code areas, census tracts)
2. Define a catchment threshold $$d_0$$
3. Identify the relevant $$J$$ supply locations within the study area
4. Collect the size values $$S_j$$ (e.g., opportunities) of all $$J$$ supply locations and the local populations $$P_k$$
5. [Calculate travel costs](#calculation-of-travel-costs) $$d_{kj}$$ and $$d_{ij}$$ (up to $$d_0$$) and store them travel cost matrices
7. Calculate $$R_j$$ for all supply locations
8. Sum $$R_j$$ over all customer origins to calculate $$A_i^F$$

## Further notes

The empirical application of the 2SFCA Analysis has the same requirements as a Huff Model analysis, including the calculation of travel costs for all $$I \times J$$ combinations of customer origins and supply locations. For more information on the calculation of a travel cost matrix, see the corresponding [Huff Model](huff-model.md#calculation-of-travel-costs) section.

The approach was extended by including [distance decay functions](huff-model.md#weighting-functions) from market area models instead of defining a distance threshold[2][3].

For a broader discussion of 2SFCA Analysis, including a comparison with other approaches to modeling accessibility, see the paper by Rauch et al.[4]

## References

[1] Luo W, Wang F (2003) Measures of spatial accessibility to health care in a GIS environment: synthesis and a case study in the Chicago region. *Environment and Planning B: Planning and Design* 30: 865-884. [10.1068/b29120](https://doi.org/10.1068/b29120)

[2] Luo J (2014) Integrating the Huff Model and Floating Catchment Area Methods to Analyze Spatial Access to Healthcare Services. *Transactions in GIS* 18(3): 436-448. [10.1111/tgis.12096](https://doi.org/10.1111/tgis.12096)

[3] Subal J, Paal P, Krisp JM (2021) Quantifying spatial accessibility of general practitioners by applying a modified huff three-step floating catchment area (MH3SFCA) method. *International Journal of Health Geography* 20: 9. [10.1186/s12942-021-00263-3](https://doi.org/10.1186/s12942-021-00263-3)

[4] Rauch S, Stangl S, Haas T, Rauh J, Heuschmann PU (2023) Spatial inequalities in preventive breast cancer care: A comparison of different accessibility approaches for prevention facilities in Bavaria, Germany. *Journal of Transport & Health* 29: 101567. [10.1016/j.jth.2023.101567](https://doi.org/10.1016/j.jth.2023.101567)