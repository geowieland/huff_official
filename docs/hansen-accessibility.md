# Hansen Accessibility

## Background

The *Hansen Accessibility*, named after him, was developed by Walter G. Hansen in the context of urban planning and land use[1]. Starting from an origin (e.g., place of residence), it represents the sum of all opportunities (e.g., jobs, shopping facilities, leisure activities, medical services), weighted by travel costs.

## Model formulation

The Hansen Accessibility is formalized as follows[1]:

$$A_i = \sum_{j=1}^J O_j f(d_{ij})$$

where $$A_i$$ is the weighted accessibility from origin $$i$$, $$O_j$$ is the number of opportunities at location $$j$$ ($$j=1,2,...,J$$), and $$d_{ij}$$ is the distance or travel time between $$i$$ and $$j$$.

The [distance decay function](huff-model.md#weighting-functions) $$f(d_{ij})$$ is assumed to be a nonlinear function, e.g., power function.

## Empirical application

Calculating the Hansen Accessibility involves the following steps:

1. Define a study area and divide it into $$I$$ customer origins (e.g., municipalities, ZIP code areas, census tracts)
2. Identify the relevant $$J$$ supply locations within the study area
3. Collect the size values $$O_j$$ (e.g., opportunities) of all $$J$$ supply locations
4. [Calculate travel costs](#calculation-of-travel-costs) $$d_{ij}$$ for all $$I \times J$$ origin-destination combinations and store them in a travel cost matrix
5. [Define a distance decay function](#weighting-functions) and the corresponding parameter(s) for $$t_{ij}$$ (and, if required, define a weighting for $$O_j$$ as well)
6. Calculate $$A_i$$ for all $$I$$ origins

## Further notes

The empirical application of the Hansen Accessibility has the same requirements as a Huff Model analysis, including the calculation of travel costs for all $$I \times J$$ combinations of customer origins and supply locations. For more information on the calculation of a travel cost matrix, see the corresponding [Huff Model](huff-model.md#calculation-of-travel-costs) section.

The Hansen Accessibility served as the model for incorporating cluster effects into the [Competing Destinations Model](competing-destinations.md).

Harris[2] developed a very similar indicator for market potential from the provider's perspective:

$$M_j = \sum^I_{i=1} O_i d_{ij}^{-1}$$

where $$M_j$$ is the market potential of supplier $$j$$, $$O_i$$ is the market potential at origin $$i$$, $$d_{ij}$$ is the distance or travel time between $$i$$ and $$j$$, and $$I$$ equals the number of customer origins.

The principle of Hansen Accessibility (a weighted sum of all options) may be applied to more complex indicators parameterized using empirical-econometric market area or choice models[3].

## References

[1] Hansen WG (1959) How Accessibility Shapes Land Use. *Journal of the American Institute of Planners* 25(2): 73-76. [10.1080/01944365908978307](https://doi.org/10.1080/01944365908978307)

[2] Harris CD (1954) The Market as a Factor in the Localization of Industry in the United States. *Annals of the Association of American Geographers* 44(4): 315–348. [10.1080/00045605409352140](https://doi.org/10.1080/00045605409352140)

[3] Rauch S, Wieland T, Rauh J (2025) Accessibility of food - A multilevel approach comparing a choice based model with perceived accessibility in Mainfranken. *Journal of Transport Geography* 128: 104367. [10.1016/j.jtrangeo.2025.104367](https://doi.org/10.1016/j.jtrangeo.2025.104367)