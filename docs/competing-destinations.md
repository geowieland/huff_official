# Competing Destinations Model

## Background

Fotheringham's *Competing Destinations Model*[1] is an extension of the [Huff Model](huff-model.md) which aims to capture the spatial structure of supply more comprehensively by taking into account (potential) positive agglomeration effects. [Economies of agglomeration](https://en.wikipedia.org/wiki/Economies_of_agglomeration) are a popular subject of study in spatial economics, from both theoretical and empirical perspectives. In retail and service industries, such positive effects may arise from the clustering of competing and complementary providers, driven by multipurpose and comparison shopping[2][3][4]. 

## Model formulation

There are $$I$$ customer origins ($$i = 1, ..., I$$) and $$J$$ supply locations ($$j = 1, ..., J$$). The Competing Destinations Model has the following probability equation[1]:

$$p_{ij} = \frac{A_j^{\gamma} \exp{-\lambda t_{ij}} C_j^{\beta}}{\sum_{j=1}^J A_j^{\gamma} \exp{-\lambda t_{ij}} C_j^{\beta}}$$

where $$A_j$$ is the attraction (size) of supply location $$j$$, $$t_{ij}$$ is the travel time from $$i$$ to $$j$$, $$C_j$$ is the so-called *relative location* of supplier $$j$$ with respect to its $$K$$ competitors ($$k=1,2,...,K$$, $$j \neq k$$), and $$\gamma$$, $$\lambda$$ and $$\beta$$ are weighting parameters.

The $$\beta$$ parameter reflects the impact of the relative location: if $$\beta$$ is positive, a higher value of $$C_j$$ increases the probability of choosing location $$j$$, which may be interpreted as positive agglomeration effects due to the clustering of competitors. A negative value would indicate that competitive effects dominate[1]. 

In defining the indicator of spatial concentration, Fotheringham refers to the [Hansen Accessibility](hansen-accessibility.md)[5]. It is thus defined as follows[1]:

$$C_j = \sum_{k=1, j \neq k}^K \frac{A_k^{\alpha}}{t_{jk}^{\delta}}$$

where $$C_j$$ is relative location of supplier $$j$$, $$A_k$$ is the attraction (size) of competitor $$k$$, $$t_{jk}$$ is the travel cost from $$j$$ to $$k$$, $$K$$ is the number of competitors, and $$\alpha$$ and $$\delta$$ are weighting parameters. As in the original Hansen accessibility, $$\alpha$$ is implicitly set to 1 and thus not displayed.

## Further notes

The empirical application of the Competing Destinations Model has the same requirements as a Huff Model analysis, including the calculation of travel costs for all $$I \times J$$ combinations of customer origins and supply locations. Additionally, travel costs between the $$J$$ locations and their $$K$$ competitors need to be calculated. For more information on the calculation of a travel cost matrix, see the corresponding [Huff Model](huff-model.md#calculation-of-travel-costs) section.

There are a few studies that have empirically applied the Competing Destinations Model, particularly in combination with [iterative parameter estimation](huff-fitting.md) or the [MCI Model](mci-model.md). The reported empirical effects vary across studies and retail contexts: Orpana and Lampinen[6] found exclusively negative effects for grocery stores when categorized by store format. In contrast, Wieland[4] identified positive agglomeration effects for clusters of supermarkets and discount grocery stores, but negative effects for supermarket-supermarket or discounter-discounter combinations, as well as positive agglomeration effects for consumer electronics stores. Li and Liu[7] empirically determined a distance threshold between competitors that separates agglomeration advantages from competitive disadvantages.


## References

[1] Fotheringham AS (1985) Spatial Competition and Agglomeration in Urban Modelling. *Environment and Planning A: Economy and Space* 17(2): 213-230. [10.1068/a170213](https://doi.org/10.1068/a170213)

[2] Mulligan GF, Partridge MD, Carruthers JI (2012) Central place theory and its re-emergence in Regional Science. *The Annals of Regional Science* 48(2): 405-431. [10.1007/s00168-011-0496-7](https://doi.org/10.1007/s00168-011-0496-7)

[3] Popkowski Leszczyc PTL, Sinha A, Sahgal A (2004) The effect of multi-purpose shopping on pricing and location strategy for grocery stores. *Journal of Retailing* 80(2): 85-99. [10.1016/j.jretai.2004.04.006](https://doi.org/10.1016/j.jretai.2004.04.006)

[4] Wieland T (2015) *Räumliches Einkaufsverhalten und Standortpolitik im Einzelhandel unter Berücksichtigung von Agglomerationseffekten - Theoretische Erklärungsansätze, modellanalytische Zugänge und eine empirisch-ökonometrische Marktgebietsanalyse anhand eines Fallbeispiels aus dem ländlichen Raum Ostwestfalens/Südniedersachsens*. Geographische Handelsforschung 23. Mannheim: MetaGIS. https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753

[5] Hansen WG (1959) How Accessibility Shapes Land Use. *Journal of the American Institute of Planners* 25(2): 73-76. [10.1080/01944365908978307](https://doi.org/10.1080/01944365908978307)

[6] Orpana T, Lampinen J (2003) Building Spatial Choice Models from Aggregate Data. *Journal of Regional Science* 43(2): 319-348. [10.1111/1467-9787.00301](https://doi.org/10.1111/1467-9787.00301)

[7] Li Y, Liu L (2012) Assessing the impact of retail location on store performance: A comparison of Wal-Mart and Kmart stores in Cincinnati. *Applied Geography* 32(2): 591-600. [10.1016/j.apgeog.2011.07.006]https://doi.org/10.1016/j.apgeog.2011.07.006