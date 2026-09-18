# Huff Model: Local Optimization of Attraction

## Background

Güssefeldt[1] developed an algorithm for fitting the [Huff Model](huff-model.md) iteratively in cases when only the total turnovers of supply locations, $$T_j$$ are available (e.g., via firm surveys or official statistics). This algorithm was then simplified by Wieland[2]. The underlying rationale is that supply locations employ their factors of production differently, though this need not be reflected in their structural indicators (e.g., size). Consequently, their attraction variable must be adjusted with respect to actual turnover, which is conducted iteratively for any supply location, with a defined threshold of acceptable deviation[1][2].

## Definitions and calculations

In the Huff Model, the total customers or total turnover of a supply location $$j$$ (from now called turnover), $$T_j$$ is a function of its own factors of production and the behavior of all $$J-1$$ competitors. More precisely, the turnover depends on the attraction of the location, holding all competitors constant. Every $$j$$ location has its own relationship between size (input) and turnover (output). Vice versa, the attraction may be described as a function of the turnover[1][2]:

$$A_j = a_j + b_j T_j$$

where $$A_j$$ is the attraction (size) of supply location $$j$$, $$T_j$$ is the total turnover of supply location $$j$$, and $$a_j$$ and $$b_j$$ are the intercept and slope of the location-specific function, respectively.

In the basic Huff Model, two locations, 1 and 2, will generate identical total turnover ($$T_1=T_2$$) if (a) they have an identical attraction value ($$A_1=A_2$$) and (b) the spatial structure of competing and demand locations is identical. However, their actual turnover may vary significantly because they employ their factors of production differently, i.e., in ways that the researcher cannot trace. Therefore, from now, we distinguish between two types of total turnover: There is an *observed* turnover for each of the $$J$$ supply locations, which was empirically obtained. The Huff Model calculates the *expected* turnover based on model assumptions. The deviation of the expected turnover of location $$j$$ from its observed turnover may be expressed by the *absolute percentage error*[1][2]:

$$APE_j = \frac{|T_{j_{exp}}-T_{j_{obs}}|}{T_{j_{obs}}} \times 100$$

where $$APE_j$$ is the absolute percentage error with respect to location $$j$$, $$T_{j_{obs}}$$ is the actual turnover of location $$j$$, and $$T_{j_{exp}}$$ is the expected turnover of location $$j$$.

The goal of this algorithm is therefore to adjust the attraction value of each individual location to its actual turnover. This is possible by parameterizing the function that describes the relationship between attraction and turnover, i.e., to find the location-specific parameters $$a_j$$ and $$b_j$$ (see above). From now, we distinguish between two types of attraction as well: the *observed* attraction value and an *adjusted* attraction of location $$j$$. In the Huff Model, if $$A_j$$ is equal to zero, also $$T_j$$ must be. Therefore, the intercept $$a_j$$ of the equation above is equal to zero, and there are two known pairs of data. The slope for each $$j$$ supply location may be calculated using the difference quotient[1]:

$$b_j = \frac{A_{j_{obs}} - A_{j_{T_j=0}}}{T_{j_{exp}} - T_{j_{A_j=0}}} = \frac{A_{j_{obs}}}{T_{j_{exp}}}$$

where $$A_{j_{obs}}$$ is the observed attraction of location $$j$$, $$A_{j_{T_j=0}}$$ is the attraction of location $$j$$ in the case that $$T_j=0$$, and $$T_{j_{A_j=0}}$$ is the turnover of location $$j$$ when $$A_j=0$$.

Now we calculate an attraction value which is adjusted to the observed output value:

$$A_{j_{adj}} = b_j T_{j_{obs}}$$

where $$A_{j_{adj}}$$ is the adjusted turnover of location $$j$$.

## Empirical application

The attraction values ​​are adjusted step-by-step by the algorithm, performing Huff Model calculations over a number of $$N$$ iterations[2]:

1. Define a study area and divide it into $$I$$ customer origins (e.g., municipalities, ZIP code areas, census tracts)
2. Identify the relevant $$J$$ supply locations competing within the study area
3. Collect the attraction values $$A_j$$ (e.g., size) of all $$J$$ supply locations and the customer potentials $$C_i$$ (e.g., people, EUR, $) of all customer origins
4. Collect total customers or turnover of all $$j$$ supply locations
4. [Calculate travel costs](#calculation-of-travel-costs) $$t_{ij}$$ for all $$I \times J$$ origin-destination combinations and store them in a travel cost matrix
5. [Define a travel cost weighting function](#weighting-functions) and the corresponding parameter(s) for $$t_{ij}$$
6. Set a tolerance value $$maxtol(APE_j)$$ to define which difference between the real and the expected turnovers of location $$j$$ is accepted, e.g. 5%
7. Calculate utilities $$U_{ij}$$, probabilities $$p_{ij}$$, and expected customer/expenditure flows $$E_{ij}$$
8. Sum the expected values $$E_{ij}$$ for each supply location as $$T_{j}$$
9. Calculate the $$APE_j$$ for all $$J$$ locations
10. For any $$j$$ location ($$j=1,2,...,J$$): 
    - If $$APE_j \le maxtol(APE_j)$$: No further local optimization for location $$j$$ is required. Continue with location $$j + 1$$.
    - If $$APE_j > maxtol(APE_j)$$: 
        - Calculate the slope of the attraction function, $$b_j$$
        - Calculate the adjusted attraction of $$j$$, $$A_{j_{adj}}$$
        - Set $$A_j$$ in the [interaction matrix](huff-model.md#empirical-application) to $$A_{j_{adj}}$$
        - Repeat steps 8 to 10
        - Continue with location $$j + 1$$
11. Calculate evaluation metrics for the global fit of the estimated model such as [RMSE](https://en.wikipedia.org/wiki/Root_mean_square_deviation), [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error), or [R-Squared](https://en.wikipedia.org/wiki/Coefficient_of_determination)
12. Repeat steps 8 to 12 for the complete location system up to $$N$$ times and/or until the local optima and/or the global optimum is sufficiently approximated

If the goal of the Huff analysis is to model changes in market areas (e.g., due to new supply locations, changes in location sizes, or changes in travel costs), define the respective scenario and re-calculate the Huff Model:

14. Add new supply locations, update existing locations, or update travel costs
15. Repeat steps 7 to 12
16. Compare market areas and total expected customers/expenditures

## Further notes

The local optimization of attraction algorithm was tested in its original or simplified form in a few studies. In all cases, after a certain number of $$N$$ iterations, a very good fit was observed between the observed $$T_{j_{obs}}$$ and the modeled $$T_{j_{exp}}$$[1][2][3][4][5].


## References

[1] Güssefeldt J (2002) Zur Modellierung von räumlichen Kaufkraftströmen in unvollkommenen Märkten. *Erdkunde* 56(4): 351–370. [10.3112/erdkunde.2002.04.02](https://doi.org/10.3112/erdkunde.2002.04.02)

[2] Wieland T (2017) Market Area Analysis for Retail and Service Locations with MCI. *R Journal* 9(1): 298-323. [10.32614/RJ-2017-020](https://doi.org/10.32614/RJ-2017-020)

[3] Güssefeldt J (2003) Empirische Aspekte einiger Modelle der "New Economic Geography" im Kontext jüngerer Entwicklungen des Einzelhandels. *Die Erde* 134(1): 81-110.

[4] Fittkau D (2004) *Beeinflussung regionaler Kaufkraftströme durch den Autobahnlückenschluß der A 49 Kassel-Gießen - Zur empirischen Relevanz der New Economic Geography in wirtschaftsgeographischen Fragestellungen*. PhD thesis, Georg-August-Universität Göttingen. [10.53846/goediss-3024](https://doi.org/10.53846/goediss-3024).

[5] Wieland T (2015) *Nahversorgung im Kontext raumökonomischer Entwicklungen im Lebensmitteleinzelhandel: Konzeption und Durchführung einer GIS-gestützten Analyse der Strukturen des Lebensmitteleinzelhandels und der Nahversorgung in Freiburg im Breisgau*. Göttingen: GOEDOC. http://resolver.sub.uni-goettingen.de/purl/?webdoc-3965