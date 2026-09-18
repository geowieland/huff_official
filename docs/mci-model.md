# Multiplicative Competitive Interaction (MCI) Model

## Background

The *Multiplicative Competitive Interaction (MCI) Model* was developed by Masao Nakanishi and Lee G. Cooper[1][2][3]. It is both a generalization of the [Huff Model](huff-model.md) and a linear transformation that transforms the Huff Model into an econometric model in which the weighting parameters may be estimated based on empirical data. An MCI Model analysis thus provides statistical inference ($$t$$ value, $$p$$ value, confidence intervals) for the weighting parameters. Performing a market area analysis using the MCI Model requires observed data on regional customer or expenditure flows, $$E_{ij}$$, or the corresponding regional market shares, $$p_{ij}$$[4]. The MCI Model is used in empirical studies to test hypotheses on the impact of marketing or location variables on customer patronage and/or to improve the fit of the Huff Model with respect to real-world data[4][5][6][7][8][9].

## Model formulation

The Multiplicative Competitive Interaction Model is formalized as follows [1]:

$$p_{ij} = \frac{\prod_{h=1}^H A_{h_j}^{\gamma_h}}{\sum_{j=1}^J \prod_{h=1}^H A_{h_j}^{\gamma_h}}$$

where $$A_{h_j}$$ is the $$h$$-th characteristic of supplier $$j$$, and $$\gamma_h$$ is the corresponding weighting coefficient.

This model equation is linearized with the *log-centering transformation*[1]:

$$\log \left(\frac{p_{ij}}{\widetilde{p}_i} \right) = \sum_{h=1}^H \hat{\gamma}_h \log \frac{A_{h_j}}{\widetilde{A}_h} + \log \left( \frac{\epsilon_{ij}}{\widetilde{\epsilon}_i} \right)$$

where $$\widetilde{p}_i$$, $$\widetilde{A}_h$$, and $$\widetilde{\epsilon}_i$$ are the geometric means of $$p_{ij}$$, $$A_h$$, and $$\epsilon_{ij}$$, and $$\epsilon_{ij}$$ denotes the stochastic error term.

The expected probabilities are thus:

$$\hat{p}_{ij} = \frac{\prod_{h=1}^H A_{h_j}^{\hat{\gamma}_h}}{\sum_{j=1}^J \prod_{h=1}^H A_{h_j}^{\hat{\gamma}_h}}$$

where $$\hat{p}_{ij}$$ is the estimated interaction probability (or market share) of origin $$i$$ with respect to location $$j$$.

Instead of substituting the estimated parameters into the formula above, one can also use the *inverse log-centering transformation*[2]:

$$\hat{p}_{ij} = \frac{\exp{\sum_{h=1}^H \hat{\gamma}_h \log \frac{A_{h_j}}{\widetilde{A}_{h_j}}}} {\sum_{j=1}^J \exp{\sum_{h=1}^H \hat{\gamma}_h \log \frac{A_{h_j}}{\widetilde{A}_{h_j}}}}$$

## Empirical application

A market area analysis using the MCI Model involves the following steps:

1. Define a study area and divide it into $$I$$ customer origins (e.g., municipalities, ZIP code areas, census tracts)
2. Collect shopping trips and/or expenditures on the individual/household level and aggregate them at the customer origins level ($$E_{ij}$$, $$p_{ij}$$)
3. Identify the relevant $$J$$ supply locations competing within the study area
4. Collect the attraction values $$A_j$$ (e.g., size) of all $$J$$ supply locations and, if necessary, further demand- or supply-specific variables
5. [Calculate travel costs](huff-model.md#calculation-of-travel-costs) $$t_{ij}$$ for all $$I \times J$$ origin-destination combinations and store them in a travel cost matrix
6. If necessary, correct or transform the variables to match the requirements of the log-centering transformation
7. Apply the log-centering transformation to the previously created interaction matrix
8. Estimate the OLS regression model
9. Calculate expected probabilities $$p_{ij}$$, and expected customer/expenditure flows $$E_{ij}$$
10. Sum the expected values $$E_{ij}$$ for each supply location as $$T_{j}$$
11. Calculate evaluation metrics for the global fit of the estimated model such as [RMSE](https://en.wikipedia.org/wiki/Root_mean_square_deviation), [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error), or [R-Squared](https://en.wikipedia.org/wiki/Coefficient_of_determination)

If the goal of the MCI analysis is to model changes in market areas (e.g., due to new supply locations, changes in location sizes, or changes in travel costs), define the respective scenario and re-calculate the estimated MCI Model:

12. Add new supply locations, update existing locations, or update travel costs
13. Repeat steps 9 and 10
14. Compare market areas and total expected customers/expenditures

## Further notes

The empirical application of the MCI Model has the same requirements as a Huff Model analysis, including the calculation of travel costs for all $$I$$ x $$J$$ combinations of customer origins and supply locations (in the case that travel times are included in the $$H$$ utility variables). For more information on the calculation of a travel cost matrix, see the corresponding [Huff Model](huff-model.md#calculation-of-travel-costs) section.

Customer store choices are typically inquired via household surveys, asking for, e.g., typical or previous shopping trips[4][10]. 

Variables entering the MCI Model must have values greater than zero. Dummy variables do not need to be (and cannot be) transformed using the log-centering transformation; they may be included using the inverse log-centering transformation[4]. In cases where interval-scaled variables, which may take on negative values ​​(e.g., scoring values), are included in the model instead of ratio-scaled variables, Cooper and Nakanishi propose the zeta-squared transformation[3]:

$$z_{h_j} = \frac{A_{h_{j}}-\bar{A_{h}}}{\sigma_{A_{h}}}$$

where $$z_{h_j}$$ is the z-score of $$A_{h_{j}}$$, and $$\bar{A_{h}}$$ and $$\sigma_{A_{h}}$$ are the arithmetic mean and the standard deviation of $$A_{h_{j}}$$.

$$\zeta_{h_j} = \begin{cases} 1+z_{h_j}^2 & \text{if } z_{h_j > 0} \\ \frac{1}{1+z_{h_j}}^2 & \text{if } z_{h_j} \le 0 \end{cases}$$

where $$\zeta_{h_j}$$ is the zeta-squared value of $$A_{h_{j}}$$.


## References

[1] Nakanishi M, Cooper LG (1974) Parameter estimation for a Multiplicative Competitive Interaction Model: Least squares approach. *Journal of Marketing Research* 11(3): 303–311. [10.2307/3151146](https://doi.org/10.2307/3151146).

[2] Nakanishi M, Cooper LG (1982) Technical Note — Simplified Estimation Procedures for MCI Models. *Marketing Science* 1(3): 314-322. [10.1287/mksc.1.3.314](https://doi.org/10.1287/mksc.1.3.314)

[3] Cooper LG, Nakanishi M (1983) Standardizing Variables in Multiplicative Choice Models. *Journal of Consumer Research* 10(1): 96–108. [10.1086/208948](https://doi.org/10.1086/208948)

[4] Wieland T (2017) Market Area Analysis for Retail and Service Locations with MCI. *R Journal* 9(1): 298-323. [10.32614/RJ-2017-020](https://doi.org/10.32614/RJ-2017-020)

[5] Baviera-Puig A, Buitrago-Vera J, Escriba-Perez C (2016) Geomarketing models in supermarket location strategies. *Journal of Business Economics and Management* 17(6): 1205–1221. [10.3846/16111699.2015.1113198](https://doi.org/10.3846/16111699.2015.1113198)

[6] Latruwe T, Van der Wee M, Vanleenhove P, Michielsen K, Verbrugge S, Colle D (2023) Improving inpatient and daycare admission estimates with gravity models. *Health Services and Outcomes Research Methodology* 23, 452–467. [10.1007/s10742-022-00298-4](https://doi.org/10.1007/s10742-022-00298-4)

[7] Oruc N, Tihi B (2013) Competitive Location Assessment – the MCI Approach. *South East European Journal of Economics and Business* 7(2): 35-49. [10.2478/v10033-012-0013-7](https://doi.org/10.2478/v10033-012-0013-7)

[8] Suárez-Vega R, Gutiérrez-Acuña JL, Rodríguez-Díaz M (2015) Locating a supermarket using a locally calibrated Huff model. *International Journal of Geographical Information Science* 29(2): 217–233. [10.1080/13658816.2014.958154](https://doi.org/10.1080/13658816.2014.958154)

[9] Wieland T (2015) *Räumliches Einkaufsverhalten und Standortpolitik im Einzelhandel unter Berücksichtigung von Agglomerationseffekten - Theoretische Erklärungsansätze, modellanalytische Zugänge und eine empirisch-ökonometrische Marktgebietsanalyse anhand eines Fallbeispiels aus dem ländlichen Raum Ostwestfalens/Südniedersachsens*. Geographische Handelsforschung 23. Mannheim: MetaGIS. https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753

[10] Huff DL, McCallum BM (2008) Calibrating the Huff Model using ArcGIS Business Analyst. ESRI White Paper, September 2008. https://www.esri.com/library/whitepapers/pdfs/calibrating-huff-model.pdf