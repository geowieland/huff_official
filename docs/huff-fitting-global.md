# Huff Model: Global Optimization

## Background

The weighting parameters in the [Huff Model](huff-model.md) have a strong influence on the model results. In many studies, it is required that these parameters are estimated empirically in order to calibrate the model. In his seminal study, Huff[1] proposed an algorithm for iteratively finding the best $$\lambda$$ weighting parameter for travel time, on condition that empircal market shares of supply locations are available. Since then, several researchers developed and used nonlinear iterative fitting approaches for the global optimization of the Huff Model, especially when no empirical customer-store interactions are available, but only total customers or total turnover of the supply locations[2][3][4][5][6][7][8]. 

Nonlinear optimization of the Huff Model is particularly useful or necessary when two conditions coincide: 1) no empirical origin-destination interactions ($$E_{ij}$$) or regional market shares ($$p_{ij}$$) are available, and 2) different [weighting functions](huff-model.md#weighting-functions) (rather than power functions) are being evaluated, where it is also possible that different weighting functions are included in the final model[3][5][6][8]. Only the case in which the Huff Model is fitted to empirical total values ​​($$T_j$$) is addressed here.

## Maximum Likelihood Estimation with aggregated data

Orpana and Lampinen[3] proposed a workflow for a [Maximum Likelihood Estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) of an extended Huff Model in cases where no empirical $$p_{ij}$$ are available but only total turnovers $$T_j$$ of all $$J$$ locations (in this case: grocery stores). Their objective function is the *negative log-likelihood* which is to be minimized:

$$L = \sum^J_{j=1} || \ln T_{j_{obs}} - \ln T_{j_{exp}} ||^2$$

where $$T_{j_{obs}}$$ is the actual turnover of location $$j$$, and $$T_{j_{exp}}$$ is the expected turnover of location $$j$$.

The optimization algorithm is not specified a priori. Orpana and Lampinen[3] employ a [sequential quadratic programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) approach. However, other optimization methods are also conceivable; for the researcher, these differ primarily in terms of computation time and the ability to define bounds and constraints[9].

## Further notes

Some studies employ highly specialized operations research techniques for optimization, such as [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)[4][8].

If empirical market shares $$p_{ij}$$ are available, calibrating the Huff Model is possible using the [Multiplicative Competitive Interaction (MCI) Model](mci-model.md).


## References

[1] Huff DL (1962) *Determination of Intra-Urban Retail Trade Areas*. Real Estate Research Program, Graduate Schools of Business Administration, University of California.

[2] Haines Jr GH, Simon LS, Alexis M (1972) Maximum Likelihood Estimation of Central-City Food Trading Areas. *Journal of Marketing Research* 9(2): 154-159. [10.2307/3149948](https://doi.org/10.2307/3149948)

[3] Orpana T, Lampinen J (2003) Building Spatial Choice Models from Aggregate Data. *Journal of Regional Science* 43(2): 319-348. [10.1111/1467-9787.00301](https://doi.org/10.1111/1467-9787.00301)

[4] De Beule M, Van den Poel D, Van de Weghe N (2014) An extended Huff-model for robustly benchmarking and predicting retail network performance. *Applied Geography* 46(1): 80–89. [10.1016/j.apgeog.2013.09.026](https://doi.org/10.1016/j.apgeog.2013.09.026)

[5] Kapitza J, Wieland T, Metzler M (2026) Modeling hospital catchment areas in pediatric oncology using an empirically parameterized extended Huff model. *International Journal of Health Geographics*. [10.1186/s12942-026-00478-2](https://doi.org/10.1186/s12942-026-00478-2)

[6] Li Y, Liu L (2012) Assessing the impact of retail location on store performance: A comparison of Wal-Mart and Kmart stores in Cincinnati. *Applied Geography* 32(2): 591-600. [10.1016/j.apgeog.2011.07.006]https://doi.org/10.1016/j.apgeog.2011.07.006

[7] Liang Y, Gao S, Cai Y, Foutz NZ, Wu L (2020) Calibrating the dynamic Huff model for business analysis using location big data. *Transactions in GIS* 24(3): 681-703. [10.1111/tgis.12624](https://doi.org/10.1111/tgis.12624)

[8] Mensing M (2018) Lebensmittel-Onlinehandel - Alternative zur zukünftigen Versorgung der Bevölkerung ländlicher Räume? PhD thesis, Rheinisch-Westfälische Technische Hochschule Aachen. https://publications.rwth-aachen.de/record/758124

[9] Nocedal J, Wright SJ (2006) Numerical Optimization. New York: Springer. [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)