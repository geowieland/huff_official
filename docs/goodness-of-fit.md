# Goodness-of-Fit Metrics

## Background

Goodness-of-fit metrics are used to evaluate how well a model's predicted values match the observed values. They quantify different aspects of model performance, including the proportion of explained variance, the magnitude of prediction errors, relative deviations, and the distribution of errors across observations[1][2]. The model fit indicators are relevant for all empirical parametrization approaches used in market area models, namely, the [MCI Model](mci-model.md), as well as the [global](huff-fitting-global.md) and [local calibration of the Huff Model](huff-fitting-local.md).

Throughout the following definitions, the notation is:

- $$y_i$$: Observed value of observation $$i$$
- $$\hat{y}_i$$: Expected (predicted) value of observation $$i$$
- $$e_i = y_i - \hat{y}_i$$: Residual (prediction error)
- $$\bar{y}$$: Mean of the observed values
- $$n$$: Number of valid observed–expected pairs
- $$p$$: Percentage scaling factor (typically 100)

The following list and the formulas are based on academic literature from the fields of econometrics and machine learning[1][2]:

## Variance-Based Metrics

Variance-based metrics evaluate the extent to which a model explains the variability in the observed data. They compare the residual variation against the total variation of the observations.

### Coefficient of Determination (R-squared, R²)

The coefficient of determination measures the proportion of the total variance in the observed data that is explained by the model. Values closer to 1 indicate a better fit, whereas a value of 0 indicates that the model performs no better than predicting the mean of the observations. Negative values are possible and indicate that the model performs worse than this mean-based reference prediction.

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2} {\sum_{i=1}^{n}(y_i-\bar{y})^2}$$

**Interpretation:** Higher values indicate better model performance. The theoretical maximum is 1, while negative values are unbounded.

For R-squared, there is a significance test (F-test) that examines whether the empirical $$R^2$$ is significantly greater than that of a model without predictors.

## Deviation-Based Metrics

Deviation-based metrics quantify the magnitude of the differences between observed and predicted values. They are expressed in the original unit of measurement (except for squared error metrics) and provide a direct assessment of prediction accuracy.

### Sum of Squared Residuals (SSR)

The sum of squared residuals represents the total squared difference between observed and predicted values. Squaring the residuals gives larger errors disproportionately more weight, making this metric particularly sensitive to outliers. Lower values indicate a better fit.

$$SSR = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

**Interpretation:** Values range from 0 to infinity, with 0 representing a perfect fit.

### Sum of Absolute Residuals (SAR)

The sum of absolute residuals measures the total magnitude of prediction errors without considering their direction. Unlike squared residuals, absolute residuals do not disproportionately penalize large errors. Lower values indicate a better fit.

$$SAR = \sum_{i=1}^{n}|y_i-\hat{y}_i|$$

**Interpretation:** Values range from 0 to infinity, with 0 representing a perfect fit.

### Mean Squared Error (MSE)

The mean squared error calculates the average squared difference between observed and predicted values. Due to the squaring operation, large prediction errors receive a disproportionately high weight. Lower values indicate better model performance.

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

**Interpretation:** Values range from 0 to infinity, with 0 indicating a perfect fit. The metric is expressed in squared units of the observed variable.

### Root Mean Squared Error (RMSE)

The root mean squared error is the square root of the mean squared error and therefore expresses prediction errors in the original unit of measurement. Similar to MSE, it penalizes larger errors more strongly than smaller ones. Lower values indicate better model performance.

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

**Interpretation:** Values range from 0 to infinity, with 0 indicating a perfect fit.

### Mean Absolute Error (MAE)

The mean absolute error represents the average absolute difference between observed and predicted values. It provides an intuitive measure of the typical prediction error in the original unit of measurement and is less sensitive to outliers than MSE or RMSE. Lower values indicate better model performance.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

**Interpretation:** Values range from 0 to infinity, with 0 indicating a perfect fit.

## Relative and Percentage-Based Metrics

Relative and percentage-based metrics express prediction errors in relation to the magnitude of the observed or predicted values. They facilitate comparisons between datasets or variables with different scales and units.

### Mean Absolute Percentage Error (MAPE)

The mean absolute percentage error measures the average absolute prediction error relative to the observed values, expressed as a percentage. It provides a scale-independent measure of prediction accuracy. However, MAPE is undefined for observations with an observed value of zero and can become disproportionately large when observed values are close to zero.

$$MAPE = \frac{p}{n^*} \sum_{i \in I^*} \left|\frac{y_i-\hat{y}_i}{y_i}\right|$$

where $$I^*$$ is the set of observations for which $$y_i \neq 0$$, and $$n^*$$ is the number of such observations.

**Interpretation:** Lower values indicate better model performance, with 0 representing a perfect fit. The metric is expressed as a percentage when \(p=100\).

Observations with zero observed values must be excluded from the MAPE calculation.

### Symmetric Mean Absolute Percentage Error (sMAPE)

The symmetric mean absolute percentage error measures the absolute prediction error relative to the average magnitude of the observed and predicted values. By considering both values in the denominator, it provides a more symmetric measure than MAPE. Lower values indicate better agreement between observed and predicted values.

$$sMAPE = \frac{1}{n}\sum_{i=1}^{n} \frac{|y_i-\hat{y}_i|} {(|y_i|+|\hat{y}_i|)/2} \cdot p$$

**Interpretation:** With $$p=100$$, values range from 0% to 200%, where 0% indicates a perfect fit. If both observed and predicted values are zero, the implementation assigns an error of zero.

### Absolute Percentage Error (APE)

The absolute percentage error represents the absolute deviation of an individual prediction relative to its observed value. It is calculated for each observation and can be used to examine the distribution of prediction errors across individual data points. Lower values indicate more accurate predictions.

$$APE_i = \left|\frac{y_i-\hat{y}_i}{y_i}\right|\cdot p$$

**Interpretation:** Values are non-negative, with 0% indicating a perfect prediction. APE is undefined for observed values equal to zero and is therefore excluded for these observations in the implementation.

APE threshold statistics describe the proportion of predictions whose absolute percentage error falls below a predefined threshold. They provide an intuitive way to assess how frequently a model achieves a specified level of relative prediction accuracy. Higher percentages indicate better performance.

For a threshold $$t$$, the statistic is calculated as:

$$APE_{<t} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(APE_i<t)\cdot p$$

where $$\mathbb{1}(\cdot)$$ is the indicator function, which equals 1 if the condition is satisfied and 0 otherwise.

**Interpretation:** This calculation allows us to answer the following question: *For what percentage of $$N$$ empirical observations is the deviation in the model at most $$t$$ percent?* Values range from 0% to 100% when $$p=100$$. Higher values indicate that a larger proportion of predictions fall below the specified percentage-error threshold.

Observations with zero observed values must be excluded from the APE calculation.

### Symmetric Absolute Percentage Error (sAPE)

The symmetric absolute percentage error quantifies the relative deviation between an individual observed and predicted value using the average absolute magnitude of both values as the denominator. Unlike APE, it treats observed and predicted values symmetrically and can accommodate negative values. Lower values indicate better agreement.

$$sAPE_i = 
\begin{cases}
0, & |y_i|+|\hat{y}_i|=0\\
\frac{|y_i-\hat{y}_i|}
{(|y_i|+|\hat{y}_i|)/2}\cdot p, & \text{otherwise}
\end{cases}
$$

**Interpretation:** With $$p=100$$, values range from 0% to 200%, where 0% indicates a perfect prediction.

## Further notes

Whether one or more indicators are used — and, if so, which ones — depends on the purpose of the analysis: for instance, if an analysis is conducted primarily to test hypotheses regarding the impact of store-related or location-based variables, then the extent to which variance is explained and the model's predictive capability are of secondary importance. For machine learning models in particular — those designed for forecasting rather than analysis, and which do not estimate interpretable parameters — the ability to produce accurate forecasts is of great importance. [R-squared](#coefficient-of-determination-r-squared-r) is the standard indicator for assessing the proportion of variance explained in linear regression models. However, it provides little insight into a model's predictive capability, which is why other metrics, such as [RMSE](#root-mean-squared-error-rmse) or [MAPE](#mean-absolute-percentage-error-mape), are preferred in the context of machine learning[1]. 

In some studies where the Huff or MCI Model are fitted to real-world data, the MAPE is used as the main indicator for finding the best-fitting model[3][4][5]. In other studies, which primarily focus on determining parameters for the purpose of testing hypotheses — for example, using the MCI model with [Ordinary Least Squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares) estimation — R-squared is used instead[6][7][8][9][10]. However, it can also be very useful to include several of the aforementioned indicators when comparing different model variants[11][12].


## References

[1] Boehmke B, Greenwell B (2020) 
*Hands-On Machine Learning with R*. 1 ed. 
New York, NY: Taylor & Francis. [10.1201/9780367816377](https://doi.org/10.1201/9780367816377)

[2] Buturac G (2022)
Measurement of Economic Forecast Accuracy: A Systematic Overview of the Empirical Literature. 
*Journal of Risk and Financial Management* 15(1), 1. [10.3390/jrfm15010001](https://doi.org/10.3390/jrfm15010001)

[3] De Beule M, Van den Poel D, Van de Weghe N (2014) An extended Huff-model for robustly benchmarking and predicting retail network performance. *Applied Geography* 46(1): 80–89. [10.1016/j.apgeog.2013.09.026](https://doi.org/10.1016/j.apgeog.2013.09.026)

[4] Kapitza J, Wieland T, Metzler M (2026) Modeling hospital catchment areas in pediatric oncology using an empirically parameterized extended Huff model. *International Journal of Health Geographics*. [10.1186/s12942-026-00478-2](https://doi.org/10.1186/s12942-026-00478-2)

[5] Latruwe T, Van der Wee M, Vanleenhove P, Michielsen K, Verbrugge S, Colle D (2023) Improving inpatient and daycare admission estimates with gravity models. *Health Services and Outcomes Research Methodology* 23, 452–467. [10.1007/s10742-022-00298-4](https://doi.org/10.1007/s10742-022-00298-4)

[6] Kubis A, Hartmann M (2007)
Analysis of Location of Large-area Shopping Centres. A Probabilistic Gravity Model for the Halle–Leipzig Area.
*Jahrbuch für Regionalwissenschaft* 27(1), 43–57. [10.1007/s10037-006-0010-3](https://doi.org/10.1007/s10037-006-0010-3)

[7] Li Y, Liu L (2012) Assessing the impact of retail location on store performance: A comparison of Wal-Mart and Kmart stores in Cincinnati. *Applied Geography* 32(2): 591-600. [10.1016/j.apgeog.2011.07.006]https://doi.org/10.1016/j.apgeog.2011.07.006

[8] Oruc N, Tihi B (2013) Competitive Location Assessment – the MCI Approach. *South East European Journal of Economics and Business* 7(2): 35-49. [10.2478/v10033-012-0013-7](https://doi.org/10.2478/v10033-012-0013-7)

[9] Suárez-Vega R, Gutiérrez-Acuña JL, Rodríguez-Díaz M (2015) Locating a supermarket using a locally calibrated Huff model. *International Journal of Geographical Information Science* 29(2): 217–233. [10.1080/13658816.2014.958154](https://doi.org/10.1080/13658816.2014.958154)

[10] Wieland T (2015) *Räumliches Einkaufsverhalten und Standortpolitik im Einzelhandel unter Berücksichtigung von Agglomerationseffekten - Theoretische Erklärungsansätze, modellanalytische Zugänge und eine empirisch-ökonometrische Marktgebietsanalyse anhand eines Fallbeispiels aus dem ländlichen Raum Ostwestfalens/Südniedersachsens*. Geographische Handelsforschung 23. Mannheim: MetaGIS. https://nbn-resolving.org/urn:nbn:de:bvb:20-opus-180753

[11] Baviera-Puig A, Buitrago-Vera J, Escriba-Perez C (2016) Geomarketing models in supermarket location strategies. *Journal of Business Economics and Management* 17(6): 1205–1221. [10.3846/16111699.2015.1113198](https://doi.org/10.3846/16111699.2015.1113198)

[12] Wieland T (2017) Market Area Analysis for Retail and Service Locations with MCI. *R Journal* 9(1): 298-323. [10.32614/RJ-2017-020](https://doi.org/10.32614/RJ-2017-020)