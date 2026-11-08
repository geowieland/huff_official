# Machine Learning-Based Estimation of Market Areas

## Background

While the empirical parameterization of the [Huff](huff-fitting-global.md) or [MCI Model](mci-model.md) typically uses empirically determined market shares as the dependent variable, there are also a number of other studies that instead model empirical customer or expenditure flows directly, employing various modeling approaches (e.g., count data models)[1][2][3]. This procedure can be generalized, as a wide range of mathematical operations and estimation procedures can be considered. Performing a [market area analysis](glossary.md#market-area-model) using the a machine learning approach as shown below requires observed data on regional customer or expenditure flows, $$E_{ij}$$.

## Concept

As in the MCI Model, we have $$I$$ [customer origins](glossary.md#customer-origins) ($$i = 1, ..., I$$) and $$J$$ [supply locations](glossary.md#supply-locations) ($$j = 1, ..., J$$). The supply locations are characterized by $$H$$ ($$h = 1,2,...,H$$) attributes. The generalized relationship can be represented as follows:

$$\hat{E}_{ij} = f_\mathcal{A} (A_{h_j}; \theta_\mathcal{A})$$

where $$\hat{E}_{ij}$$ equals the estimated customer or expenditure flows from customer origin $$i$$ to supply location $$j$$, $$A_{h_j}$$ is the $$h$$-th characteristic of shopping location $$j$$, $$\theta$$ is a set of given parameters, and $$\mathcal{A}$$ is a given list of various function and estimation types. Both the function and the parameters (if any exist) depend on the chosen method from the set $$\mathcal{A}$$. The travel time from $$i$$ to $$j$$, $$t_{ij}$$, can be included as a variable, though it does not necessarily have to be.

However, such an estimated model cannot yet be meaningfully used for forecasting, because market area models are closed systems consisting of supply and demand locations (see the [Huff Model](huff-model.md) section). If a new supply location is added, the customers' *existing* expenditures are redistributed across all locations. A model such as the one described above would be "open" — meaning that if an inflow of customers is estimated for a new supply location, this would *not* result in a corresponding outflow of customers from existing competitors. Therefore, it is necessary to calculate interaction probabilities or local market shares from the estimated customer or expenditure flows:

$$\hat{p}_{ij} = \frac{\hat{E}_{ij}}{\sum_{j=1}^J \hat{E}_{ij}}$$

where $$\hat{p}_{ij}$$ equals the estimated interaction probabilities from customer origin $$i$$ to supply location $$j$$.

The dependent variable $$E_{ij}$$ and the partial utilities $$A_{h_j}$$ may be included in its original form, but may also be transformed. If the variables are transformed via the *log-centering transformation* from the MCI Model[4], the formula shown above serves as a generalized version of the MCI Model, in which, however, the mathematical relationship between the variables and the estimation algorithm are not predetermined. This is because, starting from the same (!) interaction matrix, $$p_{ij}$$ and $$E_{ij}$$ are exactly proportional; therefore, the following holds:

$$\log \left(\frac{p_{ij}}{\widetilde{p}_i} \right) = \log \left(\frac{E_{ij}}{\widetilde{E}_i} \right)$$

where $$\widetilde{p}_i$$ and $$\widetilde{A}_h$$ are the geometric means of $$p_{ij}$$ and $$A_h$$.

Since it is not specified how the relationship should be modeled, various models (see below) can be trained and compared using [goodness-of-fit metrics](goodness-of-fit.md).

## Machine learning algorithms

To develop a model of the aforementioned relationships, one could consider any machine learning algorithm designed for regression problems (i.e., optimizations where the outcome is a continuous variable)[5][6]:

### Ordinary Least Squares (OLS)

Linear regression estimates the relationship between predictor variables and the target variable by minimizing the sum of squared residuals. It provides an interpretable baseline model and assumes a linear relationship between predictors and the target.

### Bagging with Linear Regression

This ensemble method (Bagging = bootstrap aggregating) combines multiple linear regression models trained on different [bootstrap samples](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) of the training data. The aggregated predictions can improve model stability and reduce variance.

### Bagging with Decision Trees

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a learner that inductively learns rules regarding relationships within the data. In the context of bagging, multiple decision tree regressors are trained on bootstrap samples, and their predictions are aggregated. This approach can capture nonlinear relationships while reducing the variance associated with individual decision trees.

### Random Forest (RF)

Random Forest combines multiple decision trees trained on bootstrap samples, using randomly selected subsets of predictors at each split. It captures complex nonlinear relationships and interactions while generally providing robust predictions.

### Gradient Boosting (GB)

Gradient Boosting builds an ensemble of decision trees sequentially, with each new tree attempting to correct the errors of the preceding ensemble. This allows the model to capture complex nonlinear patterns and interactions.

### K-Nearest Neighbors (KNN)

KNN predicts the target value based on the target values of the nearest observations in the predictor space. Predictions are typically calculated as the average of the selected neighbors, making the method particularly dependent on the choice of distance metric and feature scaling.

### Support Vector Regression (SVR)

SVR estimates a regression function by identifying a prediction boundary that allows a specified tolerance for errors. Through kernel functions, it can model nonlinear relationships between predictors and the target variable.

### Extreme Gradient Boosting (XGBoost)

XGBoost is an optimized gradient boosting implementation that combines decision trees sequentially to minimize a regularized loss function. It offers efficient training and supports complex nonlinear relationships and interactions.

### Light Gradient Boosting Machine (LightGBM)

LightGBM is a gradient boosting framework based on decision trees that uses histogram-based learning and leaf-wise tree growth. It is designed for computational efficiency and can handle complex relationships in structured datasets.

### Multilayer Perceptron (MLP)

MLP is a feedforward [artificial neural network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) consisting of multiple interconnected layers of neurons. Through nonlinear activation functions, it can approximate complex relationships between predictors and the target variable.


## References

[1] Fox EJ, Postrel S, McLaughlin A (2007)
The Impact of Retail Location on Retailer Revenues: An Empirical Investigation
https://www.researchgate.net/publication/228637298_The_Impact_of_Retail_Location_on_Retailer_Revenues_An_Empirical_Investigation

[2] Okoruwa A, Nourse H, Terza J (1994) 
Estimating Sales for Retail Centers: An Application of the Poisson Gravity Model. 
*Journal of Real Estate Research* 9(1), 85–97. [10.1080/10835547.1994.12090741](https://doi.org/10.1080/10835547.1994.12090741)

[3] Wieland T (2018) 
A Hurdle Model Approach of Store Choice and Market Area Analysis in Grocery Retailing. 
*Papers in Applied Geography* 4(4): 370-389. [10.1080/23754931.2018.1519458](https://doi.org/10.1080/23754931.2018.1519458)

[4] Nakanishi M, Cooper LG (1974) Parameter estimation for a Multiplicative Competitive Interaction Model: Least squares approach. *Journal of Marketing Research* 11(3): 303–311. [10.2307/3151146](https://doi.org/10.2307/3151146).

[5] Boehmke B, Greenwell B (2020) 
*Hands-On Machine Learning with R*. 1 ed. 
New York, NY: Taylor & Francis. [10.1201/9780367816377](https://doi.org/10.1201/9780367816377)

[6] Géron A (2025)
*Hands-On Machine Learning with Scikit-Learn and PyTorch: Concepts, Tools, and Techniques to Build Intelligent Systems*.
O'Reilly Media. https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/